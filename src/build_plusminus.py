"""Leakage-free plus-minus player ratings from the DB (Phase 5 Arm P).

Chronological single pass over ALL scored DB matches (ORDER BY match_date, match_id). For each
match the features are emitted from accumulator state BEFORE the match, then accumulators update:
  - on-pitch intervals from match_sub in/out minutes (starters [0,T], subs [t,T], outs end at t);
    T = 120 when extra-time events exist, else 90.
  - goal attribution: segment-level (each goal at minute t gives +1/-1 to players on pitch) when
    the match's goal events reconstruct the final score exactly; otherwise minutes-weighted share
    of the full-match goal difference.
  - rating = on-pitch GD/90 net of club GD/90 (club accumulated over ALL its matches, played or
    not — rotation/absence is where the signal lives), shrunk pm*n90/(n90+K), K=20.

Outputs (per plan plans/goalnet-ablation-phase-5-architecture-plan.md Step 4):
  data/ctx_pm.npz     {mids, feats:[pm_team_diff, pm_cov]}            for all players_imp matches
  data/players_pm.npz {mids, PMh (N,11,2), PMa (N,11,2)} channels [pm_shrunk, has_pm], slot-aligned
    to players_imp.npz. Alignment: lineup query order + stable role sort reproduces the builder's
    slot order EXACTLY for sides with no imputed slot and exactly 11 starters (~83%); sides with an
    imputed slot may permute pm within a same-role group (accepted, documented). Hard assert: the
    role sequence must equal Rh/Ra for every match; failures emit has_pm=0 and are counted.

Usage: python D:/Programming/claude/FM/src/build_plusminus.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as bd

ROLE = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}
K90 = 20.0                       # shrinkage prior strength in 90s
GOAL_TYPES = ("Goal", "Goal - Header", "Goal - Volley", "Goal - Free-kick", "Penalty - Scored")


def main():
    con = db.connect()
    z = np.load(db.ROOT / "data" / "players_imp.npz")
    nmids, Rh, Ra = z["mids"], z["Rh"], z["Ra"]              # materialize (lazy-NpzFile gotcha)
    nmids = np.asarray(nmids); Rh = np.asarray(Rh); Ra = np.asarray(Ra)
    nidx = {int(m): i for i, m in enumerate(nmids)}
    N = len(nmids)

    matches = con.execute(
        """SELECT match_id, match_date, home_club_id, away_club_id, home_goals, away_goals
           FROM match WHERE home_goals IS NOT NULL ORDER BY match_date, match_id""").fetchall()

    # lineups: SAME SQL text as build_player_dataset_imp so row order matches its defaultdict fill
    lineups = defaultdict(list)
    for mid, pid, cid, pos in con.execute(
            "SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        lineups[(mid, cid)].append((pid, pos))
    participants = defaultdict(list)                          # all players incl. subs (for accum)
    for mid, pid, cid, started in con.execute(
            "SELECT match_id, player_id, club_id, started FROM match_player"):
        participants[mid].append((pid, cid, started))

    subs = defaultdict(list)
    for mid, minute, ip, op in con.execute(
            "SELECT match_id, minute, in_player_id, out_player_id FROM match_sub WHERE minute IS NOT NULL"):
        subs[mid].append((minute, ip, op))
    goals = defaultdict(list)
    q = ",".join("?" * len(GOAL_TYPES))
    for mid, minute, side, typ in con.execute(
            f"SELECT match_id, minute, team_side, type FROM match_event "
            f"WHERE minute IS NOT NULL AND (type IN ({q}) OR type='Own Goal')", GOAL_TYPES):
        goals[mid].append((minute, side, typ == "Own Goal"))
    et = set(r[0] for r in con.execute(
        "SELECT DISTINCT match_id FROM match_event WHERE type='Start Extra Time'"))
    print(f"loaded: {len(matches):,} matches, subs for {len(subs):,}, goal events for {len(goals):,}",
          flush=True)

    # own-goal side convention check on the full data: convention A = own goal counts AGAINST
    # the event's team_side; B = counts FOR it. Pick whichever reconstructs more finals exactly.
    def recon(mid, hg, ag, own_against):
        h = a = 0
        for _, side, own in goals.get(mid, ()):
            eff = side if not own else (("away" if side == "home" else "home") if own_against else side)
            if eff == "home":
                h += 1
            else:
                a += 1
        return h == hg and a == ag
    okA = okB = tot = 0
    for mid, _, _, _, hg, ag in matches:
        if mid in goals:
            tot += 1; okA += recon(mid, hg, ag, True); okB += recon(mid, hg, ag, False)
    own_against = okA >= okB
    print(f"own-goal convention: against-side={okA:,}/{tot:,} for-side={okB:,}/{tot:,} "
          f"-> using {'against' if own_against else 'for'}; segment-eligible={max(okA, okB):,}", flush=True)

    P_gd = defaultdict(float); P_min = defaultdict(float)     # player accumulators
    C_gd = defaultdict(float); C_min = defaultdict(float)     # club accumulators

    def rating(pid, cid):
        """Shrunk net-of-club on-pitch GD/90 from CURRENT (pre-match) accumulators."""
        if P_min[pid] <= 0:
            return 0.0, 0
        n90 = P_min[pid] / 90.0
        pm = P_gd[pid] / n90                                  # on-pitch GD per 90
        club = C_gd[cid] / (C_min[cid] / 90.0) if C_min[cid] > 0 else 0.0
        return (pm - club) * n90 / (n90 + K90), 1

    PMh = np.zeros((N, 11, 2), np.float32); PMa = np.zeros((N, 11, 2), np.float32)
    ctx = np.zeros((N, 2), np.float32)
    n_align_fail = 0; n_seg = 0; n_fall = 0

    for mid, _, hc, ac, hg, ag in matches:
        i = nidx.get(mid)
        if i is not None:                                     # ---- emit BEFORE updating (no leakage)
            side_pm = {}
            ok_align = True
            for cid, Rrow in ((hc, Rh[i]), (ac, Ra[i])):
                players = [(ROLE.get(bd.POS_GROUP.get((pos or " ")[0], "MID"), 2), pid)
                           for pid, pos in lineups.get((mid, cid), [])]
                players.sort(key=lambda t: t[0])              # stable role sort (see docstring)
                players = players[:11]
                if [r for r, _ in players] != list(Rrow):
                    ok_align = False; break
                side_pm[cid] = [rating(pid, cid) for _, pid in players]
            if ok_align:
                PMh[i] = np.array(side_pm[hc], np.float32)
                PMa[i] = np.array(side_pm[ac], np.float32)
                mh = [v for v, h in side_pm[hc] if h]; ma = [v for v, h in side_pm[ac] if h]
                cov = (sum(h for _, h in side_pm[hc]) + sum(h for _, h in side_pm[ac])) / 22.0
                ctx[i] = [(np.mean(mh) if mh else 0.0) - (np.mean(ma) if ma else 0.0), cov]
            else:
                n_align_fail += 1                             # safe null row (zeros, has_pm=0)

        # ---- update accumulators
        T = 120.0 if mid in et else 90.0
        parts = participants.get(mid)
        C_gd[hc] += hg - ag; C_min[hc] += T
        C_gd[ac] += ag - hg; C_min[ac] += T
        if not parts:
            continue
        iv = {}                                               # pid -> [start, end)
        for pid, cid, started in parts:
            if started:
                iv[pid] = [0.0, T]
        for minute, ip, op in sorted(subs.get(mid, ()), key=lambda t: t[0]):
            t = min(max(float(minute), 0.0), T)
            if op is not None and op in iv:
                iv[op][1] = t
            if ip is not None:
                iv[ip] = [t, T]
        club_of = {pid: cid for pid, cid, _ in parts}
        iv = {pid: se for pid, se in iv.items() if pid in club_of}   # subs absent from match_player
        seg_ok = mid in goals and recon(mid, hg, ag, own_against)
        if seg_ok:
            n_seg += 1
            for minute, side, own in goals[mid]:
                t = min(max(float(minute), 0.0), T)
                eff = side if not own else (("away" if side == "home" else "home") if own_against else side)
                sc_club = hc if eff == "home" else ac
                for pid, (s, e) in iv.items():
                    if s < t <= e or (t == 0.0 and s == 0.0):
                        P_gd[pid] += 1.0 if club_of[pid] == sc_club else -1.0
        else:
            n_fall += 1
            gd_h = float(hg - ag)
            for pid, (s, e) in iv.items():
                frac = max(e - s, 0.0) / T
                P_gd[pid] += (gd_h if club_of[pid] == hc else -gd_h) * frac
        for pid, (s, e) in iv.items():
            P_min[pid] += max(e - s, 0.0)

    print(f"pass done: segment-attributed {n_seg:,}, fallback {n_fall:,}, "
          f"align-fail {n_align_fail:,}/{N:,} npz sides-pairs", flush=True)

    out1 = db.ROOT / "data" / "players_pm.npz"
    out2 = db.ROOT / "data" / "ctx_pm.npz"
    np.savez_compressed(out1, mids=nmids, PMh=PMh, PMa=PMa)
    np.savez_compressed(out2, mids=nmids, feats=ctx)
    cov_all = float(((PMh[:, :, 1] > 0).mean() + (PMa[:, :, 1] > 0).mean()) / 2)
    print(f"saved {out1} and {out2}: slot coverage {cov_all:.1%}, "
          f"ctx pm_diff std {ctx[:, 0].std():.4f}", flush=True)


if __name__ == "__main__":
    main()
