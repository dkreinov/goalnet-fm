"""Like build_player_dataset.py but RELAXES the strict-11v11 rule: keep a side if at least (11 - MAX_IMP)
starters are FM-graded, and impute each missing starter with the role-group mean (GK/DEF/MID/ATT). The
missing starter's ROLE is still known (from match_player.position), so only its attribute vector is filled.
This expands the training set well beyond the 48,355 strict games, to test whether the extra (partly
imputed) data improves evaluation. Writes data/players_imp.npz (+ an `imp` mask of imputed slots).
Usage: python D:/Programming/claude/FM/src/build_player_dataset_imp.py [--max-imp 1]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as bd

ROLE = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}


def main():
    MAX_IMP = int(sys.argv[sys.argv.index("--max-imp") + 1]) if "--max-imp" in sys.argv else 1
    con = db.connect()
    snaps = bd.load_snapshots(con); attrs = bd.load_attrs(con)
    idx, has_snap = bd.name_index(con); bd.build_fallback(idx, con)
    xwalk, collisions = bd.load_xwalk(con)
    eclub_to_g, gpid_clubs = bd.load_espn_bridge(con, xwalk)
    roster, ruid_pids = bd.load_roster(con); sfmv = bd.season_fmv(con)
    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}
    attr_set = set()
    for d in attrs.values():
        for (cat, name) in d:
            attr_set.add(name)
    ATTRS = sorted(attr_set); aidx = {n: i for i, n in enumerate(ATTRS)}; A = len(ATTRS)
    print(f"{A} attributes, max_imp={MAX_IMP}/side", flush=True)

    def snap_for(mid, pid, cid, season):
        target_fmv = sfmv.get(season); season_end = bd.SEASON_END.get(season, "2026-06-30")
        gp = xwalk.get(pid)
        if gp:
            u = []
            for p in gp[0]:
                u.extend(snaps.get(p, []))
            u.sort(); sn = bd.pick_snapshot(u, target_fmv, season_end)
            if sn:
                return sn[1]
        if pid in collisions:
            bridged = eclub_to_g.get(cid, set())
            cand = [gps for uu, gps in collisions[pid].items()
                    if bridged and any(gpid_clubs.get(g2, set()) & bridged for g2 in gps)]
            if len(cand) == 1:
                u = []
                for p in cand[0]:
                    u.extend(snaps.get(p, []))
                u.sort(); sn = bd.pick_snapshot(u, target_fmv, season_end)
                if sn:
                    return sn[1]
        r = bd.resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
        if r:
            sn = bd.pick_snapshot(snaps.get(r, []), target_fmv, season_end)
            if sn:
                return sn[1]
        rl = roster.get((mid, pid))
        if rl and rl[1] == "high":
            u = []
            for p in ruid_pids.get(rl[0], ()):
                u.extend(snaps.get(p, []))
            u.sort(); sn = bd.pick_snapshot(u, target_fmv, season_end)
            if sn:
                return sn[1]
        return None

    def vec(sid):
        v = np.zeros(A, dtype=np.float32)
        for (cat, name), val in attrs.get(sid, {}).items():
            j = aidx.get(name)
            if j is not None:
                v[j] = val
        return v

    matches = con.execute(
        """SELECT m.match_id, m.match_date, m.home_club_id, m.away_club_id, m.home_goals, m.away_goals, s.label
           FROM match m JOIN season s ON s.season_id=m.season_id
           WHERE m.home_goals IS NOT NULL ORDER BY m.match_date""").fetchall()
    lineups = defaultdict(list)
    for mid, pid, cid, pos in con.execute(
            "SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        lineups[(mid, cid)].append((pid, pos))

    # pass 1: collect sides (roles + vec-or-None), keep if <=MAX_IMP missing; accumulate role means
    recs = []          # (Rh, Vh[list of vec|None], Ra, Va, y, date, mid)
    rsum = {r: np.zeros(A, np.float64) for r in range(4)}; rcnt = {r: 0 for r in range(4)}
    kept = 0
    for k, (mid, date, hc, ac, hg, ag, season) in enumerate(matches):
        sides = {}; ok = True
        for cid in (hc, ac):
            xi = lineups.get((mid, cid), [])
            if len(xi) < 11:
                ok = False; break
            players = []      # (role, vec|None)
            for pid, pos in xi:
                role = ROLE.get(bd.POS_GROUP.get((pos or " ")[0], "MID"), 2)
                sid = snap_for(mid, pid, cid, season)
                players.append((role, vec(sid) if sid is not None else None))
            graded = sum(1 for _, v in players if v is not None)
            # need >=11-MAX_IMP graded; cap to 11 keeping graded first
            if graded < 11 - MAX_IMP:
                ok = False; break
            players.sort(key=lambda t: (t[1] is None, t[0]))   # graded first, then by role
            players = players[:11]
            if sum(1 for _, v in players if v is None) > MAX_IMP:
                ok = False; break
            players.sort(key=lambda t: t[0])                   # final order GK->DEF->MID->ATT
            sides[cid] = players
            for role, v in players:
                if v is not None:
                    rsum[role] += v; rcnt[role] += 1
        if not ok:
            continue
        recs.append((sides[hc], sides[ac], 0 if hg > ag else (1 if hg == ag else 2),
                     np.datetime64(date[:10]), mid))
        kept += 1
        if (k + 1) % 10000 == 0:
            print(f"  {k+1:,}/{len(matches):,} scanned, kept={kept:,}", flush=True)

    role_mean = {r: (rsum[r] / rcnt[r]).astype(np.float32) if rcnt[r] else np.zeros(A, np.float32)
                 for r in range(4)}

    def fill(side):
        X = np.stack([(v if v is not None else role_mean[r]) for r, v in side])
        R = np.array([r for r, _ in side], np.int8)
        imp = np.array([v is None for _, v in side], np.int8)
        return X, R, imp

    Xh, Xa, Rh, Ra, Ih, Ia, y, dates, mids = [], [], [], [], [], [], [], [], []
    for sh, sa, yy, dt, mid in recs:
        Xh_, Rh_, Ih_ = fill(sh); Xa_, Ra_, Ia_ = fill(sa)
        Xh.append(Xh_); Rh.append(Rh_); Ih.append(Ih_)
        Xa.append(Xa_); Ra.append(Ra_); Ia.append(Ia_)
        y.append(yy); dates.append(dt); mids.append(mid)

    out = db.ROOT / "data" / "players_imp.npz"
    np.savez_compressed(
        out, Xh=np.stack(Xh).astype(np.float32), Xa=np.stack(Xa).astype(np.float32),
        Rh=np.stack(Rh), Ra=np.stack(Ra), Ih=np.stack(Ih), Ia=np.stack(Ia),
        y=np.array(y, np.int8), dates=np.array(dates), mids=np.array(mids, np.int64), attrs=np.array(ATTRS))
    tot_imp = int(np.stack(Ih).sum() + np.stack(Ia).sum())
    print(f"\nsaved {out}: {kept:,} matches (<= {MAX_IMP} imputed/side), "
          f"{tot_imp:,} imputed starters of {kept*22:,}, X {np.stack(Xh).shape}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
