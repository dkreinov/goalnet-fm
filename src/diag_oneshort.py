"""How many not-ready matches are missing just ONE starter, and how concentrated are the culprits?
Replays build_dataset resolution (xwalk->collision->name-fallback->roster_high; roster_medium excluded
like the readiness metric) over INCLUDED leagues, counts uncovered starters per match-side, and reports:
  - match-sides that are 1-short (exactly 1 ungraded of 11), 2-short, 3+; and matches that are 1 total short
  - distinct players responsible, and the top blockers (how many not-ready match-sides each one alone blocks)
Usage: python D:/Programming/claude/FM/src/diag_oneshort.py
"""
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as bd
from build_xwalk import xnorm

EXCL = {"China Super League", "Ecuador LigaPro", "India Super League", "Paraguay Primera Division",
        "Peru Liga 1", "South Africa Premiership", "Israel Ligat haAl", "Japan J1 League", "Colombia Primera A"}


def main():
    con = db.connect()
    snaps = bd.load_snapshots(con)
    idx, has_snap = bd.name_index(con)
    bd.build_fallback(idx, con)
    xwalk, collisions = bd.load_xwalk(con)
    eclub_to_g, gpid_clubs = bd.load_espn_bridge(con, xwalk)
    roster, ruid_pids = bd.load_roster(con)
    sfmv = bd.season_fmv(con)
    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}
    rawname = {r[0]: r[1] for r in con.execute("SELECT player_id, name FROM player")}

    def covered(mid, pid, cid, season):
        target_fmv = sfmv.get(season); season_end = bd.SEASON_END.get(season, "2026-06-30")
        g = xwalk.get(pid)
        if g:
            u = []
            for p in g[0]:
                u.extend(snaps.get(p, []))
            u.sort()
            if bd.pick_snapshot(u, target_fmv, season_end):
                return True
        if pid in collisions:
            bridged = eclub_to_g.get(cid, set())
            cand = [gps for uu, gps in collisions[pid].items()
                    if bridged and any(gpid_clubs.get(g2, set()) & bridged for g2 in gps)]
            if len(cand) == 1:
                u = []
                for p in cand[0]:
                    u.extend(snaps.get(p, []))
                u.sort()
                if bd.pick_snapshot(u, target_fmv, season_end):
                    return True
        r = bd.resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
        if r and bd.pick_snapshot(snaps.get(r, []), target_fmv, season_end):
            return True
        rl = roster.get((mid, pid))
        if rl and rl[1] == "high":
            u = []
            for p in ruid_pids.get(rl[0], ()):
                u.extend(snaps.get(p, []))
            u.sort()
            if bd.pick_snapshot(u, target_fmv, season_end):
                return True
        return False

    meta = {}
    for mid, comp, season in con.execute(
            """SELECT m.match_id, co.name, s.label FROM match m JOIN competition co ON co.competition_id=m.competition_id
               JOIN season s ON s.season_id=m.season_id WHERE m.home_goals IS NOT NULL"""):
        if comp not in EXCL:
            meta[mid] = season
    lineups = defaultdict(list)
    for mid, pid, cid, pos in con.execute("SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        if mid in meta:
            lineups[(mid, cid)].append((pid, cid))

    side_short = Counter()         # how many uncovered on a side (only sides with >=11 starters)
    match_missing = defaultdict(int)
    blocker = Counter()            # player -> # match-sides where they are an uncovered starter
    oneshort_blocker = Counter()   # player -> # sides where they are THE single missing one
    match_clubs = defaultdict(list)
    for (mid, cid), pls in lineups.items():
        match_clubs[mid].append((cid, pls))
    for mid, sides in match_clubs.items():
        season = meta[mid]
        tot_missing = 0
        for cid, pls in sides:
            if len(pls) < 11:
                continue
            miss = [pid for pid, c in pls if not covered(mid, pid, c, season)]
            n = len(miss)
            side_short[min(n, 5)] += 1
            tot_missing += n
            for pid in miss:
                blocker[pid] += 1
            if n == 1:
                oneshort_blocker[miss[0]] += 1
        match_missing[mid] = tot_missing

    sides_total = sum(side_short.values())
    print(f"INCLUDED match-SIDES (>=11 starters): {sides_total:,}")
    for k in sorted(side_short):
        lbl = f"{k} missing" if k < 5 else "5+ missing"
        print(f"   {side_short[k]:>6,} sides ({100*side_short[k]/sides_total:.0f}%)  {lbl}")
    n_ready = sum(1 for m, v in match_missing.items() if v == 0)
    n_1 = sum(1 for m, v in match_missing.items() if v == 1)
    n_2 = sum(1 for m, v in match_missing.items() if v == 2)
    nm = len(match_missing)
    print(f"\nINCLUDED matches: {nm:,};  ready(0 missing) {n_ready:,} ({100*n_ready/nm:.0f}%);  "
          f"exactly 1 total missing {n_1:,} ({100*n_1/nm:.0f}%);  exactly 2 missing {n_2:,}")
    print(f"\ndistinct players that are an uncovered starter in >=1 not-ready INCLUDED side: {len(blocker):,}")
    print(f"distinct players who are THE single missing starter on a 1-short side: {len(oneshort_blocker):,}")
    print(f"  total 1-short sides those players block: {sum(oneshort_blocker.values()):,}")
    print("  top single-missing blockers (sides they alone block):")
    for pid, n in oneshort_blocker.most_common(15):
        print(f"     {n:>4}  {rawname.get(pid)}")

    # --- split the 1-short blockers: matchable-now (in FM) vs needs-scrape (absent), weighted by sides ---
    fm_src = [r[0] for r in con.execute("SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    gp = set(r[0] for r in con.execute("SELECT DISTINCT player_id FROM player_snapshot"))
    uid_pid = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute("SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_pid[uid].add(pid)
    gp_club = set(p for p, c in con.execute("SELECT player_id, club_id FROM player_snapshot WHERE club_id IS NOT NULL"))
    gp_name = {p: xnorm(rawname.get(p, "")) for p in gp}
    name_uids = defaultdict(set); tok_uids = defaultdict(set)
    for uid, pids in uid_pid.items():
        for p in pids:
            n = gp_name.get(p, "")
            if n:
                name_uids[n].add(uid)
                for t in n.split():
                    if len(t) >= 4:
                        tok_uids[t].add(uid)
    uid_hasclub = {uid: any(p in gp_club for p in pids) for uid, pids in uid_pid.items()}
    cls = Counter(); cls_sides = Counter()
    for pid, sides in oneshort_blocker.items():
        nm = xnorm(rawname.get(pid, ""))
        cands = name_uids.get(nm, set())
        if cands:
            if len(cands) == 1:
                b = "MATCHABLE: in FM, unique name"
            elif any(uid_hasclub.get(u) for u in cands):
                b = "MATCHABLE: in FM, multi-cand but >=1 has club (disambiguable)"
            else:
                b = "HARD: in FM, multi-cand, none has club"
        else:
            tc = set()
            for t in nm.split():
                if len(t) >= 4:
                    tc |= tok_uids.get(t, set())
            b = "MATCHABLE(token): in FM via shared token" if tc else "SCRAPE: absent from FM"
        cls[b] += 1; cls_sides[b] += sides
    print("\n1-SHORT BLOCKERS split (players / 1-short sides they block):")
    for b in sorted(cls_sides, key=lambda k: -cls_sides[k]):
        print(f"   {cls[b]:>5} players / {cls_sides[b]:>6,} sides  -> {b}")
    match_recov = sum(v for k, v in cls_sides.items() if k.startswith("MATCHABLE"))
    print(f"\n  recoverable WITHOUT scraping (matchable): {match_recov:,} of {sum(cls_sides.values()):,} one-short sides")
    con.close()


if __name__ == "__main__":
    main()
