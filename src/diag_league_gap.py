"""Why is a specific INCLUDED league low on true-11v11? Replays build_dataset's exact per-starter
resolution (xwalk -> ESPN-collision -> name-fallback -> roster_high; roster_medium excluded like the
readiness metric) for the given leagues, and buckets the UNCOVERED starters into:
  - LINKED_NO_GRADE : we know the FM-UID but have no grade snapshot anywhere  -> SCRAPE lever
  - IN_FM_UNMATCHED : a same-name FM grade player exists but we didn't link   -> MATCH lever
  - ABSENT          : no same-name FM grade anywhere                          -> SCRAPE / not-in-FM
Plus per-starter coverage % per league.

Usage: python D:/Programming/claude/FM/src/diag_league_gap.py "Brazil Serie A" "Saudi Pro League" "Portugal Primeira Liga"
"""
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as bd
from build_xwalk import xnorm

LEAGUES = sys.argv[1:] or ["Brazil Serie A", "Saudi Pro League", "Portugal Primeira Liga"]


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

    # uid -> has any grade; and same-xnorm-name FM grade existence
    fm_src = [r[0] for r in con.execute("SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    gp = set(r[0] for r in con.execute("SELECT DISTINCT player_id FROM player_snapshot"))
    uid_pid = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute("SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_pid[uid].add(pid)
    uid_has_grade = set(u for u, p in uid_pid.items() if p & gp)
    fm_names = set()
    for pid in gp:
        n = xnorm(rawname.get(pid, ""))
        if n:
            fm_names.add(n)

    # matches per league
    by_comp = defaultdict(list)
    for mid, comp, season, hc, ac in con.execute(
            """SELECT m.match_id, co.name, s.label, m.home_club_id, m.away_club_id
               FROM match m JOIN competition co ON co.competition_id=m.competition_id
               JOIN season s ON s.season_id=m.season_id WHERE m.home_goals IS NOT NULL"""):
        if comp in LEAGUES:
            by_comp[comp].append((mid, season, hc, ac))

    lineups = defaultdict(list)
    for mid, pid, cid, pos in con.execute("SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        lineups[(mid, cid)].append((pid, pos))

    def resolve_starter(mid, pid, cid, season):
        target_fmv = sfmv.get(season)
        season_end = bd.SEASON_END.get(season, "2026-06-30")
        gp_ = xwalk.get(pid)
        if gp_:
            union = []
            for p in gp_[0]:
                union.extend(snaps.get(p, []))
            union.sort()
            if bd.pick_snapshot(union, target_fmv, season_end):
                return "cov"
        if pid in collisions:
            bridged = set()
            for ec in (cid,):
                bridged |= eclub_to_g.get(ec, set())
            cand = [gps for u, gps in collisions[pid].items()
                    if bridged and any(gpid_clubs.get(g, set()) & bridged for g in gps)]
            if len(cand) == 1:
                union = []
                for p in cand[0]:
                    union.extend(snaps.get(p, []))
                union.sort()
                if bd.pick_snapshot(union, target_fmv, season_end):
                    return "cov"
        r = bd.resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
        if r and bd.pick_snapshot(snaps.get(r, []), target_fmv, season_end):
            return "cov"
        rl = roster.get((mid, pid))
        if rl and rl[1] == "high":
            union = []
            for p in ruid_pids.get(rl[0], ()):
                union.extend(snaps.get(p, []))
            union.sort()
            if bd.pick_snapshot(union, target_fmv, season_end):
                return "cov"
        # uncovered -> classify cause
        uids = set()
        if gp_ is None and pid in collisions:
            uids = set(collisions[pid])
        # uid via xwalk espn link
        nm = xnorm(rawname.get(pid, ""))
        if nm in fm_names:
            return "IN_FM_UNMATCHED"
        return "ABSENT"

    for comp in LEAGUES:
        tot = cov = 0
        causes = Counter()
        for mid, season, hc, ac in by_comp.get(comp, []):
            for cid in (hc, ac):
                for pid, pos in lineups.get((mid, cid), []):
                    tot += 1
                    res = resolve_starter(mid, pid, cid, season)
                    if res == "cov":
                        cov += 1
                    else:
                        causes[res] += 1
        print(f"\n=== {comp} ===  per-starter coverage {100*cov/max(tot,1):.1f}% ({cov}/{tot})")
        for k, v in causes.most_common():
            print(f"    {v:>6,} ({100*v/max(tot,1):.1f}%)  {k}")
    con.close()


if __name__ == "__main__":
    main()
