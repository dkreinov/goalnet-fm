"""Diagnose which WC2022 (competition 9) starters fail to resolve to an FM grade via build_dataset's
real resolution layers (xwalk -> collision-bridge -> name-fallback -> roster), so we can see whether the
34 not-fully-graded matches need name-form fixes, club-disambiguation, or are genuinely uncovered.
Read-only. Usage: python D:/Programming/claude/FM/src/wc2022_diag.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as bd

ROLE = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}


def main():
    con = db.connect()
    snaps = bd.load_snapshots(con)
    idx, has_snap = bd.name_index(con); bd.build_fallback(idx, con)
    xwalk, collisions = bd.load_xwalk(con)
    eclub_to_g, gpid_clubs = bd.load_espn_bridge(con, xwalk)
    roster, ruid_pids = bd.load_roster(con); sfmv = bd.season_fmv(con)
    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}
    cname = {r[0]: r[1] for r in con.execute("SELECT club_id, name FROM club")}

    def snap_for(mid, pid, cid, season):
        target_fmv = sfmv.get(season); season_end = bd.SEASON_END.get(season, "2026-06-30")
        g = xwalk.get(pid)
        if g:
            u = []
            for p in g[0]:
                u.extend(snaps.get(p, []))
            u.sort()
            if bd.pick_snapshot(u, target_fmv, season_end):
                return "xwalk"
        if pid in collisions:
            bridged = eclub_to_g.get(cid, set())
            cand = [gps for uu, gps in collisions[pid].items()
                    if bridged and any(gpid_clubs.get(g2, set()) & bridged for g2 in gps)]
            if len(cand) == 1:
                return "collision"
        r = bd.resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
        if r and bd.pick_snapshot(snaps.get(r, []), target_fmv, season_end):
            return "name"
        rl = roster.get((mid, pid))
        if rl and rl[1] == "high":
            return "roster"
        return None

    matches = con.execute(
        """SELECT m.match_id, m.home_club_id, m.away_club_id, s.label FROM match m
           JOIN season s ON s.season_id=m.season_id WHERE m.competition_id=9""").fetchall()
    lineups = defaultdict(list)
    mids = [m[0] for m in matches]
    for mid, pid, cid, pos in con.execute(
            "SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1 AND match_id IN (%s)"
            % ",".join("?" * len(mids)), mids):
        lineups[(mid, cid)].append((pid, pos))

    full = 0; short = 0; unresolved = defaultdict(int)
    for mid, hc, ac, season in matches:
        ok = True
        for cid in (hc, ac):
            xi = lineups.get((mid, cid), [])
            res = sum(1 for pid, pos in xi if snap_for(mid, pid, cid, season))
            if res < 11:
                ok = False
                for pid, pos in xi:
                    if not snap_for(mid, pid, cid, season):
                        unresolved[(pid, cid)] += 1
        full += ok; short += not ok
    print(f"WC2022: {len(matches)} matches | fully graded {full} | short {short}", flush=True)
    print(f"distinct unresolved starters: {len(unresolved)}", flush=True)
    for (pid, cid), ct in sorted(unresolved.items(), key=lambda x: -x[1])[:40]:
        print(f"  {pname.get(pid,'?'):28s} @ {cname.get(cid,'?'):18s} x{ct}", flush=True)


if __name__ == "__main__":
    main()
