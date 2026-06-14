"""Compact collection-status report: matches & lineups by league x season, plus FM-grade coverage.
Read-only; safe to run while collectors are active. Usage: python src/status_report.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def main():
    con = db.connect()
    # matches and lineups per (competition, season)
    m = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # comp -> season -> [matches, with_lineups]
    rows = con.execute("""
        SELECT co.name, se.label, m.match_id,
               EXISTS(SELECT 1 FROM match_player mp WHERE mp.match_id=m.match_id) AS has_lu
        FROM match m JOIN competition co ON co.competition_id=m.competition_id
        JOIN season se ON se.season_id=m.season_id""").fetchall()
    kind = {r[0]: None for r in con.execute("SELECT name FROM competition")}
    for r in con.execute("SELECT name, kind FROM competition"):
        kind[r[0]] = r[1]
    for name, season, mid, has in rows:
        cell = m[name][season]
        cell[0] += 1
        if has:
            cell[1] += 1

    print("=" * 96)
    print("COLLECTION STATUS — lineups/matches by league x season")
    print("=" * 96)
    hdr = "LEAGUE".ljust(30) + "".join(s[2:].rjust(11) for s in SEASONS) + "   TOTAL"
    print(hdr)
    league_names = sorted([k for k in m if kind.get(k) != "national"])
    natl_names = sorted([k for k in m if kind.get(k) == "national"])
    grand_m = grand_lu = 0
    for grp, title in ((league_names, "CLUB LEAGUES"), (natl_names, "NATIONAL TEAMS")):
        print(f"\n--- {title} ---")
        for name in grp:
            cells = []
            tm = tl = 0
            for s in SEASONS:
                mm, ll = m[name][s]
                tm += mm; tl += ll
                cells.append(f"{ll}/{mm}" if mm else "·")
            grand_m += tm; grand_lu += tl
            print(name[:29].ljust(30) + "".join(c.rjust(11) for c in cells) + f"   {tl}/{tm}")
    print("-" * 96)
    print(f"{'GRAND TOTAL':30}{'':66}{grand_lu}/{grand_m}  (lineups/matches)")

    # FM grade coverage
    print("\n" + "=" * 60)
    print("FM GRADE SNAPSHOTS by source x edition")
    print("=" * 60)
    for r in con.execute("""SELECT s.name, f.game, COUNT(*) FROM player_snapshot ps
        JOIN source s USING(source_id) LEFT JOIN fm_version f USING(fm_version_id)
        GROUP BY s.name, f.game ORDER BY s.name, f.game"""):
        print(f"  {r[0]:10} {str(r[1]):8} {r[2]:>8,}")
    tot = con.execute("SELECT COUNT(*) FROM player_snapshot").fetchone()[0]
    pa = con.execute("SELECT COUNT(*) FROM player_attribute").fetchone()[0]
    ev = con.execute("SELECT COUNT(*) FROM match_event").fetchone()[0]
    print(f"\n  total snapshots: {tot:,} | attribute values: {pa:,} | match events: {ev:,}")
    con.close()


if __name__ == "__main__":
    main()
