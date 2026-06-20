"""Load the 48 World Cup 2026 squads from the sibling worldcup project and expose them as a normalized
player list for FM-grade matching. Also (with --coverage) reports how many squad players already carry
an FM26 (fm_version_id=3) grade in fm.db, by exact normalized name and by name+club.
Usage: python D:/Programming/claude/FM/src/wc2026_squads.py [--coverage]
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db

WC_TEAMS_DIR = Path(r"D:\Programming\claude\worldcup\team_db\teams")
FM26_FMV = 3


def load_squads():
    """Return list of dicts: {nation, nation_code, name, club, club_country, age, pos, shirt, norm}."""
    out = []
    for f in sorted(glob.glob(str(WC_TEAMS_DIR / "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        nation = d["team"]["name"]
        code = d["team"]["code"]
        for p in d.get("players", []):
            out.append({
                "nation": nation, "nation_code": code,
                "name": p["name"], "norm": db.norm(p["name"]),
                "club": p.get("club"), "club_country": p.get("club_country"),
                "age": p.get("age"), "pos": p.get("position"), "shirt": p.get("shirt_no"),
            })
    return out


def main():
    sq = load_squads()
    print(f"loaded {len(sq)} players across "
          f"{len(set(s['nation'] for s in sq))} teams", flush=True)
    if "--coverage" not in sys.argv:
        return
    con = db.connect()
    rows = con.execute(
        """SELECT p.norm_name, c.norm_name
           FROM player_snapshot s JOIN player p ON p.player_id=s.player_id
           LEFT JOIN club c ON c.club_id=s.club_id
           WHERE s.fm_version_id=?""", (FM26_FMV,)).fetchall()
    fm_names = set(r[0] for r in rows)
    fm_name_club = set((r[0], r[1]) for r in rows)
    hit_name = hit_nc = 0
    miss = []
    for s in sq:
        nclub = db.norm(s["club"]) if s["club"] else None
        if s["norm"] in fm_names:
            hit_name += 1
            if (s["norm"], nclub) in fm_name_club:
                hit_nc += 1
        else:
            miss.append(s)
    n = len(sq)
    print(f"FM26 graded distinct names in db: {len(fm_names):,}")
    print(f"squad players matched by exact norm-name:      {hit_name}/{n} ({hit_name/n*100:.0f}%)")
    print(f"  of those, name+club also matches:            {hit_nc}/{n} ({hit_nc/n*100:.0f}%)")
    print(f"missing (no FM26 grade by name): {len(miss)}")
    # missing by nation
    from collections import Counter
    cm = Counter(s["nation"] for s in miss)
    print("missing-by-nation (top 25):")
    for nat, cnt in cm.most_common(25):
        print(f"  {nat:24s} {cnt}")


if __name__ == "__main__":
    main()
