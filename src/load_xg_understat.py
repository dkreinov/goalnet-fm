"""Load match xG from understat getLeagueData (EPL 2023/2024/2025) into match.xg_home/xg_away.
Usage: python D:/Programming/claude/FM/src/load_xg_understat.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch

SEASONS = {"2023": "2023-24", "2024": "2024-25", "2025": "2025-26"}
URL = "https://understat.com/getLeagueData/EPL/{y}"

ALIASES = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest", "Wolverhampton Wanderers": "Wolves",
    "Tottenham": "Tottenham", "West Ham": "West Ham", "Newcastle United": "Newcastle",
    "Luton": "Luton", "Ipswich": "Ipswich", "Leicester": "Leicester",
    "Sheffield United": "Sheffield United", "Bournemouth": "Bournemouth",
    "Brighton": "Brighton", "Leeds": "Leeds", "Sunderland": "Sunderland",
}


def main():
    con = db.connect()
    con.execute("PRAGMA busy_timeout=60000")
    for a, c in ALIASES.items():
        db.add_club_alias(con, a, c)
    upd = miss = 0
    for y, label in SEASONS.items():
        data = json.loads(fetch.get(URL.format(y=y), min_delay=1.0,
                                    headers={"X-Requested-With": "XMLHttpRequest"}))
        dates = data.get("dates", [])
        print(f"{label}: {len(dates)} matches from understat")
        for m in dates:
            if not m.get("isResult"):
                continue
            h, a = m["h"]["title"], m["a"]["title"]
            date = m["datetime"][:10]
            hid, aid = db.club_id(con, h), db.club_id(con, a)
            cur = con.execute(
                """UPDATE match SET xg_home=?, xg_away=?
                   WHERE home_club_id=? AND away_club_id=?
                   AND date(match_date) BETWEEN date(?, '-1 day') AND date(?, '+1 day')""",
                (float(m["xG"]["h"]), float(m["xG"]["a"]), hid, aid, date, date))
            if cur.rowcount:
                upd += 1
            else:
                miss += 1
                print(f"  MISS {date} {h} v {a}")
        con.commit()
    db.log(con, "understat", "", "ok", f"xg updated={upd} missed={miss}")
    print(f"updated={upd} missed={miss}")
    con.close()


if __name__ == "__main__":
    main()
