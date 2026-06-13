"""Load match xG from understat getLeagueData for every registered league understat covers,
across all seasons, into match.xg_home/xg_away. Joins by date + club (alias-aware).
Usage: python D:/Programming/claude/FM/src/load_xg_understat.py [league_name ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
import leagues as L
import match_link

URL = "https://understat.com/getLeagueData/{lg}/{year}"


def main():
    con = db.connect()
    names = sys.argv[1:]
    targets = [l for l in (([L.BY_NAME[n] for n in names]) if names else L.enabled())
               if l.get("understat")]
    grand_upd = grand_miss = 0
    for lg in targets:
        comp_id = db.competition_id(con, lg["name"])
        print(f"== {lg['name']} ==")
        for season, year in L.UNDERSTAT_YEAR.items():
            try:
                data = json.loads(fetch.get(URL.format(lg=lg["understat"], year=year),
                                            min_delay=1.0,
                                            headers={"X-Requested-With": "XMLHttpRequest"}))
            except Exception as e:
                db.log(con, "understat", f"{lg['name']} {season}", "error", str(e))
                continue
            upd = miss = 0
            for m in data.get("dates", []):
                if not m.get("isResult"):
                    continue
                h, a = m["h"]["title"], m["a"]["title"]
                date = m["datetime"][:10]
                mid = match_link.find_match(con, comp_id, date,
                                            int(m["goals"]["h"]), int(m["goals"]["a"]), h, a)
                if mid is None:
                    miss += 1
                    continue
                con.execute("UPDATE match SET xg_home=?, xg_away=? WHERE match_id=?",
                            (float(m["xG"]["h"]), float(m["xG"]["a"]), mid))
                upd += 1
            con.commit()
            db.log(con, "understat", f"{lg['name']} {season}", "ok", f"upd={upd} miss={miss}")
            print(f"  {season}: xg updated={upd} missed={miss}")
            grand_upd += upd
            grand_miss += miss
    print(f"TOTAL xg updated={grand_upd} missed={grand_miss}")
    con.close()


if __name__ == "__main__":
    main()
