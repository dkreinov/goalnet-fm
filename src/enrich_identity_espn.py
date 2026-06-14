"""Fetch DOB + nationality + height/weight for every ESPN lineup player from ESPN's athlete
endpoint, into source_identity. Enables a DOB-anchored 1:1 join to FM records. Resumable.
Usage: python D:/Programming/claude/FM/src/enrich_identity_espn.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch

URL = "http://sports.core.api.espn.com/v2/sports/soccer/athletes/{aid}"


def main():
    con = db.connect()
    src = db.source_id(con, "espn")
    todo = [r[0] for r in con.execute(
        """SELECT source_player_id FROM player_source_id
           WHERE source_id=? AND source_player_id NOT IN
             (SELECT source_player_id FROM source_identity WHERE source_id=?)""",
        (src, src))]
    print(f"ESPN athletes to enrich: {len(todo):,}")
    done = err = 0
    for i, aid in enumerate(todo):
        try:
            d = json.loads(fetch.get(URL.format(aid=aid), min_delay=0.25, timeout=30))
            dob = d.get("dateOfBirth")
            dob = dob[:10] if dob else None
            nat = d.get("citizenship")
            con.execute(
                """INSERT OR REPLACE INTO source_identity
                   (source_id, source_player_id, name, dob, nationality, height_cm, weight_kg)
                   VALUES (?,?,?,?,?,?,?)""",
                (src, aid, d.get("displayName"), dob, nat, d.get("height"), d.get("weight")))
            done += 1
        except Exception as e:
            err += 1
            con.execute("""INSERT OR IGNORE INTO source_identity (source_id, source_player_id)
                           VALUES (?,?)""", (src, aid))  # mark seen so we don't retry forever
        if (i + 1) % 200 == 0:
            con.commit()
            print(f"  {i+1}/{len(todo)} done={done} err={err}", flush=True)
    con.commit()
    got = con.execute("SELECT COUNT(*) FROM source_identity WHERE source_id=? AND dob IS NOT NULL",
                      (src,)).fetchone()[0]
    print(f"DONE. enriched={done} err={err}; ESPN players with DOB now: {got:,}")
    con.close()


if __name__ == "__main__":
    main()
