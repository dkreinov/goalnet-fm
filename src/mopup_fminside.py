"""Retry the specific fminside player pages that errored (transient timeouts).
Reads failed /players/ URLs from scrape_log, refetches, parses, saves snapshots.
Usage: python D:/Programming/claude/FM/src/mopup_fminside.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db as dbmod
import fetch
from scrape_fminside import BASE, DBS, parse_player


def main():
    con = dbmod.connect()
    src = dbmod.source_id(con, "fminside", BASE)
    # distinct player URLs that errored and were never saved
    failed = [r[0] for r in con.execute(
        """SELECT DISTINCT url FROM scrape_log
           WHERE source='fminside' AND status='error' AND url LIKE '/players/%'""").fetchall()]
    # drop any that now have a snapshot via their uid
    todo = []
    for pu in failed:
        m = re.search(r"/players/\d+-[^/]+/(\d+)-", pu)
        uid = m.group(1) if m else None
        if uid:
            got = con.execute(
                "SELECT 1 FROM player_source_id WHERE source_id=? AND source_player_id=?",
                (src, uid)).fetchone()
            if got:
                continue
        todo.append((pu, uid))
    print(f"{len(todo)} pages to retry")

    fixed = still_failing = 0
    for pu, uid in todo:
        dbid = int(re.match(r"/players/(\d+)-", pu).group(1))
        cfg = DBS[dbid]
        fmv = dbmod.fm_version_id(con, cfg["game"], cfg["db_version"], cfg["date"])
        try:
            html = fetch.get(BASE + pu, min_delay=3.0, timeout=120)
            p = parse_player(html)
            if not p or not p["attrs"]:
                still_failing += 1
                print(f"  no attrs: {pu}")
                continue
            pid = dbmod.player_id(con, p["name"], src=src, src_player_id=uid or pu)
            cid = dbmod.club_id(con, p["club"]) if p["club"] else None
            sid = dbmod.save_snapshot(
                con, pid=pid, src=src, fmv=fmv, cid=cid, snapshot_date=cfg["date"],
                attrs=p["attrs"],
                meta={k: p[k] for k in ("position", "ca", "pa", "value_eur", "wage_eur",
                                        "foot_left", "foot_right", "height_cm", "weight_kg")})
            con.commit()
            fixed += 1
            print(f"  + {p['name']} ({p['club']})")
        except Exception as e:
            still_failing += 1
            dbmod.log(con, "fminside", pu, "error", f"mopup: {e}")
            print(f"  STILL FAILING {pu}: {str(e)[:80]}")
    # clear resolved error rows so the log reflects reality
    con.execute(
        """DELETE FROM scrape_log WHERE source='fminside' AND status='error' AND url IN (
               SELECT psi.source_player_id FROM player_source_id psi WHERE 1=0)""")
    print(f"DONE fixed={fixed} still_failing={still_failing}")
    con.close()


if __name__ == "__main__":
    main()
