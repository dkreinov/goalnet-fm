"""Stadium capacity per club from Transfermarkt (ESPN has none). Reuses tm_club (our club -> TM id);
fetches each club's TM 'startseite' (home) page and parses the stadium 'N Seats' figure into
tm_club.capacity. Then matches get capacity by joining home_club -> club.name -> tm_club.capacity.
Single writer. Usage: python D:/Programming/claude/FM/src/scrape_capacity.py [--sample N]
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch

HOME = "https://www.transfermarkt.com/x/startseite/verein/{tid}"
HDR = {"Accept-Language": "en"}


def parse_capacity(html):
    m = re.search(r"([\d.]+)\s*Seats", html)
    if not m:
        return None
    try:
        return int(m.group(1).replace(".", "").replace(",", ""))
    except ValueError:
        return None


def main():
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
    con = db.connect()
    con.execute("PRAGMA synchronous=NORMAL")
    cols = [c[1] for c in con.execute("pragma table_info(tm_club)")]
    if "capacity" not in cols:
        con.execute("ALTER TABLE tm_club ADD COLUMN capacity INTEGER")
    todo = [(n, t) for n, t in con.execute(
        "SELECT club_name, tm_id FROM tm_club WHERE tm_id IS NOT NULL AND capacity IS NULL")]
    if sample:
        todo = todo[:sample]
    print(f"clubs needing capacity: {len(todo):,}", flush=True)
    got = 0
    for i, (name, tid) in enumerate(todo):
        try:
            html = fetch.get(HOME.format(tid=tid), min_delay=2.5, timeout=40, headers=HDR)
        except Exception:
            continue
        cap = parse_capacity(html)
        if cap:
            con.execute("UPDATE tm_club SET capacity=? WHERE tm_id=?", (cap, tid))
            got += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(todo)} clubs, {got} capacities", flush=True)
    print(f"\ncapacities written: {got}")
    # coverage: matches whose home club maps to a tm_club with capacity
    cov = con.execute(
        """SELECT COUNT(*) FROM match m JOIN club c ON c.club_id=m.home_club_id
           JOIN tm_club t ON t.club_name=c.name WHERE t.capacity IS NOT NULL""").fetchone()[0]
    tot = con.execute("SELECT COUNT(*) FROM match").fetchone()[0]
    print(f"matches with home-stadium capacity: {cov:,}/{tot:,} ({100*cov//tot}%)")
    con.close()


if __name__ == "__main__":
    main()
