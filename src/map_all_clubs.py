"""Map EVERY club that appears in our matches to a Transfermarkt id, then validate it in the same pass.
tm_club was only ever populated with DOB-needy (lower/foreign) clubs, so the top leagues (Man City,
Arsenal, ...) were never mapped -> TM features can't cover them. This maps the ~658 unmapped clubs
(TM search -> first /verein/ id) and immediately validates by squad-name overlap vs our lineup players
(valid=1 only if the mapping is real). Run scrape_tm_enrich + scrape_capacity afterwards to fill the
now-complete valid set. Single writer. Usage: python D:/Programming/claude/FM/src/map_all_clubs.py
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
import scrape_tm_squads as tms

SEARCH = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={q}"
HDR = {"Accept-Language": "en"}
SEASONS = ["2023", "2022", "2024", "2021"]


def search_tm(name):
    try:
        html = fetch.get(SEARCH.format(q=name.replace(" ", "+")), min_delay=2.5, timeout=40, headers=HDR)
    except Exception:
        return None
    m = re.search(r'href="/[^"]*/startseite/verein/(\d+)"', html)
    return m.group(1) if m else None


def main():
    con = db.connect()
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("CREATE TABLE IF NOT EXISTS tm_club(club_name TEXT PRIMARY KEY, tm_id TEXT)")
    if "valid" not in [c[1] for c in con.execute("pragma table_info(tm_club)")]:
        con.execute("ALTER TABLE tm_club ADD COLUMN valid INTEGER")

    cname = {cid: nm for cid, nm in con.execute("SELECT club_id, name FROM club")}
    have = {n for n, in con.execute("SELECT club_name FROM tm_club WHERE tm_id IS NOT NULL")}
    inmatch = set(c for c, in con.execute(
        "SELECT DISTINCT home_club_id FROM match UNION SELECT DISTINCT away_club_id FROM match"))
    our = defaultdict(set)
    for cid, nn in con.execute(
            "SELECT DISTINCT mp.club_id, p.norm_name FROM match_player mp JOIN player p USING(player_id)"):
        our[cid].add(nn)

    todo = [(c, cname[c]) for c in inmatch if cname.get(c) and cname[c] not in have]
    print(f"unmapped clubs to map+validate: {len(todo):,}", flush=True)
    mapped = valid = 0
    for i, (cid, name) in enumerate(todo):
        tid = search_tm(name)
        if not tid:
            con.execute("INSERT OR REPLACE INTO tm_club (club_name, tm_id, valid) VALUES (?,?,?)", (name, None, 0))
            continue
        mapped += 1
        ours = our.get(cid, set())
        ok = 0
        if ours:
            tm_names = set()
            for yr in SEASONS:
                tm_names = {nn for nn, _ in tms.tm_squad(tid, yr)}
                if tm_names:
                    break
            overlap = len(tm_names & ours)
            ok = 1 if (overlap >= 4 and overlap / max(len(tm_names), 1) >= 0.20) else 0
        con.execute("INSERT OR REPLACE INTO tm_club (club_name, tm_id, valid) VALUES (?,?,?)", (name, tid, ok))
        valid += ok
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)}  mapped={mapped} valid={valid}", flush=True)
    tot = con.execute("SELECT COUNT(*) FROM tm_club WHERE valid=1").fetchone()[0]
    print(f"\nnewly mapped: {mapped}, newly valid: {valid}; TOTAL valid tm_club now: {tot:,}")
    con.close()


if __name__ == "__main__":
    main()
