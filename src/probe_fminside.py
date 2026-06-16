"""Probe fminside fm_league/fm_nat strings for the ESPN-primary grade-pass leagues.
Reports db7 club count per league WITHOUT fetching player pages (cheap, serial).
A league returning 0 clubs has a wrong fm_league string and must be fixed before scraping.
Usage: python D:/Programming/claude/FM/src/probe_fminside.py
Run ONLY when no other fminside scrape is active (enumeration uses a per-IP session filter).
"""
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import fetch
import leagues as L
from scrape_fminside import BASE, UPDATE_FILTER, CLUB_TABLE, FILTER_DEFAULTS

# ranks 16-32 minus Israel(30) — the "all viable, exclude weak" set
TARGET_RANKS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32]
DBID = 7  # FM26 — proxy; older dbs verified per-db at scrape time by the scraper itself


def club_count(dbid, fm_league, fm_nat):
    s = requests.Session()
    s.headers.update({"User-Agent": fetch.UA})
    s.get(f"{BASE}/clubs", timeout=90)
    s.post(UPDATE_FILTER, data={**FILTER_DEFAULTS, "page": "clubs",
                                "database_version": str(dbid),
                                "league": fm_league, "nationality": fm_nat}, timeout=90)
    time.sleep(1)
    html = s.get(CLUB_TABLE, timeout=90).text
    return len(sorted({m for m in re.findall(r'href="(/clubs/[^"]+)"', html)}))


def main():
    by_rank = {l["rank"]: l for l in L.EXTRA_LEAGUES}
    leagues = [by_rank[r] for r in TARGET_RANKS]
    print(f"Probing {len(leagues)} ESPN-primary leagues on db{DBID} (club enumeration only)\n")
    ok = bad = 0
    results = []
    for lg in leagues:
        try:
            n = club_count(DBID, lg["fm_league"], lg["fm_nat"])
        except Exception as e:
            n = -1
            print(f"  {lg['name']:32} ERROR {e}")
        results.append((lg, n))
        if n > 0:
            ok += 1
            print(f"  {lg['name']:32} {n:>3} clubs   ['{lg['fm_league']}' / '{lg['fm_nat']}']")
        elif n == 0:
            bad += 1
            print(f"  {lg['name']:32} {n:>3} clubs   *** WRONG STRING -> ['{lg['fm_league']}' / '{lg['fm_nat']}']")
        time.sleep(1.0)
    print(f"\nscrape-ready (>0 clubs): {ok} / {len(leagues)} ; need string fix: {bad}")
    print("SCRAPE_READY=" + ",".join(lg["name"] for lg, n in results if n > 0))


if __name__ == "__main__":
    main()
