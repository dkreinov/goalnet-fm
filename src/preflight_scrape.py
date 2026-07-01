"""Step 1 preflight (throwaway): probe transfermarkt squad pages (per-player market value) and sofifa team
pages (overall rating) to confirm both are reachable+parseable before committing to multi-hour scrapes.
Prints TM_OK/TM_FAIL and SOFIFA_OK/SOFIFA_BLOCKED with counts. Usage: python src/preflight_scrape.py
"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import db, fetch

TM_SQUAD = "https://www.transfermarkt.com/x/kader/verein/{cid}/saison_id/2024/plus/1"
HDR = {"Accept-Language": "en"}
VAL = re.compile(r"€\s*([\d.,]+)\s*(m|k|bn)?", re.I)
SOFIFA_TRIES = [
    "https://sofifa.com/team/241/",            # Real Madrid team page
    "https://sofifa.com/players?type=all",     # players list
]


def probe_tm(con):
    cids = [r[0] for r in con.execute("SELECT tm_id FROM tm_club WHERE valid=1 LIMIT 2")]
    if not cids:
        cids = [r[0] for r in con.execute("SELECT tm_id FROM tm_club LIMIT 2")]
    total = 0
    for cid in cids:
        try:
            html = fetch.get(TM_SQUAD.format(cid=cid), min_delay=2.5, timeout=40, headers=HDR, cache=False)
        except Exception as e:
            print(f"  TM fetch error club {cid}: {type(e).__name__} {e}"); continue
        vals = VAL.findall(html)
        # market-value column lives in the squad table; a squad page yields many € tokens
        n = sum(1 for v, u in vals if u)          # values with m/k unit = real market values
        print(f"  TM club {cid}: {n} market-value tokens, page {len(html)//1000}kb")
        total += n
    print(f"TM_OK n={total}" if total >= 10 else f"TM_FAIL n={total}")
    return total >= 10


def probe_sofifa():
    for url in SOFIFA_TRIES:
        try:
            html = fetch.get(url, min_delay=3.0, timeout=40, cache=False)
        except Exception as e:
            print(f"  sofifa fetch error {url}: {type(e).__name__} {e}"); continue
        low = html.lower()
        if any(s in low for s in ("just a moment", "attention required", "cf-chl", "cloudflare", "enable javascript")):
            print(f"  sofifa {url}: Cloudflare/JS challenge ({len(html)//1000}kb)"); continue
        # sofifa lists overall ratings; count 2-digit rating tokens near player rows
        ratings = re.findall(r'<(?:td|span|em)[^>]*>\s*(\d{2})\s*</(?:td|span|em)>', html)
        rr = [int(x) for x in ratings if 40 <= int(x) <= 99]
        print(f"  sofifa {url}: {len(rr)} rating-like tokens, page {len(html)//1000}kb")
        if len(rr) >= 10:
            print(f"SOFIFA_OK n={len(rr)}"); return True
    print("SOFIFA_BLOCKED n=0")
    return False


def main():
    con = db.connect()
    missing = con.execute("""SELECT COUNT(DISTINCT c) FROM (
        SELECT home_club_id c FROM match UNION SELECT away_club_id FROM match)
        WHERE c NOT IN (SELECT club_id FROM club_season_tm)""").fetchone()[0]
    nplayers = con.execute("SELECT COUNT(DISTINCT player_id) FROM match_player").fetchone()[0]
    print(f"universe: {missing} club_ids missing squad value, {nplayers} distinct dataset players", flush=True)
    print("--- transfermarkt ---", flush=True); tm = probe_tm(con)
    print("--- sofifa ---", flush=True); so = probe_sofifa()
    print(f"\nPREFLIGHT: TM={'OK' if tm else 'FAIL'} SOFIFA={'OK' if so else 'BLOCKED'}", flush=True)


if __name__ == "__main__":
    main()
