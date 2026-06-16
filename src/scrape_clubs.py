"""Scrape fminside club attributes (Reputation, facilities, finances) per club per FM edition
into club_attribute. Reputation is the headline club-strength feature; facilities/finances are
bonus. Manager is NOT on fminside (it's a player DB) — that needs Transfermarkt separately.
Single writer only (never run alongside another DB-writing scrape).
Usage: python src/scrape_clubs.py [league names...] --db 7,6,5 [--workers N]
"""
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
import db as dbmod
import fetch
from scrape_fminside import (BASE, UPDATE_FILTER, CLUB_TABLE, FILTER_DEFAULTS, DBS, money,
                             _begin, _commit, BATCH)

NUM = {"Reputation", "Training facilities", "Youth Facilities", "Youth Recruitment", "Junior Coaching"}
MON = {"Balance", "Transfer Budget", "Total Wages"}


def club_urls(dbid, fm_league, fm_nat):
    s = requests.Session(); s.headers.update({"User-Agent": fetch.UA})
    s.get(f"{BASE}/clubs", timeout=90)
    s.post(UPDATE_FILTER, data={**FILTER_DEFAULTS, "page": "clubs", "database_version": str(dbid),
                                "league": fm_league, "nationality": fm_nat}, timeout=90)
    time.sleep(1)
    html = s.get(CLUB_TABLE, timeout=90).text
    return sorted({m for m in re.findall(r'href="(/clubs/[^"]+)"', html)})


def parse_club(html):
    s = BeautifulSoup(html, "lxml")
    name = None
    out = {}
    for li in s.select("li"):
        k = li.select_one("span.key")
        if not k:
            continue
        kt = k.get_text(strip=True)
        vs = li.select_one("span.card, span.value")
        vt = vs.get_text(" ", strip=True) if vs else ""
        if kt == "Name":
            name = vt or None
        elif kt in NUM:
            m = re.search(r"(\d+)", vt)
            if m:
                out[kt] = float(m.group(1))
        elif kt in MON:
            mv = money(vt)
            if mv is not None:
                out[kt] = float(mv)
    # regex fallback for facility labels not in li
    txt = s.get_text(" ", strip=True)
    for lbl in NUM:
        if lbl not in out:
            m = re.search(re.escape(lbl) + r"[^0-9]{0,15}(\d{1,3})\b", txt)
            if m:
                out[lbl] = float(m.group(1))
    return name, out


def main():
    args = sys.argv[1:]
    dbids = [7, 6, 5]
    workers = 6
    if "--db" in args:
        k = args.index("--db"); dbids = [int(x) for x in args[k + 1].split(",")]; args = args[:k] + args[k + 2:]
    if "--workers" in args:
        k = args.index("--workers"); workers = int(args[k + 1]); args = args[:k] + args[k + 2:]
    import leagues as L
    if args:
        leagues = [L.BY_NAME.get(n) or L.EXTRA_BY_NAME[n] for n in args]
    else:
        leagues = [l for l in (L.LEAGUES + L.EXTRA_LEAGUES) if "fm_league" in l]

    con = dbmod.connect()
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=NORMAL")
    src = dbmod.source_id(con, "fminside", BASE)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    total_saved = 0
    for lg in leagues:
        for dbid in dbids:
            cfg = DBS[dbid]
            fmv = dbmod.fm_version_id(con, cfg["game"], cfg["db_version"], cfg["date"])
            try:
                cus = club_urls(dbid, lg["fm_league"], lg["fm_nat"])
            except Exception as e:
                print(f"  {lg['name']} db{dbid} enum FAILED: {e}"); continue
            if not cus:
                continue
            print(f"== {lg['name']} db{dbid}/{cfg['game']}: {len(cus)} clubs ==", flush=True)

            def work(cu):
                return cu, fetch.get(BASE + cu, min_delay=0.0, timeout=90, headers={"Accept-Language": "en"})
            _begin(con); n = saved = 0
            try:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for fut in as_completed([ex.submit(work, cu) for cu in cus]):
                        try:
                            cu, html = fut.result()
                            name, attrs = parse_club(html)
                        except Exception:
                            continue
                        if not name or not attrs:
                            continue
                        cid = dbmod.club_id(con, name)
                        for an, av in attrs.items():
                            con.execute(
                                "INSERT OR REPLACE INTO club_attribute "
                                "(club_id, source_id, fm_version_id, snapshot_date, attr_name, attr_value) "
                                "VALUES (?,?,?,?,?,?)", (cid, src, fmv, cfg["date"], an.lower().replace(" ", "_"), av))
                        saved += 1; n += 1
                        if n >= BATCH:
                            _commit(con); _begin(con); n = 0
            finally:
                _commit(con)
            total_saved += saved
            print(f"   saved {saved} clubs' attrs", flush=True)
    print(f"\nclub_attribute total saved: {total_saved}; rows now: "
          f"{con.execute('SELECT COUNT(*) FROM club_attribute').fetchone()[0]:,}")
    con.close()


if __name__ == "__main__":
    main()
