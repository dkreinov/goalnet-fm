"""Scrape fminside.net FM player attributes for EPL across 4 database versions.
db 4 = FM24 original (2023-24 early), db 5 = FM24.3 winter (2023-24 late),
db 6 = FMU25 community (2024-25), db 7 = FM26 (2025-26).
Player pages are URL-driven (no session); squad enumeration needs session DB set
via update_filter.php. ~8k player pages, disk-cached, ~1.1s/req.
Usage: python D:/Programming/claude/FM/src/scrape_fminside.py [db ...]
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

BASE = "https://fminside.net"
UPDATE_FILTER = f"{BASE}/resources/inc/ajax/update_filter.php"
CLUB_TABLE = f"{BASE}/beheer/modules/clubs/resources/inc/frontend/generate-club-table.php?ajax_request=1"

DBS = {
    1: {"game": "FM21", "db_version": "21.0.0", "date": "2020-11-01", "season": "2020-21"},
    2: {"game": "FM22", "db_version": "22.1.0", "date": "2021-11-01", "season": "2021-22"},
    3: {"game": "FM23", "db_version": "23.4.0", "date": "2023-02-01", "season": "2022-23"},
    4: {"game": "FM24", "db_version": "24.1.0", "date": "2023-11-06", "season": "2023-24"},
    5: {"game": "FM24", "db_version": "24.3.0", "date": "2024-02-26", "season": "2023-24"},
    6: {"game": "FMU25", "db_version": "24.4.0-community", "date": "2024-10-01", "season": "2024-25"},
    7: {"game": "FM26", "db_version": "26.2.0", "date": "2026-03-01", "season": "2025-26"},
}

CAT = {}
for n in ("crossing dribbling finishing first-touch heading long-shots marking "
          "passing tackling technique").split():
    CAT[n] = "technical"
for n in ("aggression anticipation bravery composure concentration decisions determination "
          "flair leadership off-the-ball positioning teamwork vision work-rate").split():
    CAT[n] = "mental"
for n in ("acceleration agility balance jumping-reach natural-fitness pace stamina "
          "strength").split():
    CAT[n] = "physical"
for n in "corners free-kick-taking long-throws penalty-taking".split():
    CAT[n] = "set_pieces"
for n in ("aerial-reach command-of-area communication eccentricity handling kicking "
          "one-on-ones punching-tendency reflexes rushing-out-tendency throwing").split():
    CAT[n] = "goalkeeping"

FILTER_DEFAULTS = {
    "gender": "-1", "club": "", "name": "", "uid": "", "nationality": "", "min_age": "",
    "max_age": "", "min_ability": "", "max_ability": "", "min_potential": "",
    "max_potential": "", "min_value": "", "max_value": "", "max_wage": "", "clause": "",
}


def money(txt):
    if not txt:
        return None
    m = re.search(r"€\s*([\d,.]+)\s*([KM]?)", txt)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return int(v * {"K": 1e3, "M": 1e6, "": 1}[m.group(2)])


def parse_player(html: str):
    s = BeautifulSoup(html, "lxml")
    info = {}
    for li in s.select("li"):
        k, v = li.select_one("span.key"), li.select_one("span.value")
        if k and v:
            info[k.get_text(strip=True)] = v.get_text(" ", strip=True)
    if "Name" not in info:
        return None
    attrs = {}
    for tr in s.select("tr[id]"):
        td = tr.select_one("td.stat")
        if td is None:
            continue
        m = re.search(r"value_(\d+)", " ".join(td.get("class", [])))
        if not m:
            continue
        aid = tr["id"]
        attrs[aid] = (CAT.get(aid, "other"), float(m.group(1)))
    meta_cards = s.select("div.meta span.card, p.meta span.card, span.meta span.card")
    if not meta_cards:
        meta_cards = [c for c in s.select("span.card")
                      if c.parent.get("class") and "meta" in c.parent.get("class")]
    ca = pa = None
    if len(meta_cards) >= 2:
        try:
            ca, pa = int(meta_cards[0].get_text(strip=True)), int(meta_cards[1].get_text(strip=True))
        except ValueError:
            pass

    def num(key, pattern=r"(\d+)"):
        m2 = re.search(pattern, info.get(key, ""))
        return int(m2.group(1)) if m2 else None

    return {
        "name": info["Name"],
        "position": info.get("Position(s)"),
        "club": info.get("Club"),
        "ca": ca, "pa": pa,
        "value_eur": money(info.get("Sell value")),
        "wage_eur": money(info.get("Wages")),
        "foot_left": num("Left foot"), "foot_right": num("Right foot"),
        "height_cm": num("Height"), "weight_kg": num("Weight"),
        "attrs": attrs,
    }


def league_player_urls(dbid: int, fm_league: str, fm_nat: str) -> list[str]:
    """Enumerate a league's squads for a db version. Returns player page URLs."""
    s = requests.Session()
    s.headers.update({"User-Agent": fetch.UA})
    s.get(f"{BASE}/clubs", timeout=90)
    s.post(UPDATE_FILTER, data={**FILTER_DEFAULTS, "page": "clubs",
                                "database_version": str(dbid),
                                "league": fm_league, "nationality": fm_nat}, timeout=90)
    time.sleep(1)
    html = s.get(CLUB_TABLE, timeout=90).text
    club_urls = sorted({m for m in re.findall(r'href="(/clubs/[^"]+)"', html)})
    print(f"  db{dbid} {fm_league}: {len(club_urls)} clubs")
    # set session db for club squad rendering
    s.post(UPDATE_FILTER, data={**FILTER_DEFAULTS, "page": "players",
                                "database_version": str(dbid),
                                "league": fm_league}, timeout=90)
    time.sleep(1)
    purls = set()
    for cu in club_urls:
        try:
            r = s.get(BASE + cu, timeout=90)
        except Exception as e:
            print(f"    {cu}: FAILED {e}")
            continue
        links = {l for l in re.findall(r'href="(/players/\d[^"]+)"', r.text)
                 if l.startswith(f"/players/{dbid}-")}
        purls |= links
        time.sleep(1.0)
    return sorted(purls)


def scrape_set(con, src, urls, fmv, snapshot_date, tag):
    saved = skipped = errors = 0
    for i, pu in enumerate(urls):
        uid_m = re.search(r"/players/\d+-[^/]+/(\d+)-", pu)
        uid = uid_m.group(1) if uid_m else pu
        try:
            html = fetch.get(BASE + pu, min_delay=2.5, timeout=90)
            p = parse_player(html)
            if not p or not p["attrs"]:
                errors += 1
                dbmod.log(con, "fminside", pu, "skip", "no attrs parsed")
                continue
            pid = dbmod.player_id(con, p["name"], src=src, src_player_id=uid)
            cid = dbmod.club_id(con, p["club"]) if p["club"] else None
            sid = dbmod.save_snapshot(
                con, pid=pid, src=src, fmv=fmv, cid=cid, snapshot_date=snapshot_date,
                attrs=p["attrs"],
                meta={k: p[k] for k in ("position", "ca", "pa", "value_eur", "wage_eur",
                                        "foot_left", "foot_right", "height_cm", "weight_kg")})
            if sid:
                saved += 1
            else:
                skipped += 1
            if (i + 1) % 100 == 0:
                con.commit()
                print(f"    {tag}: {i+1}/{len(urls)} saved={saved} skip={skipped} err={errors}", flush=True)
        except Exception as e:
            errors += 1
            dbmod.log(con, "fminside", pu, "error", str(e))
    con.commit()
    return saved, skipped, errors


def main():
    """Args: optional league names (default = enabled registry leagues) and
    --db N,N to restrict db versions (default all six)."""
    import leagues as L
    args = sys.argv[1:]
    dbids = [1, 2, 3, 5, 6, 7]
    if "--db" in args:
        k = args.index("--db")
        dbids = [int(x) for x in args[k + 1].split(",")]
        args = args[:k] + args[k + 2:]
    if args:
        target_leagues = [L.BY_NAME.get(n) or L.EXTRA_BY_NAME[n] for n in args]
    else:
        target_leagues = L.enabled()

    con = dbmod.connect()
    src = dbmod.source_id(con, "fminside", BASE)
    for lg in target_leagues:
        print(f"== {lg['name']} (rank {lg['rank']}) ==")
        for dbid in dbids:
            cfg = DBS[dbid]
            fmv = dbmod.fm_version_id(con, cfg["game"], cfg["db_version"], cfg["date"])
            tag = f"{lg['name']} db{dbid}/{cfg['game']}"
            try:
                urls = league_player_urls(dbid, lg["fm_league"], lg["fm_nat"])
            except Exception as e:
                dbmod.log(con, "fminside", tag, "error", f"enumeration: {e}")
                print(f"  {tag} enumeration FAILED: {e}")
                continue
            if not urls:
                dbmod.log(con, "fminside", tag, "skip", "no players enumerated")
                continue
            dbmod.log(con, "fminside", tag, "ok", f"enumerated {len(urls)}")
            saved, skipped, errors = scrape_set(con, src, urls, fmv, cfg["date"], tag)
            dbmod.log(con, "fminside", tag, "ok", f"done saved={saved} skip={skipped} err={errors}")
            print(f"  {tag} DONE saved={saved} skip={skipped} err={errors}")
    con.close()


if __name__ == "__main__":
    main()
