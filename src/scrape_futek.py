"""Scrape futek.io FM24 launch database for EPL players (incl. hidden attributes, raw 0-200 CA/PA).
Search caps at 50 rows -> adaptive slicing by CA band, then age band.
Player pages ~4.6MB -> parsed JSON cached per uid in data/raw/futek/, raw HTML not kept.
Usage: python D:/Programming/claude/FM/src/scrape_futek.py
"""
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
import db as dbmod
import fetch

BASE = "https://www.futek.io"
OUT = dbmod.ROOT / "data" / "raw" / "futek"
DIVISION = "English Premier Division"
SNAP_DATE = "2023-06-01"   # futek export is pre-summer-2023 data (Vicario@Empoli etc.), despite FM24 branding

CAT_MAP = {"technical": "technical", "mental": "mental", "physical": "physical",
           "hidden": "hidden", "goalkeeping": "goalkeeping"}


def new_session():
    s = requests.Session()
    s.headers.update({"User-Agent": fetch.UA, "Referer": f"{BASE}/fmdb"})
    s.get(f"{BASE}/fmdb", timeout=30)
    return s


def search(s, min_ca, max_ca, min_age, max_age):
    data = {
        "player-name": "", "club-search4": "", "division-search3": DIVISION,
        "nationality-search1": "", "position-search2": "",
        "min-age": str(min_age), "max-age": str(max_age),
        "min-ca": str(min_ca), "max-ca": str(max_ca),
        "min-pa": "1", "max-pa": "200", "search-source": "main",
    }
    for attempt in range(3):
        r = s.post(f"{BASE}/fmdb", data=data, timeout=60)
        if r.status_code == 200:
            return parse_rows(r.text)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"search failed ca[{min_ca},{max_ca}] age[{min_age},{max_age}]: {r.status_code}")


def parse_rows(html):
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for a in soup.select('a[href^="/"]'):
        m = re.fullmatch(r"/(\d+)/", a.get("href", ""))
        if not m:
            continue
        tr = a.find_parent("tr")
        if not tr:
            continue
        tds = [td.get_text(strip=True) for td in tr.select("td")]
        # columns: [Name, Age, Position, Nationality, Club, Division, CA, PA]
        # (the name cell holds the <a>, so tds[0] is the name, not the age)
        if len(tds) >= 8:
            rows.append({"uid": m.group(1), "name": a.get_text(strip=True) or tds[0],
                         "age": tds[1], "pos": tds[2], "nat": tds[3], "club": tds[4],
                         "division": tds[5], "ca": tds[6], "pa": tds[7]})
    return rows


def enumerate_players(s):
    """Adaptive slicing: CA bands first, age bands inside saturated 1-wide CA bands."""
    found = {}
    stack = [(1, 200, 14, 60)]
    queries = 0
    while stack:
        ca_lo, ca_hi, ag_lo, ag_hi = stack.pop()
        rows = search(s, ca_lo, ca_hi, ag_lo, ag_hi)
        queries += 1
        time.sleep(1.0)
        if len(rows) >= 50:
            if ca_lo < ca_hi:
                mid = (ca_lo + ca_hi) // 2
                stack.append((ca_lo, mid, ag_lo, ag_hi))
                stack.append((mid + 1, ca_hi, ag_lo, ag_hi))
            elif ag_lo < ag_hi:
                mid = (ag_lo + ag_hi) // 2
                stack.append((ca_lo, ca_hi, ag_lo, mid))
                stack.append((ca_lo, ca_hi, mid + 1, ag_hi))
            else:
                for r_ in rows:
                    found[r_["uid"]] = r_
            continue
        for r_ in rows:
            found[r_["uid"]] = r_
        if queries % 10 == 0:
            print(f"  enum: {queries} queries, {len(found)} players, stack={len(stack)}", flush=True)
    print(f"enumeration done: {queries} queries, {len(found)} players")
    return found


def parse_player_page(html):
    soup = BeautifulSoup(html, "lxml")
    attrs = {}
    for h3 in soup.select("h3"):
        cat = CAT_MAP.get(h3.get_text(strip=True).lower())
        if not cat:
            continue
        tbl = h3.find_next("table")
        if not tbl:
            continue
        for tr in tbl.select("tr"):
            tds = tr.select("td")
            if len(tds) >= 2:
                name = tds[0].get_text(strip=True)
                try:
                    val = float(tds[1].get_text(strip=True))
                except ValueError:
                    continue
                key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
                attrs[key] = (cat, val)
    meta = {}
    for k in ("player_name", "club", "current_ability", "potential_ability", "age", "division"):
        m = re.search(rf"'{k}'\s*:\s*'([^']*)'", html)
        if m:
            meta[k] = m.group(1)
    return attrs, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = dbmod.connect()
    src = dbmod.source_id(con, "futek", BASE)
    fmv = dbmod.fm_version_id(con, "FM24", "24.0-futek-export", SNAP_DATE)
    s = new_session()
    players = enumerate_players(s)
    (OUT / "_epl_index.json").write_text(json.dumps(players, indent=1), encoding="utf-8")
    dbmod.log(con, "futek", "enum", "ok", f"{len(players)} EPL players")

    saved = errors = 0
    for i, (uid, row) in enumerate(sorted(players.items())):
        jpath = OUT / f"{uid}.json"
        try:
            if jpath.exists():
                rec = json.loads(jpath.read_text(encoding="utf-8"))
            else:
                r = s.get(f"{BASE}/{uid}/", timeout=90)
                r.raise_for_status()
                attrs, meta = parse_player_page(r.text)
                rec = {"uid": uid, "row": row, "meta": meta,
                       "attrs": {k: [c, v] for k, (c, v) in attrs.items()}}
                jpath.write_text(json.dumps(rec), encoding="utf-8")
                time.sleep(1.0)
            attrs = {k: (c, v) for k, (c, v) in
                     ((k, tuple(cv)) for k, cv in rec["attrs"].items())}
            if not attrs:
                errors += 1
                dbmod.log(con, "futek", uid, "skip", "no attrs")
                continue
            name = rec["meta"].get("player_name") or row["name"]
            pid = dbmod.player_id(con, name, src=src, src_player_id=uid)
            cid = dbmod.club_id(con, rec["meta"].get("club") or row["club"])
            ca = int(rec["meta"].get("current_ability") or row["ca"] or 0) or None
            pa = int(rec["meta"].get("potential_ability") or row["pa"] or 0) or None
            if dbmod.save_snapshot(con, pid=pid, src=src, fmv=fmv, cid=cid,
                                   snapshot_date=SNAP_DATE, attrs=attrs,
                                   meta={"position": row.get("pos"), "ca": ca, "pa": pa}):
                saved += 1
            if (i + 1) % 25 == 0:
                con.commit()
                print(f"  {i+1}/{len(players)} saved={saved} err={errors}", flush=True)
        except Exception as e:
            errors += 1
            dbmod.log(con, "futek", uid, "error", str(e)[:200])
            time.sleep(2)
    con.commit()
    dbmod.log(con, "futek", "", "ok", f"done saved={saved} errors={errors}")
    print(f"FUTEK DONE saved={saved} errors={errors}")
    con.close()


if __name__ == "__main__":
    main()
