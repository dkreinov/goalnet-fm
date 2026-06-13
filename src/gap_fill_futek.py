"""Gap-fill futek snapshots: search by name for 2023-24 EPL starters that the
division-based enumeration missed (they sat at non-EPL clubs in the pre-summer-2023 data).
Exact normalized-name match required; saves snapshot dated 2023-06-01.
Usage: python D:/Programming/claude/FM/src/gap_fill_futek.py
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
import db as dbmod
import fetch
from scrape_futek import BASE, OUT, SNAP_DATE, new_session, parse_player_page

def parse_rows_fixed(html):
    """Result rows: tds = [name, age, pos, nat, club, division, ca, pa]."""
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
        if len(tds) >= 8:
            rows.append({"uid": m.group(1), "name": tds[0] or a.get_text(strip=True),
                         "age": tds[1], "pos": tds[2], "nat": tds[3], "club": tds[4],
                         "division": tds[5], "ca": tds[6], "pa": tds[7]})
    return rows


def unmatched_starters(con):
    has_snap = {r[0] for r in con.execute("SELECT DISTINCT player_id FROM player_snapshot")}
    idx = defaultdict(list)
    for r in con.execute("SELECT player_id, norm_name FROM player"):
        if r[0] in has_snap:
            idx[r[1]].append(r[0])
    names = []
    for r in con.execute("""
        SELECT DISTINCT p.player_id, p.norm_name, p.name
        FROM match_player mp
        JOIN match m ON m.match_id=mp.match_id
        JOIN season s ON s.season_id=m.season_id AND s.label='2023-24'
        JOIN player p ON p.player_id=mp.player_id
        WHERE mp.started=1"""):
        if r[0] not in has_snap and r[1] not in idx:
            names.append((r[0], r[1], r[2]))
    return names


def main():
    con = dbmod.connect()
    src = dbmod.source_id(con, "futek", BASE)
    fmv = con.execute("SELECT fm_version_id FROM fm_version WHERE db_version='23.x-pre-summer-2023'").fetchone()[0]
    targets = unmatched_starters(con)
    print(f"{len(targets)} unmatched 2023-24 starters")
    s = new_session()
    saved = nf = amb = 0
    for pid, nn, disp in targets:
        last = nn.split()[-1] if nn.split() else nn
        data = {"player-name": last, "club-search4": "", "division-search3": "",
                "nationality-search1": "", "position-search2": "",
                "min-age": "14", "max-age": "60", "min-ca": "1", "max-ca": "200",
                "min-pa": "1", "max-pa": "200", "search-source": "main"}
        try:
            r = s.post(f"{BASE}/fmdb", data=data, timeout=60)
            rows = parse_rows_fixed(r.text)
            time.sleep(0.8)
            # exact norm-name match, or all lineup-name tokens subset of row-name tokens
            t_lineup = set(nn.split())
            cands = [row for row in rows if dbmod.norm(row["name"]) == nn]
            if not cands:
                cands = [row for row in rows
                         if t_lineup <= set(dbmod.norm(row["name"]).split())
                         or set(dbmod.norm(row["name"]).split()) <= t_lineup]
            if not cands:
                nf += 1
                continue
            cands.sort(key=lambda x: -int(x["ca"] or 0))
            if len(cands) > 1 and cands[0]["ca"] == cands[1]["ca"]:
                amb += 1
            row = cands[0]
            uid = row["uid"]
            jpath = OUT / f"{uid}.json"
            if jpath.exists():
                rec = json.loads(jpath.read_text(encoding="utf-8"))
            else:
                pr = s.get(f"{BASE}/{uid}/", timeout=90)
                pr.raise_for_status()
                attrs, meta = parse_player_page(pr.text)
                rec = {"uid": uid, "row": row, "meta": meta,
                       "attrs": {k: [c, v] for k, (c, v) in attrs.items()}}
                jpath.write_text(json.dumps(rec), encoding="utf-8")
                time.sleep(0.8)
            attrs = {k: (cv[0], cv[1]) for k, cv in rec["attrs"].items()}
            if not attrs:
                nf += 1
                continue
            dbmod.player_id(con, disp, src=src, src_player_id=uid)  # ensure mapping
            con.execute("INSERT OR IGNORE INTO player_source_id (source_id, source_player_id, player_id) VALUES (?,?,?)",
                        (src, uid, pid))
            cid = dbmod.club_id(con, rec["meta"].get("club") or row["club"])
            ca = int(rec["meta"].get("current_ability") or 0) or None
            pa = int(rec["meta"].get("potential_ability") or 0) or None
            if dbmod.save_snapshot(con, pid=pid, src=src, fmv=fmv, cid=cid,
                                   snapshot_date=SNAP_DATE, attrs=attrs,
                                   meta={"position": row.get("pos"), "ca": ca, "pa": pa}):
                saved += 1
                print(f"  + {disp} <- {row['name']} ({row['club']}, CA {row['ca']})")
        except Exception as e:
            dbmod.log(con, "futek-gap", nn, "error", str(e)[:150])
            time.sleep(2)
    con.commit()
    print(f"GAP-FILL DONE saved={saved} not_found={nf} ambiguous={amb}")
    dbmod.log(con, "futek-gap", "", "ok", f"saved={saved} nf={nf} amb={amb}")
    con.close()


if __name__ == "__main__":
    main()
