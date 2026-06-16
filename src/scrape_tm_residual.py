"""Step 4: fill residual unmatched starters via targeted Transfermarkt lookup.
For each residual ESPN starter (no player_xwalk link, target leagues): TM quick-search the name,
open the player page, read DOB + canonical name. Then match to FM by DOB (preferred) or canonical
name within the player's CLUB SQUAD (the same >=2-teammate grade-club bridge as roster_match).
Writes high-confidence links to match_grade_link (method tm_dob / tm_name) + learned_alias, and
caches TM results in tm_player so re-runs are cheap. Polite single-worker, disk-cached.
Usage: python D:/Programming/claude/FM/src/scrape_tm_residual.py [limit]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
import roster_match as RM
from build_xwalk import xnorm, dob_close

SCH = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={q}"
BASE = "https://www.transfermarkt.com"
HDR = {"Accept-Language": "en-US,en;q=0.9"}


def tm_lookup(name):
    """Return (tm_id, tm_name, tm_dob_iso) for the first player hit, or (None,None,None)."""
    try:
        html = fetch.get(SCH.format(q=name.replace(" ", "+")), min_delay=2.5, timeout=40, headers=HDR)
    except Exception:
        return None, None, None
    s = BeautifulSoup(html, "lxml")
    link = next((a for a in s.select('a[href*="/profil/spieler/"]') if a.get_text(strip=True)), None)
    if not link:
        return None, None, None
    tm_name = link.get_text(strip=True)
    m = re.search(r"/profil/spieler/(\d+)", link["href"])
    tm_id = m.group(1) if m else None
    try:
        ph = fetch.get(BASE + link["href"], min_delay=2.5, timeout=40, headers=HDR)
    except Exception:
        return tm_id, tm_name, None
    ps = BeautifulSoup(ph, "lxml")
    dob = None
    for lbl in ps.select("span.info-table__content--regular"):
        if "birth" in lbl.get_text(strip=True).lower():
            v = lbl.find_next_sibling("span")
            if v:
                mm = re.search(r"(\d{2})/(\d{2})/(\d{4})", v.get_text(" ", strip=True))
                if mm:
                    dob = f"{mm.group(3)}-{mm.group(2)}-{mm.group(1)}"
            break
    return tm_id, tm_name, dob


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    con = db.connect()
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=NORMAL")
    con.execute("""CREATE TABLE IF NOT EXISTS tm_player(
        espn_player_id TEXT PRIMARY KEY, tm_id TEXT, tm_name TEXT, tm_dob TEXT)""")
    D = RM.load(con)
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    pid_eid = {pid: eid for eid, pid in con.execute(
        "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (espn_sid,))}
    linked = {r[0] for r in con.execute("SELECT espn_player_id FROM player_xwalk WHERE fm_uid IS NOT NULL")}

    # residual unmatched starter pids per club-season squad (reuse roster bridge)
    fmname = {u: xnorm(n) for u, n in D["uid_name"].items()}
    cand_uids = defaultdict(set)        # residual pid -> candidate FM uids (their club squads)
    pid_name = {}
    for (mid, ecid), pls in D["starters"].items():
        if D["comp_name"].get(D["comp_of"].get(mid)) in EXCL_GUARD:
            continue
        lab = D["season_of"].get(mid)
        if not lab:
            continue
        squad, _ = RM._squad_of(D, mid, ecid, lab)
        for pid, pos in pls:
            r = D["resolved"].get(pid)
            eid = pid_eid.get(pid)
            if (r and r[1]) or not eid or eid in linked:
                continue
            cand_uids[pid] |= squad
            pid_name[pid] = D["pname"].get(pid, "")

    todo = [pid for pid in cand_uids if cand_uids[pid]]
    print(f"residual unmatched with a candidate squad: {len(todo):,} (scrape limit {limit})")
    cached = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT * FROM tm_player")}
    rows = []; alias = []; nlink = ntried = 0
    for pid in todo[:limit]:
        eid = pid_eid[pid]; name = pid_name[pid]
        if eid in cached:
            tm_id, tm_name, tm_dob = cached[eid]
        else:
            tm_id, tm_name, tm_dob = tm_lookup(name)
            con.execute("INSERT OR REPLACE INTO tm_player VALUES (?,?,?,?)", (eid, tm_id, tm_name, tm_dob))
            ntried += 1
        cands = cand_uids[pid]
        hit = method = None
        if tm_dob:
            dm = [u for u in cands if D["fm_dob"].get(u) and dob_close(tm_dob, D["fm_dob"][u])]
            if len(dm) == 1:
                hit, method = dm[0], "tm_dob"
        if hit is None and tm_name:
            nn = xnorm(tm_name)
            nm = [u for u in cands if fmname.get(u) == nn]
            if len(nm) == 1:
                hit, method = nm[0], "tm_name"
        if hit:
            nlink += 1
            for mid, ecid in [(m, c) for (m, c), pls in D["starters"].items() if any(p == pid for p, _ in pls)]:
                rows.append((mid, pid, hit, method, "high"))
            alias.append((eid, name, hit, D["uid_name"].get(hit, ""), 1))
        if ntried and ntried % 50 == 0:
            con.commit()
            print(f"  scraped {ntried}, linked {nlink}/{ntried}", flush=True)
    con.execute("BEGIN")
    con.executemany("INSERT OR REPLACE INTO match_grade_link VALUES (?,?,?,?,?)", rows)
    con.executemany("INSERT OR REPLACE INTO learned_alias VALUES (?,?,?,?,?)", alias)
    con.execute("COMMIT")
    print(f"\nTM residual: tried {ntried}, NEW links {nlink} ({len(rows):,} match rows, {len(alias)} aliases)")
    con.close()


EXCL_GUARD = RM.EXCL

if __name__ == "__main__":
    main()
