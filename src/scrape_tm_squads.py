"""Close the gap via DOB enrichment: the unmatched starters are mostly common names with NO DOB
(e.g. 4 FM 'James Wilson's). TM club-squad pages list every player WITH DOB; the club constraint
kills the cross-club name ambiguity. We map our club -> TM club, scrape the season squad, and write
each matched player's DOB into source_identity(espn). build_xwalk's DOB anchor then resolves them.
Single writer only. Usage: python src/scrape_tm_squads.py [--sample N]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch

SEARCH = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={q}"
SQUAD = "https://www.transfermarkt.com/x/kader/verein/{cid}/saison_id/{yr}/plus/1"
HDR = {"Accept-Language": "en"}
EXCL = {"China Super League", "Ecuador LigaPro", "India Super League", "Paraguay Primera Division",
        "Peru Liga 1", "South Africa Premiership", "Israel Ligat haAl"}
YR = {"2020-21": "2020", "2021-22": "2021", "2022-23": "2022", "2023-24": "2023",
      "2024-25": "2024", "2025-26": "2025"}


def tm_club_id(con, name):
    row = con.execute("SELECT tm_id FROM tm_club WHERE club_name=?", (name,)).fetchone()
    if row:
        return row[0]
    try:
        html = fetch.get(SEARCH.format(q=name.replace(" ", "+")), min_delay=2.5, timeout=40, headers=HDR)
    except Exception:
        return None
    m = re.search(r'href="/[^"]*/startseite/verein/(\d+)"', html)
    tid = m.group(1) if m else None
    con.execute("INSERT OR REPLACE INTO tm_club VALUES (?,?)", (name, tid))
    return tid


def tm_squad(cid, yr):
    try:
        html = fetch.get(SQUAD.format(cid=cid, yr=yr), min_delay=2.5, timeout=40, headers=HDR)
    except Exception:
        return []
    cs = BeautifulSoup(html, "lxml")
    out = []
    for tr in cs.select("table.items > tbody > tr"):
        a = tr.select_one('td.hauptlink a[href*="/profil/spieler/"]')
        if not a:
            continue
        dob = None
        for td in tr.find_all("td"):
            mm = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", td.get_text(" ", strip=True))
            if mm:
                dob = f"{mm.group(3)}-{mm.group(2)}-{mm.group(1)}"
                break
        if dob:
            out.append((db.norm(a.get_text(strip=True)), dob))
    return out


def main():
    sample = None
    if "--sample" in sys.argv:
        sample = int(sys.argv[sys.argv.index("--sample") + 1])
    con = db.connect()
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=NORMAL")
    con.execute("CREATE TABLE IF NOT EXISTS tm_club(club_name TEXT PRIMARY KEY, tm_id TEXT)")
    espn = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]

    print("building priority (club,season) list of unmatched-no-DOB starters ...", flush=True)
    linked = {e for e, in con.execute("SELECT espn_player_id FROM player_xwalk WHERE fm_uid IS NOT NULL")}
    have_dob = {e for e, in con.execute(
        f"SELECT source_player_id FROM source_identity WHERE source_id={espn} AND dob IS NOT NULL")}
    pid_eid = {pid: eid for eid, pid in con.execute(
        f"SELECT source_player_id, player_id FROM player_source_id WHERE source_id={espn}")}
    pid_needy = {pid: eid for pid, eid in pid_eid.items() if eid not in linked and eid not in have_dob}
    pnorm = {pid: nn for pid, nn in con.execute("SELECT player_id, norm_name FROM player")}
    season_of = {mid: lab for mid, lab in con.execute(
        "SELECT m.match_id, s.label FROM match m JOIN season s ON s.season_id=m.season_id")}
    comp_of = {mid: con.execute("SELECT name FROM competition WHERE competition_id=?", (c,)).fetchone()[0]
               for mid, c in con.execute("SELECT match_id, competition_id FROM match")}
    cname = {cid: nm for cid, nm in con.execute("SELECT club_id, name FROM club")}
    # one pass over starters; bucket needy ESPN players per (club, season)
    need = defaultdict(dict)
    for mid, pid, cid in con.execute("SELECT match_id, player_id, club_id FROM match_player WHERE started=1"):
        eid = pid_needy.get(pid)
        if not eid:
            continue
        lab = season_of.get(mid)
        if not lab or comp_of.get(mid) in EXCL:
            continue
        need[(cname.get(cid), lab)][eid] = pnorm.get(pid, "")
    need = {k: list(v.items()) for k, v in need.items()}
    order = sorted(need, key=lambda k: -len(need[k]))
    if sample:
        order = order[:sample]
    print(f"{len(need):,} club-seasons need DOBs; processing {len(order):,}", flush=True)

    wrote = 0
    for i, (club, lab) in enumerate(order):
        yr = YR.get(lab)
        tid = tm_club_id(con, club)
        if not tid or not yr:
            continue
        squad = dict(tm_squad(tid, yr))     # norm_name -> dob (last wins; club usually unique per name)
        if not squad:
            continue
        con.execute("BEGIN")
        for eid, nn in need[(club, lab)]:
            dob = squad.get(nn)
            if dob:
                con.execute(
                    "INSERT OR REPLACE INTO source_identity (source_id, source_player_id, name, dob) "
                    "VALUES (?,?,(SELECT name FROM source_identity WHERE source_id=? AND source_player_id=?),?)",
                    (espn, eid, espn, eid, dob))
                wrote += 1
        con.execute("COMMIT")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(order)} club-seasons, {wrote} DOBs written", flush=True)
    print(f"\nDOBs written to source_identity(espn): {wrote}")
    con.close()


if __name__ == "__main__":
    main()
