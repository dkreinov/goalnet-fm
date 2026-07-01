"""Step 2: scrape transfermarkt per-player market value + fill club_season_tm for clubs missing squad value.
For each (club, season) pair that appears in our match dataset and lacks a club_season_tm value, resolve the
club's tm_id (tm_club, else name search), fetch the squad page (plus/1 = per-player values), parse each
player's (name, dob, market value), match to our player_id by (norm_name, dob), and write:
  - player_tm_value(player_id, season, value_eur)  [matched players]
  - club_season_tm(club_id, season, squad_value_eur, top11_value_eur, avg_age, squad_size)  [from the parse]
Throttled (fetch.py 2.5s) + disk-cached => resumable (re-run picks up where it left off). Single writer.
Usage: python src/scrape_tm_values.py [--limit N]
"""
import re, sys, datetime
from pathlib import Path
from bs4 import BeautifulSoup
sys.path.insert(0, str(Path(__file__).parent))
import db, fetch

SEARCH = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={q}"
SQUAD = "https://www.transfermarkt.com/x/kader/verein/{cid}/saison_id/{yr}/plus/1"
HDR = {"Accept-Language": "en"}
YR = {"2019-20": "2019", "2020-21": "2020", "2021-22": "2021", "2022-23": "2022",
      "2023-24": "2023", "2024-25": "2024", "2025-26": "2025"}
NOW = datetime.date.today()


def parse_eur(s):
    if not s: return None
    m = re.search(r"([\d.,]+)\s*(bn|m|k)?", s.replace("\xa0", " "), re.I)
    if not m: return None
    v = float(m.group(1).replace(",", "")); u = (m.group(2) or "").lower()
    return int(v * {"bn": 1e9, "m": 1e6, "k": 1e3, "": 1}[u])


def tm_club_id(con, club_id, name):
    row = con.execute("SELECT tm_id FROM tm_club WHERE club_name=?", (name,)).fetchone()
    if row and row[0]: return row[0]
    try:
        html = fetch.get(SEARCH.format(q=name.replace(" ", "+")), min_delay=2.5, timeout=40, headers=HDR)
    except Exception:
        return None
    m = re.search(r'href="/[^"]*/startseite/verein/(\d+)"', html)
    tid = m.group(1) if m else None
    if tid: con.execute("INSERT OR REPLACE INTO tm_club(club_name,tm_id) VALUES (?,?)", (name, tid))
    return tid


def squad_rows(cid, yr):
    """[(norm_name, dob, value_eur, age)] from a TM squad page (plus/1)."""
    try:
        html = fetch.get(SQUAD.format(cid=cid, yr=yr), min_delay=2.5, timeout=40, headers=HDR)
    except Exception:
        return None
    cs = BeautifulSoup(html, "lxml"); out = []
    for tr in cs.select("table.items > tbody > tr"):
        a = tr.select_one('td.hauptlink a[href*="/profil/spieler/"]')
        if not a: continue
        dob = age = None
        for td in tr.find_all("td"):
            mm = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b(?:\s*\((\d{1,2})\))?", td.get_text(" ", strip=True))
            if mm:
                dob = f"{mm.group(3)}-{mm.group(2)}-{mm.group(1)}"; age = int(mm.group(4)) if mm.group(4) else None
                break
        vcell = tr.select_one('td.rechts.hauptlink a') or tr.select_one('td.rechts.hauptlink')
        val = parse_eur(vcell.get_text(strip=True)) if vcell else None
        if dob:
            out.append((db.norm(a.get_text(strip=True)), dob, val, age))
    return out


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    con = db.connect(); con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS player_tm_value(player_id INTEGER, season TEXT, value_eur INTEGER, PRIMARY KEY(player_id,season))")
    seasonlab = {r[0]: r[1] for r in con.execute("SELECT season_id,label FROM season")}
    clubname = {r[0]: r[1] for r in con.execute("SELECT club_id,name FROM club")}
    have = {(r[0], r[1]) for r in con.execute("SELECT club_id,season FROM club_season_tm")}
    pmap = {(nn, dob): pid for pid, nn, dob in con.execute("SELECT player_id,norm_name,dob FROM player WHERE dob IS NOT NULL AND dob!=''")}
    # (club_id, season_label) pairs in our matches lacking a squad value, that resolve to a season year
    need = []
    for cid, si in con.execute("SELECT DISTINCT home_club_id,season_id FROM match UNION SELECT DISTINCT away_club_id,season_id FROM match"):
        lab = seasonlab.get(si)
        if lab in YR and (cid, lab) not in have and cid in clubname:
            need.append((cid, lab))
    print(f"need {len(need)} (club,season) pairs; player-dob map {len(pmap)}", flush=True)
    done = pv = cs_rows = 0
    for cid, lab in need:
        if limit and done >= limit: break
        done += 1
        tmid = tm_club_id(con, cid, clubname[cid])
        if not tmid: continue
        rows = squad_rows(tmid, YR[lab])
        if not rows: continue
        vals = [v for _, _, v, _ in rows if v]; ages = [a for _, _, _, a in rows if a]
        if vals:
            con.execute("INSERT OR REPLACE INTO club_season_tm(club_id,season,squad_value_eur,top11_value_eur,avg_age,squad_size) VALUES (?,?,?,?,?,?)",
                        (cid, lab, sum(vals), sum(sorted(vals)[-11:]), round(sum(ages)/len(ages), 1) if ages else None, len(rows)))
            cs_rows += 1
        for nn, dob, v, _ in rows:
            pid = pmap.get((nn, dob))
            if pid and v:
                con.execute("INSERT OR REPLACE INTO player_tm_value(player_id,season,value_eur) VALUES (?,?,?)", (pid, lab, v)); pv += 1
        if done % 50 == 0:
            con.commit(); print(f"  {done}/{len(need)} pairs | club_season_tm +{cs_rows} | player_tm_value +{pv}", flush=True)
    con.commit()
    print(f"DONE: {done} pairs processed | club_season_tm +{cs_rows} | player_tm_value +{pv}", flush=True)


if __name__ == "__main__":
    main()
