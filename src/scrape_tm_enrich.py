"""Transfermarkt squad enrichment: per (club, season) kader page -> squad market value, average age,
squad size, top-11 value. Squad market value is a top-tier match-outcome predictor. Reuses tm_club
(our club -> TM id). Stores club_season_tm(club_id, season, squad_value_eur, top11_value_eur, avg_age,
squad_size). build_dataset can then join home/away club-season to these features.
Single writer. Usage: python D:/Programming/claude/FM/src/scrape_tm_enrich.py [--sample N]
"""
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch

SQUAD = "https://www.transfermarkt.com/x/kader/verein/{cid}/saison_id/{yr}/plus/1"
HDR = {"Accept-Language": "en"}
YR = {"2019-20": "2019", "2020-21": "2020", "2021-22": "2021", "2022-23": "2022",
      "2023-24": "2023", "2024-25": "2024", "2025-26": "2025"}


def money(num, suf):
    try:
        v = float(num.replace(",", ""))
    except ValueError:
        return 0
    return int(v * {"m": 1e6, "k": 1e3, "th.": 1e3, "": 1}.get(suf, 1))


def parse_squad(html):
    """Return list of (age, market_value_eur) per player. Targets the SPECIFIC cells: market value is
    td.rechts.hauptlink ('€8.00m'); age is the '(NN)' in the birth-date td.zentriert ('19/04/1992 (33)').
    (A naive 'last € in the row' regex misfired on some pages.)"""
    s = BeautifulSoup(html, "lxml")
    out = []
    for tr in s.select("table.items > tbody > tr"):
        if not tr.select_one('td.posrela a[href*="/profil/spieler/"], td.hauptlink a[href*="/profil/spieler/"]'):
            continue
        mvtd = tr.select_one("td.rechts.hauptlink")
        if mvtd is None:
            continue                                 # no market-value cell -> not a real squad row
        m = re.search(r"€([\d.,]+)\s*(m|k|th\.)?", mvtd.get_text(strip=True))
        mv = money(m.group(1), m.group(2) or "") if m else 0
        age = None
        for td in tr.select("td.zentriert"):
            am = re.search(r"\b\d{2}/\d{2}/\d{4}\s*\((\d{2})\)", td.get_text(" ", strip=True))
            if am:
                age = int(am.group(1)); break
        out.append((age, mv))
    return out


def main():
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
    con = db.connect()
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("""CREATE TABLE IF NOT EXISTS club_season_tm (
        club_id INTEGER, season TEXT, squad_value_eur INTEGER, top11_value_eur INTEGER,
        avg_age REAL, squad_size INTEGER, PRIMARY KEY (club_id, season))""")
    # ONLY validated mappings (validate_tm_club.py): wrong/national-team mappings inject garbage values
    tm = {n: t for n, t in con.execute("SELECT club_name, tm_id FROM tm_club WHERE tm_id IS NOT NULL AND valid=1")}
    cname = {cid: nm for cid, nm in con.execute("SELECT club_id, name FROM club")}
    # all (club, season) pairs that appear in matches (home or away)
    pairs = set()
    for hc, ac, lab in con.execute(
            "SELECT m.home_club_id, m.away_club_id, s.label FROM match m JOIN season s ON s.season_id=m.season_id"):
        pairs.add((hc, lab)); pairs.add((ac, lab))
    todo = [(cid, lab) for cid, lab in pairs
            if lab in YR and cname.get(cid) in tm
            and not con.execute("SELECT 1 FROM club_season_tm WHERE club_id=? AND season=?", (cid, lab)).fetchone()]
    if sample:
        todo = todo[:sample]
    print(f"club-seasons to enrich: {len(todo):,}", flush=True)
    got = 0
    for i, (cid, lab) in enumerate(todo):
        tid = tm[cname[cid]]
        try:
            html = fetch.get(SQUAD.format(cid=tid, yr=YR[lab]), min_delay=2.5, timeout=40, headers=HDR)
        except Exception:
            continue
        sq = parse_squad(html)
        vals = sorted((mv for _, mv in sq if mv), reverse=True)
        ages = [a for a, _ in sq if a]
        if not vals:
            continue
        con.execute(
            "INSERT OR REPLACE INTO club_season_tm VALUES (?,?,?,?,?,?)",
            (cid, lab, sum(vals), sum(vals[:11]),
             round(sum(ages) / len(ages), 1) if ages else None, len(sq)))
        got += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)} club-seasons, {got} enriched", flush=True)
    print(f"\nclub-seasons enriched: {got}")
    cov = con.execute(
        """SELECT COUNT(*) FROM match m JOIN season s ON s.season_id=m.season_id
           JOIN club hc ON hc.club_id=m.home_club_id JOIN club ac ON ac.club_id=m.away_club_id
           WHERE EXISTS(SELECT 1 FROM club_season_tm t WHERE t.club_id=m.home_club_id AND t.season=s.label)
             AND EXISTS(SELECT 1 FROM club_season_tm t WHERE t.club_id=m.away_club_id AND t.season=s.label)""").fetchone()[0]
    tot = con.execute("SELECT COUNT(*) FROM match").fetchone()[0]
    print(f"matches with BOTH clubs' squad value: {cov:,}/{tot:,} ({100*cov//tot}%)")
    con.close()


if __name__ == "__main__":
    main()
