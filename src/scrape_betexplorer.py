"""Scrape BetExplorer closing 1X2 odds for international matches (Phase-4 market anchor).

Enumerates competition-season results pages from BetExplorer's sitemap (robots-compliant; avoids the
soft-404 trap that path-guessing hits), then parses each results table into one CSV row per match.
Static HTML only, via the throttled disk-cached src/fetch.py — re-runs are free / resumable.

Output data/natl_odds_raw.csv columns:
  competition_id, season, date (DD.MM.YYYY), home, away, odd_h, odd_d, odd_a, knockout

Usage:
  python src/scrape_betexplorer.py --list          # enumerate matched pages, don't scrape
  python src/scrape_betexplorer.py --limit 3       # smoke: scrape first 3 pages
  python src/scrape_betexplorer.py                 # full scrape
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fetch
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "natl_odds_raw.csv"
TRAP_TITLE = "Football Stats, Results, Tables & H2H stats | BetExplorer"
SITEMAPS = [
    "https://www.betexplorer.com/sitemap/football/results.xml",
    "https://www.betexplorer.com/sitemap/football/results/other.xml",
    "https://www.betexplorer.com/sitemap/football/results/other2.xml",
    "https://www.betexplorer.com/sitemap/football/results/other3.xml",
]
SEASON_RE = re.compile(r"/football/([a-z-]+)/([a-z0-9-]+?)-(20(?:19|2[0-6])(?:-20\d\d)?)/results/?$")


def _compet_id(region, stem):
    """Map a BetExplorer (region, competition-stem) to our competition_id, or None to skip.
    Excludes women/club/youth/u-XX variants. Qualifiers routed by confederation."""
    s = stem
    if any(x in s for x in ("women", "-u1", "-u2", "youth", "club-", "regions", "futsal")):
        return None
    if "world-cup-qualification" in s or "world-cup-qual" in s:
        if "conmebol" in s or "south-america" in s or region == "south-america":
            return 14
        if "uefa" in s or "europe" in s or region == "europe":
            return 13
        return None  # other confederations' qualifiers aren't in our DB comps
    if s == "world-cup":
        return 9
    if s == "euro":
        return 10
    if s == "uefa-nations-league":
        return 11
    if s == "copa-america":
        return 12
    if s == "friendly-international":
        return 15
    return None


# Confirmed-real pages the sitemap omits (verified by direct title-check during Step-1 recon).
KNOWN_EXTRA = [
    ("https://www.betexplorer.com/football/south-america/copa-america-2024/results/", 12, "2024"),
]


def enumerate_pages():
    """Return [(url, competition_id, season), ...] for our national comps, seasons 2020-2026."""
    pages, seen = [], set()
    for u, cid, s in KNOWN_EXTRA:
        seen.add(u); pages.append((u, cid, s))
    for sm in SITEMAPS:
        try:
            xml = fetch.get(sm, min_delay=1.4, retries=1, timeout=30, cache=False)
        except Exception as e:
            print(f"  sitemap {sm.rsplit('/',1)[1]}: skip ({str(e)[:40]})", flush=True)
            continue
        for loc in re.findall(r"<loc>([^<]+)</loc>", xml):
            m = SEASON_RE.search(loc)
            if not m:
                continue
            region, stem, season = m.group(1), m.group(2), m.group(3)
            if int(season[:4]) < 2020:
                continue
            cid = _compet_id(region, stem)
            if cid and loc not in seen:
                seen.add(loc)
                pages.append((loc, cid, season))
    return pages


def _stage_urls(base_url, html):
    """All stage variants of a competition page: base + each ?stage=<id> from the stage-nav menus
    (list-tabs--secondary). Captures qualification + final-tournament stages the base page hides."""
    soup = BeautifulSoup(html, "html.parser")
    ids = []
    for a in soup.select("ul.list-tabs--secondary a[href*='stage=']"):
        m = re.search(r"stage=([A-Za-z0-9]{6,12})", a.get("href", ""))
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return [base_url] + [f"{base_url}?stage={sid}" for sid in ids]


def _parse_rows(html, cid, season):
    """Parse match rows from one results/stage page (returns [] for the soft-404 trap)."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    if title == TRAP_TITLE:
        return []
    rows = []
    for tr in soup.select("tr"):
        a = tr.select_one("a.in-match")
        odds = tr.select("[data-odd]")
        if not a or len(odds) < 3:
            continue
        spans = a.select("span")
        if len(spans) < 2:
            continue
        home = spans[0].get_text(strip=True)
        away = spans[-1].get_text(strip=True)
        sc = tr.select_one("td.h-text-center")
        sc_txt = sc.get_text(strip=True) if sc else ""
        knockout = 1 if re.search(r"AfP|ET|pen|awrd", sc_txt, re.I) else 0
        date_td = tr.select("td")[-1].get_text(strip=True)
        if not re.match(r"\d{2}\.\d{2}\.\d{4}$", date_td):
            continue
        try:
            oh, od, oa = (float(odds[i].get("data-odd")) for i in range(3))
        except (TypeError, ValueError):
            continue
        if not (home and away and oh > 1 and od > 1 and oa > 1):
            continue
        rows.append([cid, season, date_td, home, away, oh, od, oa, knockout])
    return rows


def parse_page(url, cid, season):
    """Fetch a competition-season page and ALL its ?stage= variants; return deduped match rows.
    The base page shows only the final stage; the stage variants add group/qualification rounds."""
    base = fetch.get(url, min_delay=1.5, retries=2, timeout=45)
    if (BeautifulSoup(base, "html.parser").title.get_text(strip=True) if "<title" in base else "") == TRAP_TITLE:
        return []
    seen, rows = set(), []
    for su in _stage_urls(url, base):
        html = base if su == url else fetch.get(su, min_delay=1.5, retries=2, timeout=45)
        for r in _parse_rows(html, cid, season):
            key = (r[2], r[3], r[4])                 # (date, home, away) — dedup across stages
            if key not in seen:
                seen.add(key); rows.append(r)
    return rows


def main():
    args = sys.argv
    pages = enumerate_pages()
    print(f"enumerated {len(pages)} national results pages (2020-2026)", flush=True)
    by_cid = {}
    for _, cid, _ in pages:
        by_cid[cid] = by_cid.get(cid, 0) + 1
    print("  pages per competition_id:", dict(sorted(by_cid.items())), flush=True)
    if "--list" in args:
        for u, cid, s in pages:
            print(f"  [{cid}] {s}  {u.split('betexplorer.com')[1]}")
        return
    if "--limit" in args:
        pages = pages[:int(args[args.index("--limit") + 1])]
    all_rows = []
    for i, (u, cid, s) in enumerate(pages):
        try:
            r = parse_page(u, cid, s)
        except Exception as e:
            print(f"  ERR {u.split('/results')[0].rsplit('/',1)[1]}: {str(e)[:50]}", flush=True)
            continue
        all_rows += r
        if r:
            print(f"  [{i+1}/{len(pages)}] cid={cid} {s}: {len(r)} matches", flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["competition_id", "season", "date", "home", "away", "odd_h", "odd_d", "odd_a", "knockout"])
        w.writerows(all_rows)
    print(f"\nwrote {OUT}: {len(all_rows)} rows", flush=True)
    ck = {}
    for row in all_rows:
        ck[row[0]] = ck.get(row[0], 0) + 1
    print("  rows per competition_id:", dict(sorted(ck.items())), flush=True)


if __name__ == "__main__":
    main()
