"""Fetch current FM26 (db7) grades for World Cup 2026 squad players via fminside's CLUB route (the
project's proven-reliable enumeration), since the player-table NAME search endpoint is unreliable.

Per nation that still has ungraded WC players: enumerate its clubs (club-table, nationality filter),
read each club's squad page (the /players/7-.../{uid}-{slug} links — slug == normalized name), build a
name pool, match the missing WC squad names against it (exact norm, else difflib>=0.90 within the
nation), then fetch + parse only the matched player pages and save as FM26 snapshots (grade_uid path).

Reliable + resumable: skips WC players already FM26-graded; single DB writer; paced for the fragile host.
Usage: python D:/Programming/claude/FM/src/scrape_wc2026_clubs.py [--only NAT[,NAT]] [--limit-nations N]
"""
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
import scrape_fminside as sf
import wc2026_squads as wc

FMV = 3
SNAP_DATE = "2026-03-01"

# worldcup nation name -> fminside nationality filter string (only the ones that differ / need care)
NAT_MAP = {
    "IR Iran": "Iran", "Korea Republic": "South Korea", "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde", "Congo DR": "DR Congo", "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey", "United States": "United States", "Curaçao": "Curacao",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
}
# worldcup club_country is mixed: full names AND 3-letter FIFA codes -> fminside nationality string
CC_MAP = {
    "QAT": "Qatar", "EGY": "Egypt", "ZAF": "South Africa", "BRA": "Brazil", "SAU": "Saudi Arabia",
    "JOR": "Jordan", "KOR": "South Korea", "BEL": "Belgium", "MEX": "Mexico", "UAE": "United Arab Emirates",
    "USA": "United States", "ENG": "England", "POR": "Portugal", "ESP": "Spain", "FRA": "France",
    "GER": "Germany", "ITA": "Italy", "NED": "Netherlands", "TUR": "Turkey", "GRE": "Greece",
    "IRN": "Iran", "IRQ": "Iraq", "UZB": "Uzbekistan", "TUN": "Tunisia", "MAR": "Morocco",
    "ALG": "Algeria", "GHA": "Ghana", "SEN": "Senegal", "CIV": "Ivory Coast", "RSA": "South Africa",
    "SUI": "Switzerland", "AUT": "Austria", "SCO": "Scotland", "CYP": "Cyprus", "CRO": "Croatia",
    "SRB": "Serbia", "RUS": "Russia", "UKR": "Ukraine", "JPN": "Japan", "KSA": "Saudi Arabia",
    "COL": "Colombia", "ARG": "Argentina", "URU": "Uruguay", "ECU": "Ecuador", "PAR": "Paraguay",
    "CHI": "Chile", "PER": "Peru", "AUS": "Australia",
}


def cc_norm(cc):
    """Normalize a worldcup club_country (full name or 3-letter code) to an fminside nationality."""
    if not cc:
        return None
    return CC_MAP.get(cc.upper(), cc) if len(cc) <= 3 else CC_MAP.get(cc, cc)


# fallback alternates to try if the primary returns 0 clubs
NAT_ALT = {
    "United States": ["USA"], "DR Congo": ["Congo DR", "Congo"], "South Korea": ["Korea Republic"],
    "Czech Republic": ["Czechia"], "Cape Verde": ["Cabo Verde"], "Ivory Coast": ["Cote d'Ivoire"],
    "Curacao": ["Curaçao"],
}


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


def new_session():
    s = requests.Session(); s.headers.update({"User-Agent": fetch.UA})
    s.get(f"{sf.BASE}/clubs", timeout=90); time.sleep(0.5)
    return s


CLUB_CACHE = db.ROOT / "data" / "wc_clubs_cache.json"


def _enum_once(nat):
    """One enumeration attempt for a nationality. Returns db7 club paths, or None if the session
    reverted to a non-db7 default (the throttle symptom: update_filter stopped applying)."""
    s = requests.Session(); s.headers.update({"User-Agent": fetch.UA})
    s.get(f"{sf.BASE}/clubs", timeout=90); time.sleep(0.5)
    s.post(sf.UPDATE_FILTER, data={**sf.FILTER_DEFAULTS, "page": "clubs",
           "database_version": "7", "league": "", "nationality": nat}, timeout=90)
    time.sleep(1.0)
    html = s.get(sf.CLUB_TABLE, timeout=90).text
    clubs = sorted(set(re.findall(r'href="(/clubs/7-[^"]+)"', html)))
    if clubs:
        return clubs
    # reverted? any non-db7 club links present = the filter dropped to the site-default edition
    other = re.findall(r'/clubs/(\d+)-', html)
    return None if other else []          # None = revert/throttle (retry); [] = genuinely no clubs


def enum_clubs(nat, cooldown=420):
    """Robust club-ID discovery for a nationality: tries name + alternates, and on a db-revert
    (throttle) does a quiet cooldown and retries. update_filter is the only fragile call and it's
    used just ONCE per nation here. Returns (club_paths, used_query)."""
    tries = [nat] + NAT_ALT.get(nat, [])
    for attempt in range(4):
        for q in tries:
            res = _enum_once(q)
            if res:
                return res, q
            if res is None:                 # throttle revert -> cooldown then retry
                print(f"    enum revert on '{q}' (throttle); cooldown {cooldown}s...", flush=True)
                time.sleep(cooldown)
                break                       # restart the tries after cooldown
            time.sleep(2)                   # res == [] (no clubs for this string) -> try next alt
    return [], nat


def squad_players(club_path):
    """Return [(uid, slug)] for a club squad page. URL-driven by db (the path carries 7-fm-26), so NO
    session/filter is needed — uses the shared disk-cached fetcher. This is the bulk of the requests
    and is now completely decoupled from the fragile update_filter endpoint."""
    try:
        html = fetch.get(sf.BASE + club_path, min_delay=1.4)
    except Exception:
        return []
    out, seen = [], set()
    for uid, slug in re.findall(r'/players/7-[^/]+/(\d+)-([a-z0-9-]+)', html):
        if uid not in seen:
            seen.add(uid); out.append((uid, slug))
    return out


def main():
    args = sys.argv[1:]
    only = set(args[args.index("--only") + 1].split(",")) if "--only" in args else None
    limit_nations = int(args[args.index("--limit-nations") + 1]) if "--limit-nations" in args else None

    sq = wc.load_squads()
    con = db.connect()
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=NORMAL")
    src = db.source_id(con, "fminside")     # integer source_id (FK target), not the name
    have = set(r[0] for r in con.execute(
        "SELECT DISTINCT p.norm_name FROM player_snapshot s JOIN player p ON p.player_id=s.player_id "
        "WHERE s.fm_version_id=?", (FMV,)))

    by_club = "--by-club" in args     # group by club_country (catches players abroad) vs player nation
    by_nat = defaultdict(list)
    for p in sq:
        if p["norm"] in have:
            continue
        key = cc_norm(p.get("club_country")) if by_club else p["nation"]
        if key:
            by_nat[key].append(p)
    nations = [n for n in by_nat if (only is None or n in only)]
    nations.sort(key=lambda n: -len(by_nat[n]))     # biggest gaps first
    if limit_nations:
        nations = nations[:limit_nations]
    print(f"{sum(len(by_nat[n]) for n in nations)} ungraded WC players across "
          f"{len(nations)} nations to scrape via club route", flush=True)

    # club-id enumerations are cached so a re-run never re-hits the fragile filter for a done nation
    import json
    cache = json.loads(CLUB_CACHE.read_text()) if CLUB_CACHE.exists() else {}

    counts = [0, 0, 0]
    grand_saved = grand_miss = 0
    for ni, nat in enumerate(nations):
        targets = by_nat[nat]
        tnorms = {t["norm"]: t for t in targets}
        fmnat = NAT_MAP.get(nat, nat)
        if cache.get(nat):
            clubs = cache[nat]
        else:
            clubs, used = enum_clubs(fmnat)
            cache[nat] = clubs
            CLUB_CACHE.write_text(json.dumps(cache))
        if not clubs:
            print(f"[{ni+1}/{len(nations)}] {nat}: 0 clubs (tried '{fmnat}') — SKIP, fix nationality string",
                  flush=True)
            grand_miss += len(targets); continue
        # squad pages are URL-driven (no filter) + disk-cached -> robust, no throttle pressure
        pool = {}      # norm slug-name -> (uid, slug)
        for cp in clubs:
            for uid, slug in squad_players(cp):
                pool[slug.replace("-", " ")] = (uid, slug)
        # match targets to pool
        pool_names = list(pool)
        matched = []
        for tn, t in tnorms.items():
            hit = tn if tn in pool else None
            if not hit:
                cm = get_close_matches(tn, pool_names, n=1, cutoff=0.90)
                hit = cm[0] if cm else None
            if hit:
                matched.append((t, *pool[hit], hit))
        # fetch + save matched player pages
        saved = 0
        for t, uid, slug, hit in matched:
            url = f"{sf.BASE}/players/7-fm-26/{uid}-{slug}"
            try:
                html = fetch.get(url, min_delay=1.8)
                pp = sf.parse_player(html)
            except Exception:
                pp = None
            if not pp or not pp.get("attrs"):
                continue
            before = counts[0]
            sf._save_parsed(con, src, FMV, SNAP_DATE, uid, url, pp, counts)
            if counts[0] > before:
                saved += 1
            if con.in_transaction and saved % 25 == 0:
                con.execute("COMMIT")
        if con.in_transaction:
            con.execute("COMMIT")
        grand_saved += saved; grand_miss += len(targets) - saved
        print(f"[{ni+1}/{len(nations)}] {nat:22s} clubs={len(clubs)} pool={len(pool)} "
              f"targets={len(targets)} matched={len(matched)} saved={saved}", flush=True)

    if con.in_transaction:
        con.execute("COMMIT")
    print(f"\nDONE: saved={grand_saved} still-missing={grand_miss}  (db saved/skip/err={counts})", flush=True)
    con.close()


if __name__ == "__main__":
    main()
