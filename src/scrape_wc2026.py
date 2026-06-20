"""Fetch current (FM26 / db7) FM grades for World Cup 2026 squad players and load them into fm.db.

Squad names come from the sibling worldcup project (wc2026_squads.load_squads). Players already carrying
an FM26 snapshot (by exact norm-name) are left as-is; the rest are resolved live on fminside by NAME
search (the player-table name filter works; club/nationality filters do not), then disambiguated by club
(fuzzy) + age from each candidate's player page, and saved via the normal grade_uid snapshot path.

fminside name search (update_filter POST) is session/IP-stateful -> searches run SERIALLY; this is the
single DB writer (per project rule). Player pages are URL-driven and disk-cached.
Usage: python D:/Programming/claude/FM/src/scrape_wc2026.py [--limit N] [--report]
"""
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
import scrape_fminside as sf
import wc2026_squads as wc

FMV = 3                      # fm_version_id for FM26 (26.2.0)
SRC = "fminside"
SNAP_DATE = "2026-03-01"     # db7 snapshot date
DB7_PREFIX = "/players/7-"
PLAYER_TABLE = f"{sf.BASE}/beheer/modules/players/resources/inc/frontend/generate-player-table.php?ajax_request=1"


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


# fminside returns this tiny default table (one player) when it ignores/soft-blocks the filter.
DEGRADED_MARK = "oussama-benbout"


def _table_for(name):
    """One fresh-session name search (stateful filter must be reset per query). Returns raw html."""
    s = requests.Session(); s.headers.update({"User-Agent": fetch.UA})
    s.get(f"{sf.BASE}/players", timeout=90)
    time.sleep(0.4)
    s.post(sf.UPDATE_FILTER, data={**sf.FILTER_DEFAULTS, "page": "players",
           "database_version": "7", "name": name}, timeout=90)
    time.sleep(1.0)
    return s.get(PLAYER_TABLE, timeout=90).text


def _links(html):
    out, seen = [], set()
    for uid, slug in re.findall(r'/players/7-[^/]+/(\d+)-([a-z0-9-]+)', html):
        if uid not in seen:
            seen.add(uid); out.append((uid, slug))
    return out


def search_name(name):
    """One name-search of db7. Returns [(uid, slug)] on a healthy response, or None when the host is
    serving the throttled default table (tiny, single 'benbout' row). NO internal retry/ping-ladder —
    repeatedly pinging during a throttle only keeps the host blocked; the caller handles cooldown."""
    try:
        html = _table_for(name)
    except Exception:
        return None
    if len(html) < 5000:                 # throttled/empty default
        return None
    out = _links(html)
    # a big table that is ONLY benbout = also the degraded default
    if len(out) == 1 and out[0][1] == DEGRADED_MARK and name.replace(" ", "-") != DEGRADED_MARK:
        return None
    return out


def host_healthy():
    """Cheap preflight: a common name must return several hits. False => host is throttling us."""
    r = search_name("silva")
    return r is not None and len(r) >= 3


def resolve_player(p, con, counts):
    """Resolve one squad player to an fminside db7 page and save its grade. Returns status string."""
    name = p["name"]
    cands = search_name(db.norm(name))
    if cands is None:
        return "RATE_LIMITED"
    if not cands:                                   # retry on surname (last token)
        toks = db.norm(name).split()
        if len(toks) > 1:
            cands = search_name(toks[-1])
            if cands is None:
                return "RATE_LIMITED"
    if not cands:
        return "no_search_hit"
    target_club = db.norm(p["club"]) if p.get("club") else ""
    target_age = p.get("age")
    best, best_score, best_parsed, best_uid, best_url = None, -1, None, None, None
    for uid, slug in cands[:10]:
        url = f"{sf.BASE}/players/7-fm-26/{uid}-{slug}"
        try:
            html = fetch.get(url, min_delay=2.0)
        except Exception:
            continue
        pp = sf.parse_player(html)
        if not pp or not pp.get("attrs"):
            continue
        cclub = db.norm(pp.get("club") or "")
        csim = sim(target_club, cclub) if target_club and cclub else 0.0
        # age proximity (page Age vs squad age, allow +/-1, snapshot is a few months stale)
        cage = None
        m = re.search(r"(\d+)", html.split("Age", 1)[-1][:40]) if "Age" in html else None
        page_age = pp.get("age")
        ascore = 0.0
        if target_age and page_age:
            ascore = 1.0 if abs(page_age - target_age) <= 1 else (0.5 if abs(page_age - target_age) <= 2 else 0.0)
        nsim = sim(db.norm(name), db.norm(pp.get("name") or ""))
        score = 2.0 * csim + ascore + nsim
        if score > best_score:
            best, best_score, best_parsed, best_uid, best_url = (uid, slug), score, pp, uid, url
    if best_parsed is None:
        return "no_parsable_candidate"
    # accept if club matches well, OR single strong-name candidate with plausible age
    cclub = db.norm(best_parsed.get("club") or "")
    csim = sim(target_club, cclub) if target_club and cclub else 0.0
    nsim = sim(db.norm(name), db.norm(best_parsed.get("name") or ""))
    accept = csim >= 0.6 or (len(cands) == 1 and nsim >= 0.7) or (nsim >= 0.85 and best_score >= 1.5)
    if not accept:
        return f"low_conf(csim={csim:.2f},nsim={nsim:.2f},cands={len(cands)})"
    sf._save_parsed(con, SRC, FMV, SNAP_DATE, best_uid, best_url, best_parsed, counts)
    return f"SAVED uid={best_uid} club='{best_parsed.get('club')}' csim={csim:.2f}"


def already_graded(con):
    return set(r[0] for r in con.execute(
        "SELECT DISTINCT p.norm_name FROM player_snapshot s JOIN player p ON p.player_id=s.player_id "
        "WHERE s.fm_version_id=?", (FMV,)))


def main():
    sq = wc.load_squads()
    con = db.connect()
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=NORMAL")
    have = already_graded(con)
    todo = [p for p in sq if p["norm"] not in have]
    print(f"{len(sq)} WC players; {len(sq)-len(todo)} already FM26-graded; {len(todo)} to resolve", flush=True)
    if "--report" in sys.argv:
        return
    if "--limit" in sys.argv:
        todo = todo[:int(sys.argv[sys.argv.index("--limit") + 1])]
        print(f"  (limited to {len(todo)})", flush=True)

    # preflight: don't even start if the host is currently throttling us
    if not host_healthy():
        print("PREFLIGHT: host is throttling (no real results for a common name). "
              "Stopping — let the IP go quiet ~10-15 min, then re-run.", flush=True)
        con.close(); return

    COOLDOWN = 420            # quiet seconds when we hit a throttle mid-run
    counts = [0, 0, 0]       # saved, skipped, errors
    n_saved = n_fail = n_rl = 0
    cooldowns = 0
    i = 0
    while i < len(todo):
        p = todo[i]
        try:
            st = resolve_player(p, con, counts)
        except Exception as e:
            st = f"EXC {e}"
        if st == "RATE_LIMITED":
            # do ONE long quiet cooldown (no pinging), re-check health, then retry the SAME player
            if con.in_transaction:
                con.execute("COMMIT")
            cooldowns += 1
            if cooldowns > 4:
                print("  ABORT: still throttled after 4 cooldowns — stopping to avoid hammering.", flush=True)
                break
            print(f"  throttled at [{i+1}/{len(todo)}]; quiet cooldown {COOLDOWN}s "
                  f"(#{cooldowns})...", flush=True)
            time.sleep(COOLDOWN)
            if not host_healthy():
                time.sleep(COOLDOWN)        # one more quiet stretch before giving up this cooldown
            continue                        # retry same player
        # normal outcome
        if st.startswith("SAVED"):
            n_saved += 1
        else:
            n_fail += 1
        if con.in_transaction and (i + 1) % 25 == 0:
            con.execute("COMMIT")
        if (i + 1) % 10 == 0 or st.startswith("SAVED"):
            print(f"  [{i+1}/{len(todo)}] {p['nation']:14s} {p['name']:24s} -> {st}", flush=True)
        i += 1
        time.sleep(4.0)      # polite inter-player pacing
    if con.in_transaction:
        con.execute("COMMIT")
    print(f"\ndone: saved={n_saved} unresolved={n_fail} cooldowns={cooldowns}  "
          f"(db counts saved/skip/err={counts})", flush=True)
    con.close()


if __name__ == "__main__":
    main()
