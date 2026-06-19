"""Backfill stadium (venue), attendance, kickoff time and formations into `match` rows that lack them.
The football-data-primary leagues (English lower tiers, LaLiga2, Serie B, Turkey, Ligue 2, Ger-2, ...)
got their matches from football-data (no venue) and lineups from load_lineups_espn (which never extracted
venue/attendance/formation). But ESPN's summary HAS those fields, and the summaries are already cached from
the lineup load -> this backfill is OFFLINE. Mirrors load_espn_league's gameInfo/roster extraction; only
UPDATEs rows where venue IS NULL (idempotent). Single writer.

Usage: python D:/Programming/claude/FM/src/backfill_venue.py [league_name ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
import leagues as L
import match_link

SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={win}&limit=1000"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={eid}"


def league_windows(lg):
    """ESPN-primary leagues tile the calendar; football-data leagues use the per-season Aug-Jun windows."""
    if lg.get("season_type"):
        return L.espn_windows()
    return list(L.ESPN_WINDOW.values())


def backfill_league(con, lg, stats):
    code = lg.get("espn")
    if not code:
        return
    comp_id = db.competition_id(con, lg["name"])
    # which of this league's matches still lack a venue?
    need = {r[0] for r in con.execute(
        "SELECT match_id FROM match WHERE competition_id=? AND venue IS NULL", (comp_id,))}
    if not need:
        return
    print(f"== {lg['name']}: {len(need):,} matches need venue ==", flush=True)
    for win in league_windows(lg):
        try:
            sb = json.loads(fetch.get(SB.format(code=code, win=win), min_delay=0.6))
        except Exception:
            continue
        for ev in sb.get("events", []):
            if not ev.get("status", {}).get("type", {}).get("completed"):
                continue
            comp = ev["competitions"][0]
            home = away = hg = ag = None
            for c in comp["competitors"]:
                try:
                    sc = int(c.get("score"))
                except (TypeError, ValueError):
                    sc = None
                if c["homeAway"] == "home":
                    home, hg = c["team"]["displayName"], sc
                else:
                    away, ag = c["team"]["displayName"], sc
            if hg is None or ag is None:
                continue
            mid = match_link.find_match(con, comp_id, ev["date"][:10], hg, ag, home, away)
            if mid is None or mid not in need:
                continue
            try:
                summ = json.loads(fetch.get(SUM.format(code=code, eid=ev["id"]), min_delay=0.6))
            except Exception:
                continue
            gi = summ.get("gameInfo", {})
            ven = gi.get("venue") or {}
            addr = ven.get("address") or {}
            hf = af = None
            for r in summ.get("rosters", []):
                if r.get("homeAway") == "home":
                    hf = r.get("formation")
                elif r.get("homeAway") == "away":
                    af = r.get("formation")
            con.execute(
                """UPDATE match SET kickoff_time=COALESCE(kickoff_time,?), venue=?, venue_city=?,
                     attendance=COALESCE(attendance,?), home_formation=COALESCE(home_formation,?),
                     away_formation=COALESCE(away_formation,?) WHERE match_id=?""",
                (ev.get("date"), ven.get("fullName"), addr.get("city"), gi.get("attendance"),
                 hf, af, mid))
            con.commit()
            need.discard(mid)
            stats["filled"] += 1
            if stats["filled"] % 200 == 0:
                print(f"   filled={stats['filled']:,}", flush=True)


def main():
    con = db.connect()
    con.execute("PRAGMA synchronous=NORMAL")
    args = sys.argv[1:]
    targets = ([L.BY_NAME.get(n) or L.EXTRA_BY_NAME[n] for n in args]
               if args else (L.LEAGUES + L.EXTRA_LEAGUES + L.UEFA_CUPS))
    stats = {"filled": 0}
    for lg in targets:
        backfill_league(con, lg, stats)
    print(f"\nvenue/attendance/formation backfilled: {stats['filled']:,} matches")
    con.close()


if __name__ == "__main__":
    main()
