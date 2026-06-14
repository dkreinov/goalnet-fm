"""ESPN-primary club-league loader for leagues without football-data (Brazil, Argentina, MLS,
Saudi, Israel, etc.). Creates match rows + lineups + context (kickoff/venue/attendance/formation)
+ minute-stamped events, all from ESPN. FM grades come later from fminside.
Usage: python D:/Programming/claude/FM/src/load_espn_league.py [league_name ...]
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
import leagues as L
from load_national_espn import season_of

SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={win}&limit=1000"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={eid}"


def parse_minute(ev):
    m = re.match(r"(\d+)", (ev.get("clock") or {}).get("displayValue") or "")
    return int(m.group(1)) if m else None


def load_league(con, lg, src, stats):
    comp_id = db.competition_id(con, lg["name"], lg["country"], lg["tier"], lg["rank"], "league")
    print(f"== {lg['name']} (rank {lg['rank']}, {lg['espn']}) ==", flush=True)
    seen_local = 0
    for win in L.espn_windows():
        try:
            sb = json.loads(fetch.get(SB.format(code=lg["espn"], win=win), min_delay=0.5))
        except Exception as e:
            db.log(con, "espn-league", f"{lg['name']} {win}", "error", str(e)[:120])
            continue
        events = [e for e in sb.get("events", [])
                  if e.get("status", {}).get("type", {}).get("completed")]
        for ev in events:
            comp = ev["competitions"][0]
            home = away = hg = ag = None
            for c in comp["competitors"]:
                nm = c["team"]["displayName"]
                try:
                    sc = int(c.get("score"))
                except (TypeError, ValueError):
                    sc = None
                if c["homeAway"] == "home":
                    home, hg = nm, sc
                else:
                    away, ag = nm, sc
            date = ev["date"][:10]
            if hg is None or ag is None or not home or not away:
                continue
            hcid, acid = db.club_id(con, home), db.club_id(con, away)
            sid = db.season_id(con, season_of(date))
            con.execute(
                """INSERT INTO match (season_id, competition_id, match_kind, match_date,
                      home_club_id, away_club_id, home_goals, away_goals)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(match_date, home_club_id, away_club_id) DO UPDATE SET
                      competition_id=excluded.competition_id, home_goals=excluded.home_goals,
                      away_goals=excluded.away_goals""",
                (sid, comp_id, "league", date, hcid, acid, hg, ag))
            mid = con.execute(
                "SELECT match_id FROM match WHERE match_date=? AND home_club_id=? AND away_club_id=?",
                (date, hcid, acid)).fetchone()[0]
            if con.execute("SELECT 1 FROM match_player WHERE match_id=? LIMIT 1", (mid,)).fetchone():
                continue  # lineups already loaded
            try:
                summ = json.loads(fetch.get(SUM.format(code=lg["espn"], eid=ev["id"]), min_delay=0.5))
            except Exception:
                continue
            rosters = summ.get("rosters", [])
            gi = summ.get("gameInfo", {})
            ven = gi.get("venue") or {}
            addr = ven.get("address") or {}
            officials = [o.get("displayName") for o in gi.get("officials", []) if o.get("displayName")]
            hf = af = None
            for r in rosters:
                if r.get("homeAway") == "home":
                    hf = r.get("formation")
                elif r.get("homeAway") == "away":
                    af = r.get("formation")
            con.execute(
                """UPDATE match SET kickoff_time=?, venue=?, venue_city=?, attendance=?,
                     home_formation=?, away_formation=?, referee=COALESCE(referee,?) WHERE match_id=?""",
                (ev.get("date"), ven.get("fullName"), addr.get("city"), gi.get("attendance"),
                 hf, af, officials[0] if officials else None, mid))
            for tr in rosters:
                cid = hcid if tr.get("homeAway") == "home" else acid
                for entry in tr.get("roster", []):
                    ath = entry.get("athlete") or {}
                    pn = ath.get("displayName")
                    if not pn:
                        continue
                    pid = db.player_id(con, pn, src=src, src_player_id=ath.get("id"))
                    started = 1 if entry.get("starter") else 0
                    con.execute(
                        """INSERT OR REPLACE INTO match_player
                           (match_id, player_id, club_id, started, minutes, position)
                           VALUES (?,?,?,?,?,?)""",
                        (mid, pid, cid, started, None, (entry.get("position") or {}).get("abbreviation")))
            ke = summ.get("keyEvents", [])
            if ke:
                rows = []
                hname = con.execute("SELECT name FROM club WHERE club_id=?", (hcid,)).fetchone()[0]
                for i, e in enumerate(ke):
                    tn = (e.get("team") or {}).get("displayName")
                    side = club = None
                    if tn:
                        side, club = ("home", hcid) if db.norm(tn) == db.norm(hname) else ("away", acid)
                    ath = e.get("athletesInvolved") or []
                    rows.append((mid, i, parse_minute(e), (e.get("type") or {}).get("text") or "?",
                                 side, club, ath[0].get("displayName") if ath else None, None))
                con.executemany(
                    """INSERT OR IGNORE INTO match_event
                       (match_id, seq, minute, type, team_side, club_id, player, detail)
                       VALUES (?,?,?,?,?,?,?,?)""", rows)
                stats["events"] += len(rows)
            stats["lineups"] += 1
            seen_local += 1
            if seen_local % 100 == 0:
                con.commit()
                print(f"  {lg['name']}: {seen_local} matches w/lineups, {stats['events']} events", flush=True)
    con.commit()
    db.log(con, "espn-league", lg["name"], "ok", f"lineups+={seen_local}")
    print(f"{lg['name']} DONE: {seen_local} matches with lineups", flush=True)


def main():
    con = db.connect()
    src = db.source_id(con, "espn", "https://site.api.espn.com")
    names = sys.argv[1:]
    targets = [L.EXTRA_BY_NAME[n] for n in names] if names else L.EXTRA_LEAGUES
    stats = {"lineups": 0, "events": 0}
    for lg in targets:
        load_league(con, lg, src, stats)
    print(f"ALL DONE: {stats['lineups']} matches w/lineups, {stats['events']} events")
    con.close()


if __name__ == "__main__":
    main()
