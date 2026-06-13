"""Load lineups from ESPN hidden API for every registered league x season.
Links each ESPN event to our football-data `match` row by (competition, date, score) so club
identity stays consistent with football-data; players are assigned to that match's home/away club.
Resumable (disk cache) and idempotent. Usage:
  python D:/Programming/claude/FM/src/load_lineups_espn.py [league_name ...] [--season 2023-24]
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


def minutes_from_stats(entry):
    for s in entry.get("stats", []):
        if s.get("name") in ("minutes", "appearances-minutes", "MIN"):
            try:
                return int(float(s.get("value")))
            except (TypeError, ValueError):
                return None
    return None


def load_league_season(con, lg, comp_id, season, win, stats):
    code = lg["espn"]
    try:
        sb = json.loads(fetch.get(SB.format(code=code, win=win), min_delay=0.6))
    except Exception as e:
        db.log(con, "espn", f"{lg['name']} {season} scoreboard", "error", str(e))
        return
    events = [e for e in sb.get("events", [])
              if e.get("status", {}).get("type", {}).get("completed")]
    print(f"  {lg['name']} {season}: {len(events)} completed events")
    done = {r[0] for r in con.execute(
        "SELECT DISTINCT match_id FROM match_player WHERE match_id IN "
        "(SELECT match_id FROM match WHERE competition_id=?)", (comp_id,)).fetchall()}
    for i, ev in enumerate(events):
        eid = ev["id"]
        comp = ev["competitions"][0]
        home = away = hg = ag = None
        for c in comp["competitors"]:
            name = c["team"]["displayName"]
            try:
                score = int(c.get("score"))
            except (TypeError, ValueError):
                score = None
            if c["homeAway"] == "home":
                home, hg = name, score
            else:
                away, ag = name, score
        date = ev["date"][:10]
        if hg is None or ag is None:
            stats["no_score"] += 1
            continue
        mid = match_link.find_match(con, comp_id, date, hg, ag, home, away)
        if mid is None:
            stats["no_match_row"] += 1
            db.log(con, "espn", eid, "skip", f"no match {lg['name']} {date} {home} {hg}-{ag} {away}")
            continue
        if mid in done:
            stats["skipped"] += 1
            continue
        hcid, acid = con.execute(
            "SELECT home_club_id, away_club_id FROM match WHERE match_id=?", (mid,)).fetchone()
        try:
            summ = json.loads(fetch.get(SUM.format(code=code, eid=eid), min_delay=0.6))
        except Exception as e:
            db.log(con, "espn", eid, "error", f"summary: {e}")
            continue
        rosters = summ.get("rosters", [])
        if not rosters or not any(r.get("roster") for r in rosters):
            stats["no_rosters"] += 1
            continue
        for team_roster in rosters:
            ha = team_roster.get("homeAway")
            cid = hcid if ha == "home" else acid
            for entry in team_roster.get("roster", []):
                ath = entry.get("athlete") or {}
                pname = ath.get("displayName")
                if not pname:
                    continue
                pid = db.player_id(con, pname, src=db.source_id(con, "espn"),
                                   src_player_id=ath.get("id"))
                started = 1 if entry.get("starter") else 0
                played = 0 if (not started and not entry.get("subbedIn")) else minutes_from_stats(entry)
                con.execute(
                    """INSERT OR REPLACE INTO match_player
                       (match_id, player_id, club_id, started, minutes, position)
                       VALUES (?,?,?,?,?,?)""",
                    (mid, pid, cid, started, played, (entry.get("position") or {}).get("abbreviation")))
        con.commit()
        stats["matched"] += 1
        if (i + 1) % 50 == 0:
            print(f"    {lg['name']} {season}: {i+1}/{len(events)}  matched={stats['matched']}", flush=True)


def main():
    args = sys.argv[1:]
    only_season = None
    if "--season" in args:
        k = args.index("--season")
        only_season = args[k + 1]
        args = args[:k] + args[k + 2:]
    con = db.connect()
    db.source_id(con, "espn", "https://site.api.espn.com")
    targets = [L.BY_NAME[n] for n in args] if args else L.enabled()
    stats = {"matched": 0, "no_match_row": 0, "no_rosters": 0, "no_score": 0, "skipped": 0}
    for lg in targets:
        comp_id = db.competition_id(con, lg["name"])
        print(f"== {lg['name']} (rank {lg['rank']}) ==")
        for season, win in L.ESPN_WINDOW.items():
            if only_season and season != only_season:
                continue
            load_league_season(con, lg, comp_id, season, win, stats)
        db.log(con, "espn", lg["name"], "ok", json.dumps(stats))
    print("FINAL:", stats)
    con.close()


if __name__ == "__main__":
    main()
