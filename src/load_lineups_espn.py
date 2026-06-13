"""Load EPL lineups from ESPN hidden API into match_player.
Scoreboard: site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates=YYYYMMDD-YYYYMMDD&limit=1000
Summary:    .../summary?event={id} -> rosters (11 starters + bench, sub flags)
Usage: python D:/Programming/claude/FM/src/load_lineups_espn.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch

SEASONS = {
    "2023-24": "20230801-20240601",
    "2024-25": "20240801-20250601",
    "2025-26": "20250801-20260601",
}
SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={win}&limit=1000"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary?event={eid}"

# ESPN displayName -> canonical (football-data) club name
ALIASES = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest", "AFC Bournemouth": "Bournemouth",
    "Wolverhampton Wanderers": "Wolves", "Brighton & Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham",
    "Newcastle United": "Newcastle", "Luton Town": "Luton",
    "Ipswich Town": "Ipswich", "Leicester City": "Leicester",
    "Leeds United": "Leeds", "Sheffield United": "Sheffield United",
}


def minutes_from_stats(entry):
    for s in entry.get("stats", []):
        if s.get("name") in ("minutes", "appearances-minutes", "MIN"):
            try:
                return int(float(s.get("value")))
            except (TypeError, ValueError):
                return None
    return None


def main():
    con = db.connect()
    src = db.source_id(con, "espn", "https://site.api.espn.com")
    for alias, canon in ALIASES.items():
        db.add_club_alias(con, alias, canon)
    con.commit()

    done_matches = {r[0] for r in con.execute(
        "SELECT DISTINCT match_id FROM match_player").fetchall()}

    stats = {"matched": 0, "no_match_row": 0, "no_rosters": 0, "skipped": 0}
    for label, win in SEASONS.items():
        sb = json.loads(fetch.get(SB.format(win=win), min_delay=0.7))
        events = [e for e in sb.get("events", [])
                  if e.get("status", {}).get("type", {}).get("completed")]
        print(f"{label}: {len(events)} completed events")
        for i, ev in enumerate(events):
            eid = ev["id"]
            comp = ev["competitions"][0]
            home = away = None
            for c in comp["competitors"]:
                name = c["team"]["displayName"]
                if c["homeAway"] == "home":
                    home = name
                else:
                    away = name
            date = ev["date"][:10]
            hid, aid = db.club_id(con, home), db.club_id(con, away)
            row = con.execute(
                """SELECT match_id FROM match WHERE home_club_id=? AND away_club_id=?
                   AND date(match_date) BETWEEN date(?, '-1 day') AND date(?, '+1 day')""",
                (hid, aid, date, date)).fetchone()
            if not row:
                stats["no_match_row"] += 1
                db.log(con, "espn", eid, "error", f"no match row {date} {home} v {away}")
                continue
            mid = row[0]
            if mid in done_matches:
                stats["skipped"] += 1
                continue
            try:
                summ = json.loads(fetch.get(SUM.format(eid=eid), min_delay=0.7))
            except Exception as e:
                db.log(con, "espn", eid, "error", f"summary fetch: {e}")
                continue
            rosters = summ.get("rosters", [])
            if not rosters or not any(r.get("roster") for r in rosters):
                stats["no_rosters"] += 1
                db.log(con, "espn", eid, "skip", f"no rosters {date} {home} v {away}")
                continue
            for team_roster in rosters:
                tname = team_roster.get("team", {}).get("displayName", "")
                cid = db.club_id(con, tname)
                for entry in team_roster.get("roster", []):
                    ath = entry.get("athlete") or {}
                    pname = ath.get("displayName")
                    if not pname:
                        continue
                    pid = db.player_id(con, pname, src=src, src_player_id=ath.get("id"))
                    started = 1 if entry.get("starter") else 0
                    if not started and not entry.get("subbedIn"):
                        played_min = 0
                    else:
                        played_min = minutes_from_stats(entry)
                    con.execute(
                        """INSERT OR REPLACE INTO match_player
                           (match_id, player_id, club_id, started, minutes, position)
                           VALUES (?,?,?,?,?,?)""",
                        (mid, pid, cid, started, played_min,
                         (entry.get("position") or {}).get("abbreviation")))
            con.commit()
            stats["matched"] += 1
            if (i + 1) % 25 == 0:
                print(f"  {label}: {i+1}/{len(events)} done", flush=True)
        print(f"{label} complete: {stats}")
    db.log(con, "espn", "", "ok", json.dumps(stats))
    print("FINAL:", stats)
    con.close()


if __name__ == "__main__":
    main()
