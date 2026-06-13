"""Collect national-team matches + lineups from ESPN (primary source — football-data has no
internationals). Creates national-team 'clubs', tags matches match_kind='national', maps each
match to the football season it falls in (so build_dataset joins players to that season's FM db).
Players are the same humans as in club squads, so they join to FM snapshots by name.
Usage: python D:/Programming/claude/FM/src/load_national_espn.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch

SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={win}&limit=1000"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={eid}"

# (espn_code, competition_name) -> list of date windows (<330 days each)
COMPETITIONS = {
    ("fifa.world", "FIFA World Cup"): ["20221101-20221220"],
    ("uefa.euro", "UEFA European Championship"): ["20210601-20210715", "20240601-20240715"],
    ("uefa.nations", "UEFA Nations League"):
        ["20200901-20201120", "20220601-20220701", "20220901-20221001",
         "20240901-20241120", "20250301-20250401", "20250601-20250701"],
    ("conmebol.america", "Copa America"): ["20210601-20210712", "20240620-20240715"],
    ("fifa.worldq.uefa", "WC Qualifying UEFA"):
        ["20210301-20211120", "20250301-20251120"],
    ("fifa.worldq.conmebol", "WC Qualifying CONMEBOL"):
        ["20210901-20220401", "20230901-20241120"],
    ("fifa.friendly", "International Friendly"):
        ["20210301-20211130", "20220301-20221130", "20230301-20231130",
         "20240301-20241130", "20250301-20251130", "20260301-20260601"],
}


def season_of(date):
    """Football season label for an ISO date (Aug-Jun boundary)."""
    y, m = int(date[:4]), int(date[5:7])
    start = y if m >= 7 else y - 1
    return f"{start}-{str(start + 1)[2:]}"


def main():
    con = db.connect()
    src = db.source_id(con, "espn", "https://site.api.espn.com")
    stats = {"matches": 0, "skipped": 0, "no_rosters": 0}
    for (code, comp_name), windows in COMPETITIONS.items():
        comp_id = db.competition_id(con, comp_name, country="International", tier=None,
                                    rank=None, kind="national")
        print(f"== {comp_name} ({code}) ==")
        for win in windows:
            try:
                sb = json.loads(fetch.get(SB.format(code=code, win=win), min_delay=0.6))
            except Exception as e:
                db.log(con, "espn-natl", f"{comp_name} {win}", "error", str(e))
                continue
            events = [e for e in sb.get("events", [])
                      if e.get("status", {}).get("type", {}).get("completed")]
            print(f"  {win}: {len(events)} events")
            for ev in events:
                eid = ev["id"]
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
                if hg is None or ag is None:
                    continue
                hcid, acid = db.club_id(con, home), db.club_id(con, away)
                sid = db.season_id(con, season_of(date))
                cur = con.execute(
                    """INSERT INTO match (season_id, competition_id, match_kind, match_date,
                          home_club_id, away_club_id, home_goals, away_goals)
                       VALUES (?,?,?,?,?,?,?,?)
                       ON CONFLICT(match_date, home_club_id, away_club_id) DO UPDATE SET
                          competition_id=excluded.competition_id, match_kind='national',
                          home_goals=excluded.home_goals, away_goals=excluded.away_goals""",
                    (sid, comp_id, "national", date, hcid, acid, hg, ag))
                mid = con.execute(
                    "SELECT match_id FROM match WHERE match_date=? AND home_club_id=? AND away_club_id=?",
                    (date, hcid, acid)).fetchone()[0]
                if con.execute("SELECT 1 FROM match_player WHERE match_id=? LIMIT 1", (mid,)).fetchone():
                    stats["skipped"] += 1
                    continue
                try:
                    summ = json.loads(fetch.get(SUM.format(code=code, eid=eid), min_delay=0.6))
                except Exception as e:
                    db.log(con, "espn-natl", eid, "error", f"summary: {e}")
                    continue
                rosters = summ.get("rosters", [])
                if not rosters or not any(r.get("roster") for r in rosters):
                    stats["no_rosters"] += 1
                    continue
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
                            (mid, pid, cid, started, None,
                             (entry.get("position") or {}).get("abbreviation")))
                con.commit()
                stats["matches"] += 1
        db.log(con, "espn-natl", comp_name, "ok", json.dumps(stats))
    print("FINAL:", stats)
    con.close()


if __name__ == "__main__":
    main()
