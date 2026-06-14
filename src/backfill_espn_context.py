"""Backfill match context (kickoff time, venue, city, attendance, referee, formations) and the
minute-stamped event timeline from ESPN summaries already on disk (cache). Near-zero new network.
Covers club leagues (leagues registry) + national competitions (load_national_espn.COMPETITIONS).
Usage: python D:/Programming/claude/FM/src/backfill_espn_context.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
import leagues as L
import match_link
from load_national_espn import COMPETITIONS, season_of

SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={win}&limit=1000"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={eid}"


def parse_minute(ev):
    c = (ev.get("clock") or {}).get("displayValue") or ""
    m = re.match(r"(\d+)", c)
    return int(m.group(1)) if m else None


def backfill_event(con, code, win, comp_id, kind, stats):
    try:
        sb = json.loads(fetch.get(SB.format(code=code, win=win), min_delay=0.4))
    except Exception:
        return
    for ev in sb.get("events", []):
        if not ev.get("status", {}).get("type", {}).get("completed"):
            continue
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
        mid = match_link.find_match(con, comp_id, date, hg, ag, home, away)
        if mid is None:
            continue
        eid = ev["id"]
        try:
            summ = json.loads(fetch.get(SUM.format(code=code, eid=eid), min_delay=0.4))
        except Exception:
            continue
        gi = summ.get("gameInfo", {})
        ven = gi.get("venue") or {}
        addr = ven.get("address") or {}
        officials = [o.get("displayName") for o in gi.get("officials", []) if o.get("displayName")]
        rosters = summ.get("rosters", [])
        hf = af = None
        for r in rosters:
            if r.get("homeAway") == "home":
                hf = r.get("formation")
            elif r.get("homeAway") == "away":
                af = r.get("formation")
        con.execute(
            """UPDATE match SET kickoff_time=COALESCE(?,kickoff_time),
                 venue=COALESCE(?,venue), venue_city=COALESCE(?,venue_city),
                 attendance=COALESCE(?,attendance),
                 home_formation=COALESCE(?,home_formation), away_formation=COALESCE(?,away_formation),
                 referee=COALESCE(referee,?)
               WHERE match_id=?""",
            (ev.get("date"), ven.get("fullName"), addr.get("city"),
             gi.get("attendance"), hf, af, officials[0] if officials else None, mid))
        # event timeline
        hcid, acid = con.execute(
            "SELECT home_club_id, away_club_id FROM match WHERE match_id=?", (mid,)).fetchone()
        ke = summ.get("keyEvents", [])
        if ke and not con.execute("SELECT 1 FROM match_event WHERE match_id=? LIMIT 1", (mid,)).fetchone():
            rows = []
            for i, e in enumerate(ke):
                etype = (e.get("type") or {}).get("text") or "?"
                tname = (e.get("team") or {}).get("displayName")
                side = club = None
                if tname:
                    if db.norm(tname) == db.norm(con.execute("SELECT name FROM club WHERE club_id=?", (hcid,)).fetchone()[0]):
                        side, club = "home", hcid
                    else:
                        side, club = "away", acid
                ath = e.get("athletesInvolved") or []
                pl = ath[0].get("displayName") if ath and ath[0].get("displayName") else None
                rows.append((mid, i, parse_minute(e), etype, side, club, pl,
                             (e.get("type") or {}).get("text")))
            con.executemany(
                """INSERT OR IGNORE INTO match_event
                   (match_id, seq, minute, type, team_side, club_id, player, detail)
                   VALUES (?,?,?,?,?,?,?,?)""", rows)
            stats["events"] += len(rows)
        stats["ctx"] += 1
        if stats["ctx"] % 500 == 0:
            con.commit()
            print(f"  context backfilled: {stats['ctx']} matches, {stats['events']} events", flush=True)


def main():
    con = db.connect()
    stats = {"ctx": 0, "events": 0}
    print("== club leagues ==")
    for lg in L.enabled():
        comp_id = db.competition_id(con, lg["name"])
        for season, win in L.ESPN_WINDOW.items():
            backfill_event(con, lg["espn"], win, comp_id, "league", stats)
        con.commit()
        print(f"{lg['name']}: ctx={stats['ctx']} events={stats['events']}")
    print("== national ==")
    for (code, comp_name), windows in COMPETITIONS.items():
        comp_id = db.competition_id(con, comp_name)
        for win in windows:
            backfill_event(con, code, win, comp_id, "national", stats)
        con.commit()
    con.commit()
    print(f"DONE: {stats['ctx']} matches context-filled, {stats['events']} events")
    # report
    for col in ("kickoff_time", "venue", "attendance", "home_formation", "referee"):
        n = con.execute(f"SELECT COUNT(*) FROM match WHERE {col} IS NOT NULL").fetchone()[0]
        print(f"  matches with {col}: {n:,}")
    print(f"  total match_event rows: {con.execute('SELECT COUNT(*) FROM match_event').fetchone()[0]:,}")
    con.close()


if __name__ == "__main__":
    main()
