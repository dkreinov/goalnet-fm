"""Step 5: backfill substitution IDENTITY (who came on / off + minute) from ESPN summaries
already on disk. The original event-builder read keyEvents.athletesInvolved (empty for subs);
subs actually live in keyEvents[].participants ([0]=in, [1]=out) with clock.displayValue=minute.
Writes a match_sub table; links in/out ESPN athlete ids to our player_id for grade joins.
Usage: python D:/Programming/claude/FM/src/backfill_subs.py
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


def minute_of(ev):
    c = (ev.get("clock") or {}).get("displayValue") or ""
    m = re.match(r"(\d+)", c)
    return int(m.group(1)) if m else None


def run_code(con, code, win, comp_id, espn_pid, stats):
    try:
        sb = json.loads(fetch.get(SB.format(code=code, win=win), min_delay=0.3))
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
        if hg is None or ag is None:
            continue
        date = ev["date"][:10]
        mid = match_link.find_match(con, comp_id, date, hg, ag, home, away)
        if mid is None:
            continue
        if con.execute("SELECT 1 FROM match_sub WHERE match_id=? LIMIT 1", (mid,)).fetchone():
            stats["skip"] += 1
            continue
        eid = ev["id"]
        try:
            summ = json.loads(fetch.get(SUM.format(code=code, eid=eid), min_delay=0.3))
        except Exception:
            continue
        rows = []
        for e in summ.get("keyEvents", []):
            if (e.get("type") or {}).get("type") != "substitution":
                continue
            parts = e.get("participants") or []
            if len(parts) < 2:
                continue
            a_in = (parts[0].get("athlete") or {})
            a_out = (parts[1].get("athlete") or {})
            tname = (e.get("team") or {}).get("displayName")
            rows.append((mid, minute_of(e), tname,
                         a_in.get("id"), a_in.get("displayName"),
                         a_out.get("id"), a_out.get("displayName"),
                         espn_pid.get(str(a_in.get("id"))), espn_pid.get(str(a_out.get("id")))))
        if rows:
            con.executemany(
                """INSERT OR IGNORE INTO match_sub
                   (match_id, minute, team_name, in_espn_id, in_name, out_espn_id, out_name,
                    in_player_id, out_player_id) VALUES (?,?,?,?,?,?,?,?,?)""", rows)
            stats["subs"] += len(rows)
        stats["matches"] += 1
        if stats["matches"] % 1000 == 0:
            con.commit()
            print(f"  {stats['matches']} matches, {stats['subs']} subs, linked_in={stats['linked']}", flush=True)


def main():
    con = db.connect()
    con.execute("""CREATE TABLE IF NOT EXISTS match_sub(
        match_id INTEGER, minute INTEGER, team_name TEXT,
        in_espn_id TEXT, in_name TEXT, out_espn_id TEXT, out_name TEXT,
        in_player_id INTEGER, out_player_id INTEGER,
        PRIMARY KEY(match_id, in_espn_id, out_espn_id))""")
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    espn_pid = {str(sid): pid for sid, pid in con.execute(
        "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (espn_sid,))}
    stats = {"matches": 0, "subs": 0, "skip": 0, "linked": 0}
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=NORMAL")

    club = list(L.LEAGUES) + list(L.EXTRA_LEAGUES) + list(L.UEFA_CUPS)
    for lg in club:
        code = lg.get("espn")
        if not code:
            continue
        cid = db.competition_id(con, lg["name"])
        print(f"== {lg['name']} ==", flush=True)
        for win in L.espn_windows():
            run_code(con, code, win, cid, espn_pid, stats)
        con.commit()
    for (code, comp_name), windows in COMPETITIONS.items():
        cid = db.competition_id(con, comp_name)
        print(f"== {comp_name} ==", flush=True)
        for win in windows:
            run_code(con, code, win, cid, espn_pid, stats)
        con.commit()
    # link rate
    tot = con.execute("SELECT COUNT(*) FROM match_sub").fetchone()[0]
    lin = con.execute("SELECT COUNT(*) FROM match_sub WHERE in_player_id IS NOT NULL").fetchone()[0]
    print(f"\nmatch_sub: {tot:,} subs across {stats['matches']:,} matches; "
          f"in-player linked: {lin:,} ({100*lin/max(tot,1):.0f}%)")
    con.close()


if __name__ == "__main__":
    main()
