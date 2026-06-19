"""Plan 2 (sample): targeted FM-grade lookup for starters genuinely ABSENT from our FM grade data
(no token overlap with any graded FM name). Builds a work-list and probes fminside name-search.

--worklist : print the top-N absent players (name, nationality, clubs, seasons, appearances). No network.
--probe NAME : live-search fminside for NAME, print candidate player pages (1 request).
Usage: python D:/Programming/claude/FM/src/scrape_absent.py --worklist 100
"""
import re
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
from build_xwalk import xnorm
import scrape_fminside as sf


def absent_worklist(con, topn=100):
    fm_src = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    gp = set(r[0] for r in con.execute("SELECT DISTINCT player_id FROM player_snapshot"))
    pname = {pid: nm for pid, nm in con.execute("SELECT player_id, name FROM player")}
    # FM graded distinctive tokens
    fm_tok = set()
    for pid in gp:
        for t in xnorm(pname.get(pid, "")).split():
            if len(t) >= 4:
                fm_tok.add(t)
    # covered set (xwalk uid w/ grade OR roster)
    pid_eids = defaultdict(list)
    for eid, pid in con.execute("SELECT source_player_id, player_id FROM player_source_id WHERE source_id=1"):
        pid_eids[pid].append(eid)
    eid_uid = {r[0]: r[1] for r in con.execute(
        "SELECT espn_player_id, fm_uid FROM player_xwalk WHERE fm_uid IS NOT NULL")}
    uid_pid = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute("SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_pid[uid].add(pid)
    uid_g = set(u for u, p in uid_pid.items() if p & gp)
    roster = set(p for p, in con.execute("SELECT DISTINCT player_id FROM match_grade_link WHERE fm_uid IS NOT NULL"))
    ap = Counter()
    for pid, n in con.execute("SELECT player_id, COUNT(*) FROM match_player WHERE started=1 GROUP BY player_id"):
        ap[pid] = n
    nat = {e: n for e, n in con.execute("SELECT source_player_id, nationality FROM source_identity WHERE source_id=1")}
    clubs_of = defaultdict(set)
    cname = {c: n for c, n in con.execute("SELECT club_id, name FROM club")}
    for pid, cid in con.execute("SELECT DISTINCT player_id, club_id FROM match_player"):
        clubs_of[pid].add(cname.get(cid))
    seasons_of = defaultdict(set)
    for pid, lab in con.execute(
            "SELECT DISTINCT mp.player_id, s.label FROM match_player mp JOIN match m ON m.match_id=mp.match_id "
            "JOIN season s ON s.season_id=m.season_id"):
        seasons_of[pid].add(lab)
    work = []
    for pid, n in ap.items():
        uids = {eid_uid.get(e) for e in pid_eids.get(pid, [])}
        if (uids & uid_g) or pid in roster:
            continue
        nm = xnorm(pname.get(pid, ""))
        if not nm or any(t in fm_tok for t in nm.split() if len(t) >= 4):
            continue  # has token overlap -> not 'truly absent'
        eid = pid_eids.get(pid, [None])[0]
        work.append({"pid": pid, "name": pname.get(pid), "nat": nat.get(eid),
                     "clubs": sorted(c for c in clubs_of.get(pid, ()) if c),
                     "seasons": sorted(seasons_of.get(pid, ())), "apps": n})
    work.sort(key=lambda w: -w["apps"])
    return work[:topn]


def probe_fminside(name, nat=""):
    """Live: search fminside players by free-text name (+optional nationality). Returns candidate
    (uid, dbid, url) from the player table. Uses db7 (FM26) session."""
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": fetch.UA})
    s.get(f"{sf.BASE}/players", timeout=90)
    s.post(sf.UPDATE_FILTER, data={**sf.FILTER_DEFAULTS, "page": "players",
                                   "database_version": "7", "name": name, "nationality": nat}, timeout=90)
    time.sleep(1)
    # player table ajax (mirror of CLUB_TABLE but for players)
    url = f"{sf.BASE}/beheer/modules/players/resources/inc/frontend/generate-player-table.php?ajax_request=1"
    html = s.get(url, timeout=90).text
    links = sorted(set(re.findall(r'href="(/players/\d[^"]+)"', html)))
    return html, links


def main():
    con = db.connect()
    args = sys.argv[1:]
    if "--worklist" in args:
        n = int(args[args.index("--worklist") + 1]) if len(args) > args.index("--worklist") + 1 else 100
        work = absent_worklist(con, n)
        print(f"TOP {len(work)} truly-absent starters (no FM token overlap):")
        for w in work:
            print(f"  x{w['apps']:>3}  {w['name']}  [{w['nat']}]  clubs={w['clubs'][:3]}  seasons={w['seasons']}")
    elif "--probe" in args:
        name = args[args.index("--probe") + 1]
        html, links = probe_fminside(name)
        print(f"probe '{name}': {len(html)} bytes, {len(links)} player links")
        for l in links[:15]:
            print("   ", l)
    con.close()


if __name__ == "__main__":
    main()
