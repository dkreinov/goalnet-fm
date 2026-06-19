"""Why does build_xwalk's name+squad fail for Brazil/Portugal common names (post-kaggle-club)?
Replicates the name+squad decision for each unmatched ESPN starter in the given leagues and reports the
candidate-in-bridged-club count (cmatch): 0 = bridge doesn't reach the club (bridge-thin, fixable by a
stronger bridge); >1 = several same-name players in the club (genuinely ambiguous — DOB needed); and for
cmatch>1, whether ESPN DOB uniquely picks one (i.e. DOB-recoverable).

Usage: python D:/Programming/claude/FM/src/diag_brazil.py "Brazil Serie A" "Portugal Primeira Liga"
"""
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
from build_xwalk import xnorm, dob_close

LEAGUES = sys.argv[1:] or ["Brazil Serie A", "Portugal Primeira Liga"]


def posbucket(p):
    if not p:
        return "?"
    p = p.upper()
    if p.startswith("G"):
        return "GK"
    if p[0] == "D" or p in ("CB", "LB", "RB", "RWB", "LWB"):
        return "DEF"
    if p[0] == "M" or p in ("DM", "AM", "CM", "LM", "RM"):
        return "MID"
    if p[0] in ("F", "W", "S") or p in ("ST", "CF", "LW", "RW"):
        return "ATT"
    return "?"


def main():
    con = db.connect()
    fmuid_sid = con.execute("SELECT source_id FROM source WHERE name='fm-uid'").fetchone()[0]
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    fm_src = [r[0] for r in con.execute("SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]

    # name_to_uids: xnorm name -> set(uid)  (fm-uid formal names + grade-player names)
    fm_formal = {u: n for u, n in con.execute(
        f"SELECT source_player_id, name FROM source_identity WHERE source_id={fmuid_sid}")}
    fm_dob = {u: d for u, d in con.execute(
        f"SELECT source_player_id, dob FROM source_identity WHERE source_id={fmuid_sid}")}
    uid_pids = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute("SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_pids[uid].add(pid)
    pid_name = {p: n for p, n in con.execute("SELECT player_id, name FROM player")}
    name_to_uids = defaultdict(set)
    for u, n in fm_formal.items():
        x = xnorm(n or "")
        if x:
            name_to_uids[x].add(u)
    for u, pids in uid_pids.items():
        for p in pids:
            x = xnorm(pid_name.get(p, ""))
            if x:
                name_to_uids[x].add(u)

    pid_clubs = defaultdict(set)
    for pid, cid in con.execute("SELECT player_id, club_id FROM player_snapshot WHERE club_id IS NOT NULL"):
        pid_clubs[pid].add(cid)
    fm_clubs = {u: set().union(*(pid_clubs[p] for p in pids)) if pids else set() for u, pids in uid_pids.items()}

    # the ACTUAL bridge build_xwalk used: from confirmed/high player_xwalk links
    espn_pid = {r[0]: r[1] for r in con.execute(
        f"SELECT source_player_id, player_id FROM player_source_id WHERE source_id={espn_sid}")}
    pid_match_clubs = defaultdict(set)
    for pid, cid in con.execute("SELECT DISTINCT player_id, club_id FROM match_player"):
        pid_match_clubs[pid].add(cid)
    eclub_gc = defaultdict(Counter)
    for eid, uid, conf in con.execute(
            "SELECT espn_player_id, fm_uid, confidence FROM player_xwalk WHERE fm_uid IS NOT NULL AND confidence IN ('confirmed','high')"):
        for ec in pid_match_clubs.get(espn_pid.get(eid), ()):
            for gc in fm_clubs.get(uid, ()):
                eclub_gc[ec][gc] += 1
    eclub_to_g = {ec: ({c for c, n in cnt.items() if n >= 2} or {cnt.most_common(1)[0][0]}) for ec, cnt in eclub_gc.items()}

    espn_name = {r[0]: r[1] for r in con.execute(
        f"SELECT source_player_id, name FROM source_identity WHERE source_id={espn_sid}")}
    espn_dob = {r[0]: r[1] for r in con.execute(
        f"SELECT source_player_id, dob FROM source_identity WHERE source_id={espn_sid}")}
    # ESPN player dominant position bucket (from lineups) and FM uid position buckets (from snapshots)
    espn_pos = {}
    for pid, pos in con.execute("SELECT player_id, position FROM match_player WHERE position IS NOT NULL"):
        espn_pos.setdefault(pid, Counter())[posbucket(pos)] += 1
    espn_posb = {p: c.most_common(1)[0][0] for p, c in espn_pos.items()}
    pid_pos = defaultdict(Counter)
    for p, pos in con.execute("SELECT player_id, position FROM player_snapshot WHERE position IS NOT NULL"):
        pid_pos[p][posbucket(pos)] += 1
    uid_posb = {}
    for u, pids in uid_pids.items():
        c = Counter()
        for p in pids:
            c += pid_pos.get(p, Counter())
        if c:
            uid_posb[u] = {b for b, _ in c.most_common(2)}   # allow up to 2 buckets (FM lists multiple)
    # unmatched ESPN players (no fm_uid in xwalk)
    unmatched_eids = set(r[0] for r in con.execute(
        "SELECT espn_player_id FROM player_xwalk WHERE fm_uid IS NULL"))

    for comp in LEAGUES:
        # ESPN starters (internal pids) in this comp -> their eids
        pids = set(p for p, in con.execute(
            "SELECT DISTINCT mp.player_id FROM match_player mp JOIN match m ON m.match_id=mp.match_id "
            "JOIN competition co ON co.competition_id=m.competition_id WHERE co.name=? AND mp.started=1", (comp,)))
        eid_of = defaultdict(list)
        for eid, pid in espn_pid.items():
            eid_of[pid].append(eid)
        cls = Counter(); dob_rec = 0; multi = 0
        for pid in pids:
            for eid in eid_of.get(pid, []):
                if eid not in unmatched_eids:
                    continue
                nn = xnorm(espn_name.get(eid) or pid_name.get(pid) or "")
                cands = name_to_uids.get(nn, set())
                if len(cands) < 2:
                    cls["<2 name candidates (mononym/absent/unique-handled)"] += 1
                    continue
                bridged = set()
                for ec in pid_match_clubs.get(pid, ()):
                    bridged |= eclub_to_g.get(ec, set())
                cmatch = [u for u in cands if fm_clubs.get(u, set()) & bridged]
                if len(cmatch) == 0:
                    cls["cmatch=0 (bridge doesn't reach club -> bridge-thin)"] += 1
                elif len(cmatch) == 1:
                    cls["cmatch=1 (should have resolved!? )"] += 1
                else:
                    multi += 1
                    ed = espn_dob.get(eid)
                    dm = [u for u in cmatch if ed and fm_dob.get(u) and dob_close(ed, fm_dob[u])]
                    if len(dm) == 1:
                        cls["cmatch>1 but DOB picks one (DOB-recoverable)"] += 1; dob_rec += 1
                        continue
                    # try POSITION as the safe tiebreaker
                    epb = espn_posb.get(pid)
                    pm = [u for u in cmatch if epb and epb != "?" and epb in uid_posb.get(u, set())]
                    if len(pm) == 1:
                        cls["cmatch>1, POSITION picks one (position-recoverable)"] += 1
                    else:
                        cls["cmatch>1, neither DOB nor position breaks it (truly ambiguous)"] += 1
        print(f"\n=== {comp} === unmatched ESPN starters by name+squad failure mode:")
        for k, v in cls.most_common():
            print(f"    {v:>5}  {k}")
    con.close()


if __name__ == "__main__":
    main()
