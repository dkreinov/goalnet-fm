"""Diagnose why Portugal Primeira Liga matches rarely clear the 11+11 graded-starter bar
despite grades being scraped. Read-only. Buckets every distinct starter by crosswalk status,
and for UNMATCHED starters checks token overlap against the FM name index to distinguish
'fixable name-format mismatch' (an FM record exists under a different name) from 'genuinely absent'.
Usage: python D:/Programming/claude/FM/src/diag_portugal.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
from build_xwalk import xnorm

PORTUGAL = "Portugal Primeira Liga"


def main():
    con = db.connect()
    fmuid_sid = con.execute("SELECT source_id FROM source WHERE name='fm-uid'").fetchone()[0]
    fm_src = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]

    # FM name/token index (formal names + grade-linked common names), and which uids have grades
    pid_name = {pid: nm for pid, nm in con.execute("SELECT player_id, name FROM player")}
    uid_grade_pids = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute(
                "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_grade_pids[uid].add(pid)
    tok_to_uids = defaultdict(set)
    fm_formal = {}
    for uid, nm in con.execute(
            f"SELECT source_player_id, name FROM source_identity WHERE source_id={fmuid_sid}"):
        fm_formal[uid] = nm
        for t in xnorm(nm).split():
            if len(t) >= 4:
                tok_to_uids[t].add(uid)
    for uid, pids in uid_grade_pids.items():
        for p in pids:
            for t in xnorm(pid_name.get(p, "")).split():
                if len(t) >= 4:
                    tok_to_uids[t].add(uid)

    cid = con.execute("SELECT competition_id FROM competition WHERE name=?", (PORTUGAL,)).fetchone()[0]
    # distinct STARTERS in Portugal matches -> their crosswalk row
    rows = con.execute(
        """SELECT DISTINCT mp.player_id, p.name
           FROM match_player mp
           JOIN match m ON m.match_id=mp.match_id
           JOIN player p ON p.player_id=mp.player_id
           WHERE m.competition_id=? AND mp.started=1""", (cid,)).fetchall()
    xw = {}
    for pid, uid, fpid, conf, meth in con.execute(
            "SELECT espn_player_pid, fm_uid, fm_player_id, confidence, method FROM player_xwalk"):
        xw[pid] = (uid, fpid, conf, meth)

    buckets = defaultdict(int)
    examples = defaultdict(list)
    fixable_absent = [0, 0]   # [fixable name-format mismatch, genuinely absent] among unmatched
    for pid, name in rows:
        row = xw.get(pid)
        if row is None:
            buckets["not in crosswalk"] += 1
            if len(examples["not in crosswalk"]) < 4:
                examples["not in crosswalk"].append(name)
            continue
        uid, fpid, conf, meth = row
        if uid and fpid:
            buckets["linked + has grade (xwalk-OK)"] += 1
        elif uid and not fpid:
            buckets["linked but FM player has NO grade in DB"] += 1
            if len(examples["linked but FM player has NO grade in DB"]) < 4:
                examples["linked but FM player has NO grade in DB"].append(f"{name} -> {fm_formal.get(uid, uid)}")
        elif conf == "ambiguous":
            buckets[f"ambiguous ({meth})"] += 1
            if len(examples["ambiguous"]) < 4:
                examples["ambiguous"].append(name)
        else:
            buckets[f"unmatched ({meth})"] += 1
            # token-overlap probe: does an FM record exist under a different name format?
            toks = [t for t in xnorm(name).split() if len(t) >= 4]
            cand = set()
            for t in toks:
                cand |= tok_to_uids.get(t, set())
            if cand:
                fixable_absent[0] += 1
                if len(examples["unmatched-but-FM-has-a-token-match (FIXABLE)"]) < 6:
                    one = next(iter(cand))
                    examples["unmatched-but-FM-has-a-token-match (FIXABLE)"].append(
                        f"{name}  ~  {fm_formal.get(one, one)}")
            else:
                fixable_absent[1] += 1
                if len(examples["unmatched-genuinely-absent"]) < 4:
                    examples["unmatched-genuinely-absent"].append(name)

    total = len(rows)
    ok = buckets["linked + has grade (xwalk-OK)"]
    fail = total - ok
    print(f"Portugal Primeira Liga — distinct STARTERS: {total}")
    print(f"  linked + has grade (xwalk-OK): {ok}  ({100*ok/total:.0f}%)")
    print(f"  failing (not fully usable):    {fail}  ({100*fail/total:.0f}%)\n")
    print("FAILURE CAUSE HISTOGRAM (sums to failing):")
    s = 0
    for b in sorted(buckets, key=lambda b: -buckets[b]):
        if b == "linked + has grade (xwalk-OK)":
            continue
        print(f"  {b:48} {buckets[b]:>5}")
        s += buckets[b]
    print(f"  {'(sum)':48} {s:>5}")
    print(f"\n  of the unmatched: {fixable_absent[0]} have an FM token-match (FIXABLE name format) "
          f"vs {fixable_absent[1]} genuinely absent")
    print("\nEXAMPLES:")
    for k, ex in examples.items():
        print(f"  [{k}]")
        for e in ex:
            print(f"      {e}")
    # dominant-cause hypothesis
    top = max((b for b in buckets if b != "linked + has grade (xwalk-OK)"), key=lambda b: buckets[b], default=None)
    print(f"\nDOMINANT CAUSE: {top}")


if __name__ == "__main__":
    main()
