"""Step 2: roster-constrained per-match assignment of unmatched starters.

For each match-side, bridge the ESPN club to its FM grade-club via the matched starters,
then assign each unmatched starter to an unassigned squad member of the SAME position
(name similarity + appearance prior only break ties). Writes match-level links to
`match_grade_link` — does NOT mutate player_xwalk (global identity stays guarded).

  python roster_match.py --selftest   # held-out accuracy (hide a known starter, recover it)
  python roster_match.py              # build match_grade_link
"""
import sys
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import leagues as L
from build_xwalk import xnorm
from diag_roster import posbucket, EXCL


def load(con):
    season_of = {mid: lab for mid, lab in con.execute(
        "SELECT m.match_id, s.label FROM match m JOIN season s ON s.season_id=m.season_id")}
    comp_of = {mid: c for mid, c in con.execute("SELECT match_id, competition_id FROM match")}
    comp_name = {cid: nm for cid, nm in con.execute("SELECT competition_id, name FROM competition")}
    fm_src = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    src_in = ",".join("?" * len(fm_src))
    ver_season = {}
    for lab, (dbid, game, dbver, date) in L.SEASON_DB.items():
        for fid, in con.execute("SELECT fm_version_id FROM fm_version WHERE game=? AND db_version=?", (game, dbver)):
            ver_season[fid] = lab
    for game, lab in {"FM20": "2019-20", "FM21": "2020-21", "FM22": "2021-22", "FM23": "2022-23"}.items():
        for fid, in con.execute("SELECT fm_version_id FROM fm_version WHERE game=?", (game,)):
            ver_season.setdefault(fid, lab)
    pid_uid = {}
    for sid in fm_src:
        for uid, pid in con.execute(
                "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            pid_uid[pid] = uid
    uid_name = {u: n for u, n in con.execute(
        "SELECT source_player_id, name FROM source_identity WHERE source_id="
        "(SELECT source_id FROM source WHERE name='fm-uid')")}
    members = defaultdict(dict)        # (grade_club, season) -> {uid: posbucket}
    uid_clubs = defaultdict(set)       # (uid, season) -> {grade_club}
    for pid, cid, fid, pos in con.execute(
            f"SELECT player_id, club_id, fm_version_id, position FROM player_snapshot "
            f"WHERE club_id IS NOT NULL AND source_id IN ({src_in})", fm_src):
        lab = ver_season.get(fid); uid = pid_uid.get(pid)
        if lab and uid:
            members[(cid, lab)].setdefault(uid, posbucket(pos))
            uid_clubs[(uid, lab)].add(cid)
    resolved = {}
    for epid, uid, fpid in con.execute(
            "SELECT espn_player_pid, fm_uid, fm_player_id FROM player_xwalk WHERE fm_uid IS NOT NULL"):
        resolved[epid] = (uid, fpid)
    pname = {pid: nm for pid, nm in con.execute("SELECT player_id, name FROM player")}
    starters = defaultdict(list)
    for mid, pid, cid, pos in con.execute(
            "SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        starters[(mid, cid)].append((pid, pos))
    # appearance prior: matched-starter count per (uid, season)
    appear = Counter()
    for (mid, cid), pls in starters.items():
        lab = season_of.get(mid)
        for pid, pos in pls:
            r = resolved.get(pid)
            if r and lab:
                appear[(r[0], lab)] += 1
    return dict(season_of=season_of, comp_of=comp_of, comp_name=comp_name, members=members,
                uid_clubs=uid_clubs, resolved=resolved, pname=pname, uid_name=uid_name,
                starters=starters, appear=appear)


def assign_side(D, mid, ecid, lab, hide_pid=None):
    """Return list of (espn_pid, uid, confidence) assignments for unmatched starters.
    hide_pid: pretend this resolved starter is unmatched (for self-test)."""
    assigned = set(); matched_uids = []; unmatched = []
    for pid, pos in D["starters"][(mid, ecid)]:
        r = D["resolved"].get(pid)
        if pid == hide_pid:                      # self-test: pretend this graded starter is unmatched
            matched_uids.append(r[0]); unmatched.append((pid, posbucket(pos)))
        elif r and r[1]:                          # matched WITH grade
            assigned.add(r[0]); matched_uids.append(r[0])
        elif r:                                   # linked but no grade: bridge only, NOT reassigned (Step 4 fills it)
            matched_uids.append(r[0])
        else:                                     # truly unmatched (name floor): roster-assignable
            unmatched.append((pid, posbucket(pos)))
    if not unmatched:
        return []
    # bridge using ONLY the visible matched-with-grade teammates; the grade-club id is
    # fragmented across sources/editions, so take EVERY grade-club shared by >=2 teammates
    # (drops loan/transfer singletons) and union their members into one squad.
    ctr = Counter()
    for u in assigned:
        for gc in D["uid_clubs"].get((u, lab), ()):
            ctr[gc] += 1
    if not ctr:
        return []
    gclubs = [gc for gc, c in ctr.items() if c >= 2] or [ctr.most_common(1)[0][0]]
    squad = {}
    for gc in gclubs:
        squad.update(D["members"].get((gc, lab), {}))
    free = {u: pb for u, pb in squad.items() if u not in assigned}
    out = []
    for pid, pb in unmatched:
        if not free:
            break
        en = xnorm(D["pname"].get(pid, ""))
        maxap = max((D["appear"].get((u, lab), 0) for u in free), default=0) or 1
        def score(u, pbu):
            ns = SequenceMatcher(None, en, xnorm(D["uid_name"].get(u, ""))).ratio()
            pos = 1.0 if pbu == pb and pb != "?" else 0.0
            ap = D["appear"].get((u, lab), 0) / maxap
            return 0.6 * ns + 0.25 * pos + 0.15 * ap, ns
        best_u = max(free, key=lambda u: score(u, free[u]))
        sc, ns = score(best_u, free[best_u])
        only_pos = sum(1 for p in free.values() if p == pb) == 1
        conf = "high" if (ns > 0.55 or (only_pos and pb != "?")) else "medium"
        out.append((pid, best_u, conf))
        del free[best_u]
    return out


def _squad_of(D, mid, ecid, lab, exclude=None):
    """Squad uids for a side (>=2-teammate grade-club bridge), as the matcher sees it."""
    assigned = set()
    for pid, pos in D["starters"][(mid, ecid)]:
        if pid == exclude:
            continue
        r = D["resolved"].get(pid)
        if r and r[1]:
            assigned.add(r[0])
    ctr = Counter()
    for u in assigned:
        for gc in D["uid_clubs"].get((u, lab), ()):
            ctr[gc] += 1
    if not ctr:
        return set(), assigned
    gclubs = [gc for gc, c in ctr.items() if c >= 2] or [ctr.most_common(1)[0][0]]
    squad = set()
    for gc in gclubs:
        squad |= set(D["members"].get((gc, lab), {}))
    return squad, assigned


def selftest(con, D, n=4000):
    """Honest test: hide a starter who genuinely HAS a season-specific grade (is in this
    season's squad), then check the roster matcher re-assigns them correctly."""
    ok = miss = 0
    cnt = 0
    for (mid, cid) in D["starters"]:
        if D["comp_name"].get(D["comp_of"].get(mid)) in EXCL:
            continue
        lab = D["season_of"].get(mid)
        if not lab:
            continue
        squad, assigned = _squad_of(D, mid, cid, lab)
        # starters whose uid is genuinely in this season's squad (recoverable in principle)
        in_sq = [pid for pid, pos in D["starters"][(mid, cid)]
                 if (D["resolved"].get(pid) and D["resolved"][pid][1]
                     and D["resolved"][pid][0] in squad)]
        if len(in_sq) < 7:      # need enough teammates to still bridge after hiding one
            continue
        cnt += 1
        if cnt % 5:             # deterministic subsample
            continue
        pid_hide = in_sq[mid % len(in_sq)]
        true_uid = D["resolved"][pid_hide][0]
        res = assign_side(D, mid, cid, lab, hide_pid=pid_hide)
        got = dict((p, u) for p, u, _ in res).get(pid_hide)
        ok += (got == true_uid); miss += (got != true_uid)
        if ok + miss >= n:
            break
    tot = ok + miss
    print(f"HELD-OUT recovery (season-graded starters): {ok}/{tot} = {100*ok/tot:.1f}%  (bar >=85%)")
    return ok / tot if tot else 0


def main():
    con = db.connect()
    D = load(con)
    if "--selftest" in sys.argv:
        selftest(con, D)
        return
    con.execute("DROP TABLE IF EXISTS match_grade_link")
    con.execute("""CREATE TABLE match_grade_link(
        match_id INTEGER, player_id INTEGER, fm_uid TEXT, method TEXT, confidence TEXT,
        PRIMARY KEY(match_id, player_id))""")
    rows = []
    nconf = Counter()
    for (mid, ecid), pls in D["starters"].items():
        if D["comp_name"].get(D["comp_of"].get(mid)) in EXCL:
            continue
        lab = D["season_of"].get(mid)
        if not lab:
            continue
        for pid, uid, conf in assign_side(D, mid, ecid, lab):
            rows.append((mid, pid, uid, "roster", conf)); nconf[conf] += 1
    con.execute("BEGIN")
    con.executemany("INSERT OR REPLACE INTO match_grade_link VALUES (?,?,?,?,?)", rows)
    con.execute("COMMIT")
    print(f"match_grade_link: {len(rows):,} roster-assigned starters  ({dict(nconf)})")

    # LEARN name synonyms: persist HIGH-confidence (espn name -> FM uid/name) discoveries where
    # the names actually differ, so build_xwalk can seed them as confirmed links next run.
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    pid_eid = {pid: eid for eid, pid in con.execute(
        "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (espn_sid,))}
    con.execute("""CREATE TABLE IF NOT EXISTS learned_alias(
        espn_player_id TEXT PRIMARY KEY, espn_name TEXT, fm_uid TEXT, fm_name TEXT,
        votes INTEGER DEFAULT 1)""")
    alias = {}   # espn_id -> (espn_name, fm_uid, fm_name); vote across matches, keep majority
    votes = Counter()
    for mid, pid, uid, method, conf in rows:
        if conf != "high":
            continue
        eid = pid_eid.get(pid)
        en = D["pname"].get(pid, ""); fn = D["uid_name"].get(uid, "")
        if not eid or not en or xnorm(en) == xnorm(fn):   # skip if NORMALIZED names already identical
            continue
        votes[(eid, uid)] += 1
        alias[eid] = (D["pname"].get(pid, ""), uid, D["uid_name"].get(uid, ""))
    # keep, per espn id, the uid with the most match-level votes (stability)
    best = {}
    for (eid, uid), v in votes.items():
        if eid not in best or v > best[eid][1]:
            best[eid] = (uid, v)
    arows = [(eid, alias[eid][0], uid, alias[eid][2], v) for eid, (uid, v) in best.items()]
    con.execute("BEGIN")
    con.executemany("INSERT OR REPLACE INTO learned_alias VALUES (?,?,?,?,?)", arows)
    con.execute("COMMIT")
    print(f"learned_alias: {len(arows):,} ESPN->FM name synonyms recorded (high-conf, names differ)")


if __name__ == "__main__":
    main()
