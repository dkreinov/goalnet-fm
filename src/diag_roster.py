"""Step 1: size the roster-constrained matching opportunity (read-only).

Cross-source club ids differ (match_player uses ESPN club_id; player_snapshot uses grade
club_id). Bridge them via the MATCHED starters themselves: the FM grade-club that this
ESPN club's matched starters belong to (that season) IS the FM club for this ESPN club.
Then squad(espn_club, season) = all graded UIDs at that FM grade-club for that season,
and unmatched starters are assigned from the unassigned remainder.
Usage: python D:/Programming/claude/FM/src/diag_roster.py
"""
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import leagues as L

EXCL = {"China Super League", "Ecuador LigaPro", "India Super League", "Paraguay Primera Division",
        "Peru Liga 1", "South Africa Premiership", "Israel Ligat haAl", "Japan J1 League", "Colombia Primera A"}


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
    season_of = {mid: lab for mid, lab in con.execute(
        "SELECT m.match_id, s.label FROM match m JOIN season s ON s.season_id=m.season_id")}
    comp_of = {mid: c for mid, c in con.execute("SELECT match_id, competition_id FROM match")}
    comp_name = {cid: nm for cid, nm in con.execute("SELECT competition_id, name FROM competition")}
    fm_src = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    src_in = ",".join("?" * len(fm_src))

    # fm_version_id -> season label
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

    # grade-club membership per season: members[(grade_club, season)] = {uid: posbucket}
    members = defaultdict(dict)
    uid_clubs = defaultdict(set)   # (uid, season) -> {grade_club}
    for pid, cid, fid, pos in con.execute(
            f"SELECT player_id, club_id, fm_version_id, position FROM player_snapshot "
            f"WHERE club_id IS NOT NULL AND source_id IN ({src_in})", fm_src):
        lab = ver_season.get(fid); uid = pid_uid.get(pid)
        if lab and uid:
            members[(cid, lab)].setdefault(uid, posbucket(pos))
            uid_clubs[(uid, lab)].add(cid)

    resolved = {}   # espn player_id -> (uid, has_grade)
    for epid, uid, fpid in con.execute(
            "SELECT espn_player_pid, fm_uid, fm_player_id FROM player_xwalk WHERE fm_uid IS NOT NULL"):
        resolved[epid] = (uid, fpid)

    starters = defaultdict(list)
    for mid, pid, cid, pos in con.execute(
            "SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        starters[(mid, cid)].append((pid, pos))

    cls = Counter()
    forced = recoverable = 0
    for (mid, ecid), pls in starters.items():
        if comp_name.get(comp_of.get(mid)) in EXCL:
            continue
        lab = season_of.get(mid)
        if not lab:
            continue
        assigned = set()
        matched_uids = []
        unmatched = []
        for pid, pos in pls:
            r = resolved.get(pid)
            if r and r[1]:
                assigned.add(r[0]); matched_uids.append(r[0])
            elif r:
                matched_uids.append(r[0]); unmatched.append((pid, posbucket(pos)))  # linked no grade
            else:
                unmatched.append((pid, posbucket(pos)))
        if not unmatched:
            continue
        # bridge: dominant FM grade-club for this espn club-season
        ctr = Counter()
        for u in matched_uids:
            for gc in uid_clubs.get((u, lab), ()):
                ctr[gc] += 1
        if not ctr:
            cls["no bridge (no matched starter has a grade-club)"] += 1
            continue
        gclub = ctr.most_common(1)[0][0]
        sq = members.get((gclub, lab), {})
        free = {u: pb for u, pb in sq.items() if u not in assigned}
        if not free:
            cls["unresolvable (squad fully assigned / absent)"] += 1
            continue
        free_pos = Counter(pb for pb in free.values())
        if len(unmatched) == 1:
            pb = unmatched[0][1]
            if free_pos.get(pb, 0) >= 1:
                cls["forced (1 missing, position-match free)"] += 1; forced += 1; recoverable += 1
            elif free:
                cls["forced-ish (1 missing, free but pos mismatch)"] += 1; recoverable += 1
            else:
                cls["unresolvable"] += 1
        else:
            need = Counter(pb for _, pb in unmatched)
            if all(free_pos.get(pb, 0) >= need[pb] for pb in need):
                cls["resolvable (multi, position-covered)"] += 1; recoverable += 1
            elif len(free) >= len(unmatched):
                cls["resolvable-ish (multi, enough free)"] += 1; recoverable += 1
            else:
                cls["partial (multi, not enough)"] += 1

    print("target match-SIDES with >=1 unmatched starter:")
    for k in sorted(cls, key=lambda k: -cls[k]):
        print(f"  {k:48} {cls[k]:>7,}")
    print(f"\n  RECOVERABLE match-sides (forced + resolvable): {recoverable:,}")
    print(f"  forced single-assignment (highest confidence):  {forced:,}")


if __name__ == "__main__":
    main()
