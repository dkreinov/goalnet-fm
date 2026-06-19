"""Step 1 (read-only): verify + size the grade-player NAME-merge bug.

(a) internal grade-players whose grade snapshots span >=2 distinct clubs in the SAME season
    (the over-merge signature: different real people pooled on one player_id).
(b) internal grade-players carrying >=2 distinct FM-UIDs from grade sources (fminside/kaggle/futek)
    -> these UIDs are distinct real players wrongly collapsed by norm_name in db.player_id().
(c) recoverable estimate: of unmatched/ambiguous TARGET starters, how many have a same-name grade
    snapshot at the club that the side's matched teammates bridge to (the dominant grade-club that
    season). After un-merge each such UID sits at exactly that club, so build_xwalk's 'name+squad'
    club-season squad disambiguation would fire. Mirrors diag_roster's teammate-bridge.

Usage: python D:/Programming/claude/FM/src/diag_merge.py
"""
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import leagues as L

# same target exclusion set as diag_roster (leagues with no usable lineups / quality-excluded)
EXCL = {"China Super League", "Ecuador LigaPro", "India Super League", "Paraguay Primera Division",
        "Peru Liga 1", "South Africa Premiership", "Israel Ligat haAl", "Japan J1 League", "Colombia Primera A"}


def main():
    con = db.connect()
    fm_src = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    src_in = ",".join("?" * len(fm_src))

    # fm_version_id -> season label (same construction as diag_roster)
    ver_season = {}
    for lab, (dbid, game, dbver, date) in L.SEASON_DB.items():
        for fid, in con.execute("SELECT fm_version_id FROM fm_version WHERE game=? AND db_version=?", (game, dbver)):
            ver_season[fid] = lab
    for game, lab in {"FM20": "2019-20", "FM21": "2020-21", "FM22": "2021-22", "FM23": "2022-23"}.items():
        for fid, in con.execute("SELECT fm_version_id FROM fm_version WHERE game=?", (game,)):
            ver_season.setdefault(fid, lab)

    # ---- (a) grade-players whose snapshots span >=2 clubs in the SAME season ----
    pid_season_clubs = defaultdict(lambda: defaultdict(set))
    for pid, cid, fid in con.execute(
            f"SELECT player_id, club_id, fm_version_id FROM player_snapshot "
            f"WHERE club_id IS NOT NULL AND source_id IN ({src_in})", fm_src):
        lab = ver_season.get(fid)
        if lab:
            pid_season_clubs[pid][lab].add(cid)
    span_same_season = sum(1 for pid, d in pid_season_clubs.items()
                           if any(len(cs) >= 2 for cs in d.values()))
    # also span >=2 clubs across any seasons (broader over-merge signal)
    span_any = sum(1 for pid, d in pid_season_clubs.items()
                   if len(set().union(*d.values())) >= 2)

    # ---- (b) grade-players carrying >=2 distinct FM-UIDs from grade sources ----
    pid_uids = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute(
                "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            pid_uids[pid].add(uid)
    multi_uid_pids = {pid: u for pid, u in pid_uids.items() if len(u) >= 2}
    extra_uids = sum(len(u) - 1 for u in multi_uid_pids.values())  # distinct players to un-merge into

    print("=== (a) over-merge by club ===")
    print(f"  grade-players with snapshots at >=2 clubs in the SAME season : {span_same_season:,}")
    print(f"  grade-players with snapshots at >=2 clubs across any seasons  : {span_any:,}")
    print("=== (b) over-merge by FM-UID (grade sources) ===")
    print(f"  grade-players carrying >=2 distinct FM-UIDs                   : {len(multi_uid_pids):,}")
    print(f"  extra distinct real players hidden inside them (sum uids-1)   : {extra_uids:,}")

    # ---- (c) recoverable estimate ----
    # name+club+season index of grade snapshots (norm_name -> {(club,season)})
    pid_name = {pid: name for pid, name in con.execute("SELECT player_id, name FROM player")}
    name_club_season = defaultdict(set)
    for pid, cid, fid in con.execute(
            f"SELECT player_id, club_id, fm_version_id FROM player_snapshot "
            f"WHERE club_id IS NOT NULL AND source_id IN ({src_in})", fm_src):
        lab = ver_season.get(fid)
        nm = db.norm(pid_name.get(pid, ""))
        if lab and nm:
            name_club_season[nm].add((cid, lab))

    season_of = {mid: lab for mid, lab in con.execute(
        "SELECT m.match_id, s.label FROM match m JOIN season s ON s.season_id=m.season_id")}
    comp_of = {mid: c for mid, c in con.execute("SELECT match_id, competition_id FROM match")}
    comp_name = {cid: nm for cid, nm in con.execute("SELECT competition_id, name FROM competition")}

    # xwalk: espn player_id (the internal pid) -> fm_uid (NULL if ambiguous/unmatched)
    resolved_uid = {}      # espn_player_pid -> fm_uid (only when linked)
    for epid, uid in con.execute(
            "SELECT espn_player_pid, fm_uid FROM player_xwalk WHERE fm_uid IS NOT NULL"):
        resolved_uid[epid] = uid

    # grade-club membership per UID-season, for bridging (uid -> clubs that season).
    # UIDs share merged pids, so use pid->clubs as a proxy of the matched-teammate's grade-club.
    pid_season_clubset = pid_season_clubs  # reuse

    starters = defaultdict(list)
    for mid, pid, cid, pos in con.execute(
            "SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        starters[(mid, cid)].append((pid, pos))

    # map espn internal pid -> its merged grade pid via xwalk fm_player_id (representative)
    epid_gpid = {}
    for epid, fpid in con.execute(
            "SELECT espn_player_pid, fm_player_id FROM player_xwalk WHERE fm_player_id IS NOT NULL"):
        epid_gpid[epid] = fpid

    recoverable = 0
    examined = 0
    no_bridge = 0
    for (mid, ecid), pls in starters.items():
        if comp_name.get(comp_of.get(mid)) in EXCL:
            continue
        lab = season_of.get(mid)
        if not lab:
            continue
        # bridge: dominant grade-club among matched (linked) teammates that season
        ctr = Counter()
        unmatched = []
        for pid, pos in pls:
            if pid in resolved_uid and pid in epid_gpid:
                for gc in pid_season_clubset.get(epid_gpid[pid], {}).get(lab, ()):
                    ctr[gc] += 1
            else:
                unmatched.append(pid)
        if not unmatched:
            continue
        if not ctr:
            no_bridge += len(unmatched)
            continue
        gclub = ctr.most_common(1)[0][0]
        for pid in unmatched:
            examined += 1
            nm = db.norm(pid_name.get(pid, ""))
            if not nm:
                continue
            if (gclub, lab) in name_club_season.get(nm, set()):
                recoverable += 1

    print("=== (c) recoverable target starters (un-merge + name+squad would fire) ===")
    print(f"  unmatched/ambiguous target starters examined (had a bridge)   : {examined:,}")
    print(f"  unmatched starters with NO teammate bridge (skipped)          : {no_bridge:,}")
    print(f"  RECOVERABLE (same-name grade snapshot at bridged club-season) : {recoverable:,}")
    con.close()


if __name__ == "__main__":
    main()
