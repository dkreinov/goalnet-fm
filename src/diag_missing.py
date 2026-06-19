"""Why are 'missing' starters missing? For every starter NOT resolved to an FM grade (after the
Stage-A ESPN-side fix), ask: does a same-name FM grade player exist — this season, an adjacent
season, or never? Splits the gap into IN-FM-but-unlinked (recoverable by cross-season continuity)
vs NOT-IN-FM (genuine coverage gap needing a scrape).

Usage: python D:/Programming/claude/FM/src/diag_missing.py
"""
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import leagues as L
from build_xwalk import xnorm

SEASON_ORDER = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SIDX = {s: i for i, s in enumerate(SEASON_ORDER)}


def main():
    con = db.connect()
    fm_src = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    src_in = ",".join("?" * len(fm_src))

    # fm_version -> season
    ver_season = {}
    for lab, (dbid, game, dbver, date) in L.SEASON_DB.items():
        for fid, in con.execute("SELECT fm_version_id FROM fm_version WHERE game=? AND db_version=?", (game, dbver)):
            ver_season[fid] = lab
    for game, lab in {"FM20": "2019-20", "FM21": "2020-21", "FM22": "2021-22", "FM23": "2022-23"}.items():
        for fid, in con.execute("SELECT fm_version_id FROM fm_version WHERE game=?", (game,)):
            ver_season.setdefault(fid, lab)

    # FM grade players: xnorm(name) -> {season: set(grade_club_id)}  (only graded snapshots)
    pname = {pid: nm for pid, nm in con.execute("SELECT player_id, name FROM player")}
    fmname_season_clubs = defaultdict(lambda: defaultdict(set))
    for pid, cid, fid in con.execute(
            f"SELECT player_id, club_id, fm_version_id FROM player_snapshot WHERE source_id IN ({src_in})", fm_src):
        lab = ver_season.get(fid)
        nm = xnorm(pname.get(pid, ""))
        if lab and nm:
            fmname_season_clubs[nm][lab].add(cid)
    fm_names = set(fmname_season_clubs)

    # which starters are COVERED (mirror build_dataset Stage-A): xwalk single-uid w/ grade, OR
    # ESPN-collision resolvable, OR roster-assigned. Anything else = missing.
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    pid_eids = defaultdict(list)
    for eid, pid in con.execute("SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (espn_sid,)):
        pid_eids[pid].append(eid)
    eid_uid = {r[0]: r[1] for r in con.execute(
        "SELECT espn_player_id, fm_uid FROM player_xwalk WHERE fm_uid IS NOT NULL")}
    uid_has_grade = set()
    uid_pids = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute("SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_pids[uid].add(pid)
    gp = set(r[0] for r in con.execute("SELECT DISTINCT player_id FROM player_snapshot"))
    for uid, pids in uid_pids.items():
        if pids & gp:
            uid_has_grade.add(uid)

    def covered_pid(pid):
        uids = {eid_uid.get(e) for e in pid_eids.get(pid, [])}
        uids = {u for u in uids if u and u in uid_has_grade}
        return len(uids) >= 1   # single or collision: at least one ESPN id maps to a graded uid

    roster = set((m, p) for m, p in con.execute(
        "SELECT match_id, player_id FROM match_grade_link WHERE fm_uid IS NOT NULL"))

    season_of = {mid: lab for mid, lab in con.execute(
        "SELECT m.match_id, s.label FROM match m JOIN season s ON s.season_id=m.season_id")}

    # walk starters, collect MISSING ones with their (xnorm name, season)
    miss_app = Counter()                 # internal pid -> appearances missing
    miss_name_season = defaultdict(set)  # pid -> set(season appeared)
    for mid, pid in con.execute("SELECT match_id, player_id FROM match_player WHERE started=1"):
        if covered_pid(pid) or (mid, pid) in roster:
            continue
        miss_app[pid] += 1
        if season_of.get(mid):
            miss_name_season[pid].add(season_of[mid])

    # classify each missing distinct player by same-name FM existence vs the seasons he played
    buckets = Counter(); buckets_app = Counter()
    examples = defaultdict(list)
    for pid, napp in miss_app.items():
        nm = xnorm(pname.get(pid, ""))
        played = miss_name_season.get(pid, set())
        if nm not in fm_names:
            b = "NOT_IN_FM (no same-name grade anywhere -> needs scrape / not in FM)"
        else:
            fmseasons = set(fmname_season_clubs[nm])
            if played & fmseasons:
                b = "SAME_SEASON same-name FM exists (identity/ambiguity fail -> matchable)"
            else:
                # only adjacent seasons -> transfer / data-gap
                adj = any(abs(SIDX.get(p, -9) - SIDX.get(f, 99)) == 1 for p in played for f in fmseasons)
                b = ("ADJACENT_SEASON only (transfer/data-gap -> cross-season recover)"
                     if adj else "OTHER_SEASON only (non-adjacent same-name)")
        buckets[b] += 1; buckets_app[b] += napp
        if len(examples[b]) < 8:
            examples[b].append(f"{pname.get(pid)} (played {sorted(played)}, FM {sorted(fmname_season_clubs[nm]) if nm in fm_names else '-'}) x{napp}")

    print(f"MISSING distinct starters: {len(miss_app):,}  ({sum(miss_app.values()):,} appearances)")
    for b in sorted(buckets, key=lambda k: -buckets_app[k]):
        print(f"\n[{buckets[b]:,} players / {buckets_app[b]:,} apps]  {b}")
        for ex in examples[b]:
            print(f"    {ex}")
    con.close()


if __name__ == "__main__":
    main()
