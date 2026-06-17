"""Step 1 audit: size the simple-but-broken issues this session may have left.
(a) ambiguous crosswalk players resolvable by club-season squad uniqueness (the Bruno pattern)
(b) blocking-wrong-DOBs: a same-name FM candidate exists but the stored ESPN DOB matches none
(c) dataset coverage % of the recently-added features (squad value / reputation / age)
Read-only.
"""
import sys
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import db
from build_xwalk import xnorm, dob_close


def main():
    con = db.connect()
    espn = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    fmuid = con.execute("SELECT source_id FROM source WHERE name='fm-uid'").fetchone()[0]

    # FM name -> [(uid, dob)] and uid -> has grade
    fm_by_name = defaultdict(list)
    fm_dob = {}
    for u, n, d in con.execute(
            f"SELECT source_player_id, name, dob FROM source_identity WHERE source_id={fmuid}"):
        fm_by_name[xnorm(n)].append(u); fm_dob[u] = d
    fm_src = [r[0] for r in con.execute("SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    graded_uid = set()
    for sid in fm_src:
        for u, in con.execute("SELECT DISTINCT source_player_id FROM player_source_id WHERE source_id=?", (sid,)):
            graded_uid.add(u)

    espn_dob = {e: d for e, d in con.execute(
        f"SELECT source_player_id, dob FROM source_identity WHERE source_id={espn} AND dob IS NOT NULL")}
    pid_eid = {p: e for e, p in con.execute(f"SELECT source_player_id, player_id FROM player_source_id WHERE source_id={espn}")}
    pname = {p: n for p, n in con.execute("SELECT player_id, norm_name FROM player")}

    # ambiguous crosswalk rows
    amb = [(e, pid) for e, pid, c in con.execute(
        "SELECT espn_player_id, espn_player_pid, confidence FROM player_xwalk") if c == "ambiguous"]

    # (b) blocking-wrong-DOBs: same-name FM candidate exists, espn dob matches none of them
    block = 0
    eid_pid = {e: p for p, e in pid_eid.items()}
    for e, d in espn_dob.items():
        pid = eid_pid.get(e)
        if pid is None:
            continue
        cands = fm_by_name.get(pname.get(pid, ""), [])
        if cands and not any(fm_dob.get(u) and dob_close(d, fm_dob[u]) for u in cands):
            block += 1

    # (a) ambiguous resolvable by club-season squad uniqueness: among the name candidates, exactly one is graded
    # (a cheap proxy for 'unique at club' — refined in Step 3 with the real club bridge)
    resolvable = 0
    for e, pid in amb:
        cands = fm_by_name.get(pname.get(pid, ""), [])
        graded = [u for u in cands if u in graded_uid]
        if len(graded) == 1:
            resolvable += 1

    print("=" * 60)
    print(f"(a) ambiguous crosswalk players: {len(amb):,}")
    print(f"    of which exactly ONE same-name FM candidate is graded (proxy-resolvable): {resolvable:,}")
    print(f"(b) blocking-wrong-DOBs (same-name FM exists, DOB matches none): {block:,}")

    # (c) feature coverage in the dataset
    df = pd.read_parquet(db.ROOT / "data" / "dataset.parquet")
    graded_match = (df.home_n_matched >= 8) & (df.away_n_matched >= 8)
    print("(c) recently-added feature coverage (of graded matches, both sides >=8):")
    for col in ("home_squad_value_total", "home_squad_ca_mean", "home_club_reputation", "home_age_mean"):
        if col in df.columns:
            cov = df.loc[graded_match, col].notna().mean()
            print(f"      {col:26} {100*cov:5.0f}% non-null")
        else:
            print(f"      {col:26} MISSING")

    # Bruno spot-check
    print("\nBruno Fernandes spot-check:")
    bru = con.execute(f"SELECT source_player_id,dob FROM source_identity WHERE source_id={fmuid} AND lower(name)='bruno fernandes'").fetchall()
    print(f"  FM 'Bruno Fernandes' count: {len(bru)}  dobs: {[d for _,d in bru]}")
    realgr = con.execute("""SELECT COUNT(*) FROM player_snapshot ps JOIN player_source_id psi ON psi.player_id=ps.player_id
        WHERE psi.source_player_id='43124203'""").fetchone()[0]
    print(f"  real Bruno uid 43124203 grade snapshots: {realgr}")
    x = con.execute("SELECT x.fm_uid,x.confidence FROM player_xwalk x JOIN player p ON p.player_id=x.espn_player_pid WHERE p.norm_name='bruno fernandes' ORDER BY (SELECT COUNT(*) FROM match_player mp WHERE mp.player_id=p.player_id) DESC LIMIT 1").fetchone()
    print(f"  ESPN Bruno xwalk: fm_uid={x[0]} confidence={x[1]}")


if __name__ == "__main__":
    main()
