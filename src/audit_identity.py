"""Identity-crosswalk audit: per-league confidence, false merges the DOB anchor corrected,
flagged-for-review queue, and spot-checks. Read-only.
Usage: python D:/Programming/claude/FM/src/audit_identity.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import db


def main():
    con = db.connect()

    # 1. per-league confidence (from the built dataset)
    df = pd.read_parquet(db.ROOT / "data" / "dataset.parquet")
    print("=" * 78)
    print("PER-LEAGUE GRADE CONFIDENCE (matched starters)")
    print("=" * 78)
    print(f"{'LEAGUE':32}{'matched':>9}{'confirmed':>10}{'high':>8}{'fallback':>9}{'hi-conf%':>9}")
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for _, r in df.iterrows():
        a = agg[r["competition"]]
        for side in ("home", "away"):
            a[0] += int(r.get(f"{side}_n_matched", 0) or 0)
            a[1] += int(r.get(f"{side}_n_confirmed", 0) or 0)
            a[2] += int(r.get(f"{side}_n_high", 0) or 0)
            a[3] += int(r.get(f"{side}_n_fallback", 0) or 0)
    for comp in sorted(agg, key=lambda c: -agg[c][0]):
        m, cf, hi, fb = agg[comp]
        if m == 0:
            continue
        print(f"{comp[:31]:32}{m:>9,}{cf:>10,}{hi:>8,}{fb:>9,}{100*(cf+hi)/m:>8.0f}%")

    # 2. false merges the crosswalk corrected (DOB-confirmed link points to a different UID
    #    than the legacy name-merge attached to that lineup player record)
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    fm_src = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    pid_eid = {}
    pid_eid_count = defaultdict(int)
    for eid, pid in con.execute(
            "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (espn_sid,)):
        pid_eid[pid] = eid
        pid_eid_count[pid] += 1
    pid_namemerge_uids = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute(
                "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            pid_namemerge_uids[pid].add(uid)
    xw = {r[0]: r[1] for r in con.execute(
        "SELECT espn_player_id, fm_uid FROM player_xwalk WHERE confidence='confirmed' AND fm_uid IS NOT NULL")}
    disambiguated = 0   # name-merge pulled in >=2 same-name UIDs; crosswalk picks the DOB-correct one
    hard_corrected = 0  # crosswalk UID is entirely outside what the name-merge attached
    for pid, eid in pid_eid.items():
        if pid_eid_count[pid] != 1:
            continue
        u = xw.get(eid)
        merged = pid_namemerge_uids.get(pid)
        if not u or not merged:
            continue
        if u not in merged:
            hard_corrected += 1
        elif len(merged) >= 2:
            disambiguated += 1
    print("\n" + "=" * 60)
    print("FALSE MERGES the DOB anchor corrected:")
    print(f"  disambiguated (name-merge had >=2 same-name UIDs, crosswalk picked the right one): {disambiguated:,}")
    print(f"  hard-corrected (crosswalk UID not among the name-merge's attached UIDs):           {hard_corrected:,}")

    # 3. flagged-for-review queue
    for conf in ("ambiguous", "unmatched"):
        n = con.execute("SELECT COUNT(*) FROM player_xwalk WHERE confidence=?", (conf,)).fetchone()[0]
        print(f"  flagged '{conf}': {n:,}")

    # 4. spot-checks
    print("\nSPOT-CHECKS (espn_id -> fm_uid / confidence / method):")
    for eid, who in (("173896", "Mohamed Salah"),):
        r = con.execute("SELECT fm_uid, confidence, method FROM player_xwalk WHERE espn_player_id=?", (eid,)).fetchone()
        print(f"  {who} ({eid}): uid={r[0]} conf={r[1]} method={r[2]}" if r else f"  {who}: not found")
    con.close()


if __name__ == "__main__":
    main()
