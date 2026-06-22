"""Build per-15-minute goal segments from match_event goal timings, for the autoregressive model.
For each match: seg[t] = (home_goals, away_goals) scored in window t (0:1-15, 1:16-30, 2:31-45+,
3:46-60, 4:61-75, 5:76-90+/ET). Validates that segments sum to the final score (drops mismatches, e.g.
own-goal side errors / missing events). Writes data/segments.npz (mids, seg (M,6,2), valid).
Usage: python D:/Programming/claude/FM/src/build_segments.py
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import db

GOAL = ("Goal", "Goal - Free-kick", "Goal - Header", "Goal - Volley", "Own Goal", "VAR - Goal Awarded")


def main():
    con = db.connect()
    fin = {r[0]: (r[1], r[2]) for r in con.execute("SELECT match_id,home_goals,away_goals FROM match WHERE home_goals IS NOT NULL")}
    ev = defaultdict(list)
    q = "SELECT match_id,minute,team_side FROM match_event WHERE type IN (%s)" % ",".join("?" * len(GOAL))
    for mid, minute, side in con.execute(q, GOAL):
        if mid in fin and side in ("home", "away") and minute:
            ev[mid].append((minute, side))
    mids, segs, valid = [], [], []
    nval = 0
    for mid, (fh, fa) in fin.items():
        seg = np.zeros((6, 2), np.int16)
        for minute, side in ev.get(mid, []):
            t = min((max(minute, 1) - 1) // 15, 5)
            seg[t][0 if side == "home" else 1] += 1
        ok = ev.get(mid) and seg[:, 0].sum() == fh and seg[:, 1].sum() == fa
        mids.append(mid); segs.append(seg); valid.append(1 if ok else 0); nval += bool(ok)
    out = db.ROOT / "data" / "segments.npz"
    np.savez_compressed(out, mids=np.array(mids, np.int64), seg=np.stack(segs), valid=np.array(valid, np.int8))
    print(f"saved {out}: {len(mids):,} matches, {nval:,} with VALID segments (sum==final score)", flush=True)
    # quick look at state-effect: scoring rate when leading vs trailing vs level (per 15-min window)
    lead = level = trail = 0.0; nl = nv = nt = 0
    for seg, v in zip(segs, valid):
        if not v: continue
        ch = ca = 0
        for t in range(6):
            diff = ch - ca   # home perspective at start of window
            gh = seg[t][0]
            if diff > 0: lead += gh; nl += 1
            elif diff == 0: level += gh; nv += 1
            else: trail += gh; nt += 1
            ch += seg[t][0]; ca += seg[t][1]
    print(f"home goals/15min — leading: {lead/max(nl,1):.3f}  level: {level/max(nv,1):.3f}  trailing: {trail/max(nt,1):.3f}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
