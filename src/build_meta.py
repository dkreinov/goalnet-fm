"""Per-match metadata for the feature-ablation experiments. Writes data/meta.npz keyed to match_id:
  comp   int   competition_id - 1            (0..53, for an embedding)
  kh     f32   kickoff hour / 23, else -1     (time-of-day)
  logatt f32   log1p(attendance)              (crowd size; 0 if missing)
  hasatt f32   1 if attendance present
  hform  int   home formation id (0=missing, else 1..N)
  aform  int   away formation id
Usage: python D:/Programming/claude/FM/src/build_meta.py
"""
import sys
import re
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db


def main():
    con = db.connect()
    rows = con.execute("""SELECT match_id, competition_id, kickoff_time, attendance,
                                 home_formation, away_formation FROM match WHERE home_goals IS NOT NULL""").fetchall()
    forms = sorted({r[4] for r in rows if r[4]} | {r[5] for r in rows if r[5]})
    fid = {f: i + 1 for i, f in enumerate(forms)}     # 0 reserved for missing
    print(f"{len(rows):,} matches, {len(forms)} formations", flush=True)
    mids, comp, kh, logatt, hasatt, hform, aform = [], [], [], [], [], [], []
    for mid, cid, kt, att, hf, af in rows:
        mids.append(mid)
        comp.append((cid or 1) - 1)
        h = -1.0
        if kt:
            m = re.search(r"T(\d\d):", kt)
            if m:
                h = int(m.group(1)) / 23.0
        kh.append(h)
        logatt.append(float(np.log1p(att)) if att else 0.0)
        hasatt.append(1.0 if att else 0.0)
        hform.append(fid.get(hf, 0)); aform.append(fid.get(af, 0))
    out = db.ROOT / "data" / "meta.npz"
    np.savez_compressed(out, mids=np.array(mids, np.int64), comp=np.array(comp, np.int64),
                        kh=np.array(kh, np.float32), logatt=np.array(logatt, np.float32),
                        hasatt=np.array(hasatt, np.float32), hform=np.array(hform, np.int64),
                        aform=np.array(aform, np.int64), forms=np.array(forms))
    print(f"saved {out}: comp 0..{max(comp)}, formations 0..{len(forms)}, "
          f"attendance present {int(sum(hasatt)):,}/{len(rows):,}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
