"""Per-match team-strength context the lineup can't see: Elo rating + recent form, computed over ALL
90k matches chronologically with strict no-leakage (pre-match values, ratings updated only after the
game). club_ids are self-consistent within the match table, so no cross-space bridging is needed.
Writes data/context.npz: mids (int64) + ctx (M,6) float32 aligned to it.
ctx columns: [home_elo, away_elo, elo_diff, home_form, away_form, form_diff, home_gdform, away_gdform,
home_rest, away_rest] (elo/400 scaled; form = mean points/game last 5; gdform = mean goal-difference
last 5; rest = days since last match, capped 14, /14). Read-only.
Usage: python D:/Programming/claude/FM/src/build_context.py
"""
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db

K = 20.0
HOME_ADV = 60.0      # Elo points
BASE = 1500.0


def main():
    con = db.connect()
    rows = con.execute(
        """SELECT match_id, match_date, home_club_id, away_club_id, home_goals, away_goals
           FROM match WHERE home_goals IS NOT NULL ORDER BY match_date, match_id""").fetchall()
    elo = defaultdict(lambda: BASE)
    form = defaultdict(lambda: deque(maxlen=5))     # recent points per club
    gdform = defaultdict(lambda: deque(maxlen=5))   # recent goal difference per club
    last_date = {}                                  # club -> last match date (np.datetime64)
    mids, ctx = [], []
    for mid, date, hc, ac, hg, ag in rows:
        d = np.datetime64(date[:10])
        eh, ea = elo[hc], elo[ac]
        fh = np.mean(form[hc]) if form[hc] else 1.0     # neutral prior 1 pt/game
        fa = np.mean(form[ac]) if form[ac] else 1.0
        gh = np.mean(gdform[hc]) if gdform[hc] else 0.0
        ga = np.mean(gdform[ac]) if gdform[ac] else 0.0
        rh = min((d - last_date[hc]) / np.timedelta64(1, "D"), 14.0) / 14.0 if hc in last_date else 0.5
        ra = min((d - last_date[ac]) / np.timedelta64(1, "D"), 14.0) / 14.0 if ac in last_date else 0.5
        mids.append(mid)
        ctx.append([eh / 400.0, ea / 400.0, (eh - ea) / 400.0, fh, fa, fh - fa, gh, ga, rh, ra])
        # update AFTER recording (no leakage)
        exp_h = 1.0 / (1.0 + 10 ** (-((eh + HOME_ADV) - ea) / 400.0))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo[hc] = eh + K * (s_h - exp_h)
        elo[ac] = ea - K * (s_h - exp_h)
        ph = 3 if hg > ag else (1 if hg == ag else 0)
        form[hc].append(ph); form[ac].append(3 - ph if ph != 1 else 1)
        gdform[hc].append(hg - ag); gdform[ac].append(ag - hg)
        last_date[hc] = d; last_date[ac] = d

    out = db.ROOT / "data" / "context.npz"
    np.savez_compressed(out, mids=np.array(mids, dtype=np.int64),
                        ctx=np.array(ctx, dtype=np.float32))
    print(f"saved {out}: {len(mids):,} matches, ctx shape {np.array(ctx).shape}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
