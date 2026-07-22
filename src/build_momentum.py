"""Elo-momentum / form-trajectory context bundle (Phase-3 ablation feature, joins via --ctx-extra).

The base context.npz already carries Elo LEVEL, form LEVEL (mean pts last 5), goal-diff level and
rest-days. This bundle adds TRAJECTORY — direction/change the levels miss — computed over ALL matches
chronologically with the SAME K/HOME_ADV/BASE and update order as build_context.py, strict no-leakage
(pre-match values only, ratings updated after the game).

feats columns (M,7): [home_elo_mom, away_elo_mom, elo_mom_diff, home_form_trend, away_form_trend,
form_trend_diff, mom_cov] where
  elo_mom      = (elo_now - elo_5_team-matches_ago)/400   (0 until a team has >=5 prior matches)
  form_trend   = OLS slope of the last-5 result-points sequence (0 until >=3 prior matches)
  mom_cov      = fraction of the two teams with full (>=5) Elo history (1.0/0.5/0.0) = a missingness signal
Writes data/ctx_momentum.npz: mids (int64) + feats (M,7) float32 aligned to it. Read-only.
Usage: python D:/Programming/claude/FM/src/build_momentum.py
"""
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db

K = 20.0
HOME_ADV = 60.0
BASE = 1500.0
WIN = 5                       # look-back window (team-matches) for the Elo delta


def _slope(pts):
    """OLS slope of a short points sequence (direction of recent form); 0 if <3 points."""
    n = len(pts)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    return float(np.polyfit(x, np.asarray(pts, np.float64), 1)[0])


def main():
    con = db.connect()
    rows = con.execute(
        """SELECT match_id, match_date, home_club_id, away_club_id, home_goals, away_goals
           FROM match WHERE home_goals IS NOT NULL ORDER BY match_date, match_id""").fetchall()
    elo = defaultdict(lambda: BASE)
    elo_hist = defaultdict(lambda: deque(maxlen=WIN))   # team -> last WIN pre-match Elo values
    form = defaultdict(lambda: deque(maxlen=5))         # team -> last 5 result points
    mids, feats = [], []
    for mid, date, hc, ac, hg, ag in rows:
        eh, ea = elo[hc], elo[ac]
        # Elo momentum = change over the last WIN team-matches (needs WIN prior records)
        hh, ah = elo_hist[hc], elo_hist[ac]
        h_full = len(hh) >= WIN
        a_full = len(ah) >= WIN
        h_mom = (eh - hh[0]) / 400.0 if h_full else 0.0
        a_mom = (ea - ah[0]) / 400.0 if a_full else 0.0
        h_trend = _slope(form[hc])
        a_trend = _slope(form[ac])
        cov = (int(h_full) + int(a_full)) / 2.0
        mids.append(mid)
        feats.append([h_mom, a_mom, h_mom - a_mom, h_trend, a_trend, h_trend - a_trend, cov])
        # update AFTER recording (no leakage) — mirror build_context exactly
        elo_hist[hc].append(eh); elo_hist[ac].append(ea)
        exp_h = 1.0 / (1.0 + 10 ** (-((eh + HOME_ADV) - ea) / 400.0))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo[hc] = eh + K * (s_h - exp_h)
        elo[ac] = ea - K * (s_h - exp_h)
        ph = 3 if hg > ag else (1 if hg == ag else 0)
        form[hc].append(ph); form[ac].append(3 - ph if ph != 1 else 1)

    feats = np.array(feats, np.float32)
    out = db.ROOT / "data" / "ctx_momentum.npz"
    np.savez_compressed(out, mids=np.array(mids, dtype=np.int64), feats=feats)
    con.close()
    cols = ["home_elo_mom", "away_elo_mom", "elo_mom_diff", "home_form_trend", "away_form_trend",
            "form_trend_diff", "mom_cov"]
    print(f"saved {out}: {len(mids):,} matches, feats {feats.shape}", flush=True)
    print("  col means:", dict(zip(cols, np.round(feats.mean(0), 4))), flush=True)
    print("  col stds :", dict(zip(cols, np.round(feats.std(0), 4))), flush=True)
    print(f"  full-Elo-history coverage (mom_cov==1): {(feats[:,6]==1.0).mean()*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
