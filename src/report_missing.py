"""Per league x season GAP report: how many matches are NOT training-ready.
not-ready = missing lineup OR lineups present but not all 11+11 starters graded.
Reads match/match_player (DB) for totals+lineups, dataset.parquet for grade coverage.
"""
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import db
import leagues as L

con = db.connect()
RANK = {l["name"]: l["rank"] for l in (L.LEAGUES + L.EXTRA_LEAGUES + L.UEFA_CUPS)}
SEASONS = ["2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"]

# total completed matches per (competition, season)
M = defaultdict(lambda: defaultdict(int))
for comp, season, n in con.execute(
    """SELECT c.name, s.label, COUNT(*)
       FROM match m JOIN competition c ON c.competition_id=m.competition_id
                    JOIN season s ON s.season_id=m.season_id
       WHERE m.home_goals IS NOT NULL GROUP BY c.name, s.label"""):
    M[comp][season] = n
# matches with >=1 lineup row
Ln = defaultdict(lambda: defaultdict(int))
for comp, season, n in con.execute(
    """SELECT c.name, s.label, COUNT(DISTINCT mp.match_id)
       FROM match m JOIN competition c ON c.competition_id=m.competition_id
                    JOIN season s ON s.season_id=m.season_id
                    JOIN match_player mp ON mp.match_id=m.match_id
       WHERE m.home_goals IS NOT NULL GROUP BY c.name, s.label"""):
    Ln[comp][season] = n

# fully-graded matches (both sides >=11 starters with a grade) from dataset
df = pd.read_parquet(db.ROOT / "data" / "dataset.parquet")
# conservative readiness: count xwalk/fallback + HIGH-confidence roster links, but NOT
# the riskier medium roster guesses (still present in the parquet for ablation).
hm = df["home_n_matched"] - df.get("home_n_roster_medium", 0)
am = df["away_n_matched"] - df.get("away_n_roster_medium", 0)
df["ready"] = (hm >= 11) & (am >= 11)
G = defaultdict(lambda: defaultdict(int))
for (comp, season), g in df.groupby(["competition", "season"]):
    G[comp][season] = int(g["ready"].sum())

comps = sorted(M, key=lambda c: (RANK.get(c, 999), c))
print("NOT TRAINING-READY matches per league x season  (gap = total - fully-graded)")
print("  ready = both lineups present AND all 11+11 starters carry an FM grade")
print("="*104)
print(f"{'LEAGUE':30}" + "".join(f"{s[2:]:>11}" for s in SEASONS) + f"{'TOT gap':>12}")
tot_gap = tot_all = 0
for comp in comps:
    if RANK.get(comp, 999) > 45:   # skip national-team comps here
        continue
    cells = []; gap_sum = 0; all_sum = 0
    for s in SEASONS:
        m = M[comp].get(s, 0); g = G[comp].get(s, 0)
        gap = m - g
        gap_sum += gap; all_sum += m
        cells.append("·" if m == 0 else f"{gap}/{m}")
    tot_gap += gap_sum; tot_all += all_sum
    print(f"{comp[:29]:30}" + "".join(f"{c:>11}" for c in cells) + f"{f'{gap_sum}/{all_sum}':>12}")
print("="*104)
print(f"{'TOTAL gap (not-ready / all)':30}" + " "*66 + f"{f'{tot_gap}/{tot_all}':>12}")
print(f"\ntraining-ready now: {tot_all - tot_gap:,} of {tot_all:,} matches "
      f"({100*(tot_all-tot_gap)/tot_all:.0f}%)")

# split the gap into its two causes
miss_lu = ungraded = 0
for comp in comps:
    if RANK.get(comp, 999) > 45:
        continue
    for s in SEASONS:
        m = M[comp].get(s, 0); ln = Ln[comp].get(s, 0); g = G[comp].get(s, 0)
        miss_lu += max(0, m - ln)          # no lineup at all
        ungraded += max(0, ln - g)         # lineup present but not all starters graded
print(f"  gap cause: {miss_lu:,} missing lineups + {ungraded:,} lineup-but-not-fully-graded")
