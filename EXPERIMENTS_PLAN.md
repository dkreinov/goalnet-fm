# Overnight experiment plan (2026-06-22)

Goal: win the WC2026 fantasy league (scoring **exact=3, outcome=1, wrong=0**; user=YOU). Model is at the
information ceiling (RPS ~0.205); the untapped surface is the **pick/betting layer** and **new features**.
Every experiment A/Bs on a fixed harness: held-out test (n=10,457; natl=203) by RPS / pts/g / exact-count,
AND the 40 played WC2026 games (pts under 3/1). Baseline = current 5-seed ensemble goalnet.pt.

## Grounded data facts (recon 2026-06-22)
- DB `match` already has: b365/avg 1X2 odds (46% of matches, **0% of nationals**), xG (12%), venue/city/
  attendance/kickoff (~99%), formations, referee, ht scores.
- **Odds reach club games only** → market is a *prediction-time blend* for WC (via `data/wc_odds.csv`), NOT a
  training feature that reaches nationals.
- Score clusters: 1-1 12.2%, 1-0 10.3%, 2-1 8.8%, 0-1 7.8%, 0-0 7.6%, 2-0 7.3% → empirical-prior ready.
- 2,707 national matches with lineups (match_player) → squad-cohesion computable from DB.

## Queue (cheapest / highest league-value first)

### Phase 1 — pick/betting layer, NO retrain (reuse ensemble grids)
- **E0 harness**: `experiments/eval_harness.py` — dump ensemble score grids for test + WC played once; score
  any grid-transform or pick-policy against it. Foundation for all of Phase 1.
- **E1 empirical score-prior blend**: P=(1-α)·model+α·empirical, tune α on val for exact-hit & pts/g.
- **E2 exact-score calibration**: jointly tune DC ρ + grid sharpening γ for fantasy pts (not RPS).
- **E3 game-theoretic picks**: Monte-Carlo a field of opponents (chalk = bookmaker/modal + noise); optimise
  pick for P(rank #1) & E(rank) instead of E(points). Adaptive risk vs leaderboard.
- **E4 market blend (WC games)**: market-DC grid from wc_odds.csv blended with model; tune weight; A/B pick.

### Phase 2 — retrain features (~20-40min CPU each)
- **E5 market-as-feature (club-only)**: add de-vigged odds to ctx; measure lift where odds exist; honest re
  NATL gap.
- **E6 squad cohesion**: caps-played-together / lineup-stability feature from national match history.
- **E7 player-level Poisson**: per-attacker goal rate from FM finishing/off-the-ball → summed team λ; better
  exact-score shape.
- **E8 multinomial top-K head**: classify over ~20 common scorelines vs factored Poisson.

### Phase 3 — collect data + WC-specific
- **E9 venue altitude/heat/travel**: collect WC2026 host-city altitude+climate+travel; nationals context.
- **E10 injury/lineup news**: collect current injury/availability for upcoming WC games (prediction-time).

### Phase 4 — crazy
- **E11 tournament-forward sim**: roll the bracket, leaderboard-aware risk budget (extends E3).
- **E12 LLM-as-feature**: team-news/form text → numeric prior; cheap A/B.

Results appended to RESULTS_WC2026.md; roadmap memory updated per experiment.
