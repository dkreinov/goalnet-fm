# GoalNet post-WC2026 ablation program — phase state

**Overall goal:** Rebuild GoalNet as a tournament-agnostic, calibrated scoreline predictor
(core model = P(home_goals, away_goals); tournament scoring = separate pick-layer heads), via a
reproducible ablation harness, then run the experiment program: loss purity, time-decay, momentum,
stage/rest, market anchor (BetExplorer odds), architecture (cross-team attention, plus-minus),
day-by-day WC2026 replay backtest.

**User-decided principles (2026-07-21, do not re-litigate):**
- Primary output = accurate/calibrated score distribution. NOT fantasy points. Points under any
  scoring schedule are computed by downstream heads (existing `--strategy` layer pattern).
- Must answer the "is the network just guessing modal scores (1-0/1-1)?" question quantitatively:
  grid-NLL vs empirical-prior null, per-scoreline lift, calibration (ECE/reliability), sharpness.
- Day-by-day tournament replay backtest (walk-forward, only-past-info) is a first-class eval mode.
- Continuous data collection continues; future uses include fantasy (EPL money-league) heads.
- Free data sources only (BetExplorer scrape, not The Odds API) unless user approves spend later.

**Branch:** master (repo's working branch — no `main` exists).

## Phase map

| # | Phase | Status | Plan file | Handoff |
|---|---|---|---|---|
| 1 | Harness v2 + diagnostics + re-baseline | ✅ COMPLETE (2026-07-21) | plans/goalnet-ablation-phase-1-harness-plan.md | plans/goalnet-ablation-phase-1-to-2-handoff.md |
| 2 | Loss-level levers: β purity, time-decay, natl-weight recheck | ✅ COMPLETE (2026-07-22) | plans/goalnet-ablation-phase-2-loss-levers-plan.md | plans/goalnet-ablation-phase-2-to-3-handoff.md |
| 3 | Context features: Elo momentum; stage/rest backfill + feature | ✅ COMPLETE (2026-07-22, null) | plans/goalnet-ablation-phase-3-context-features-plan.md | plans/goalnet-ablation-phase-3-to-4-handoff.md |
| 4 | Market anchor: BetExplorer scrape → de-vigged odds feature | ✅ COMPLETE (2026-07-22, ADOPTED) | plans/goalnet-ablation-phase-4-market-anchor-plan.md | plans/goalnet-ablation-phase-4-to-5-handoff.md |
| 5 | Architecture: cross-team attention; plus-minus ratings | ✅ COMPLETE (2026-07-22, NULL) | plans/goalnet-ablation-phase-5-architecture-plan.md | plans/goalnet-ablation-phase-5-to-6-handoff.md |
| 6 | WC2026 day-by-day replay backtest + model selection + production retrain | 🔄 IN PROGRESS (planned 2026-07-22) | plans/goalnet-ablation-phase-6-replay-production-plan.md | — (final phase; retrospective instead) |

**Current phase:** 6 (EXECUTING, SESSION_MODE: autonomous). Gate decisions: oddsportal-first (no
paid API), bench feature backlogged, fine-tune ALL candidates, FULL production cutover.
PHASE_MODE: moot (final phase).
**Last completed step:** Phase 5 COMPLETE (NULL) 2026-07-22 — see
plans/goalnet-ablation-phase-5-to-6-handoff.md. Registry 23 rows.
**Next action:** PLAN Phase 6. Candidate set is small: combo-beta0-w1 core ± market layer
(feature / λ-blend / both). Decide early whether multi-source odds collection (oddsportal free →
The Odds API paid) runs first — there are NO WC2026 odds yet, so the replay's market layer needs it.
Production retrain lifts the train_goals/goalnet.pt no-edit rule ONLY in its retrain step.
**Phase-5 result (NULL):** all 4 arms below baseline on eval_natl grid_info (cross22 −0.018,
latecross −0.017, ctx-pm −0.018, pm-channel −0.012): cross-team attention dilutes at this scale;
plus-minus is a rotation proxy (inverse corr −0.10) with no incremental signal. Per-team GoalNet
β0/W1 stands; only NEW info (odds) moves the needle. Kept infra: --arch/models.py (parity
verified), --pm-channel, build_plusminus.py, players_pm/ctx_pm npz.
**Phase-4 result (ADOPTED):** market signal is real (+0.024 nats covered natl; acc +5pp) — captured
equivalently by ctx-odds feature or post-hoc λ0.9 blend (no stacking); KL training-anchor neutral;
thin stage flag rejected. ctx_odds.npz = 38,403 matches (club DB + scraped natl).
**Phase-2 result:** stripping the two points-biases (β=3→0, W=15→1) adds +0.10..+0.13 nats national/WC
score-level info, flips the club lane positive, keeps exact-pick ability — adopted β0+W1.
**Phase-3 result (NULL):** Elo-momentum/trajectory adds nothing over β0+W1 (eval_natl grid_info −0.004);
stage not backfillable. Re-derived context features are at their ceiling → next lever must bring NEW
information (market odds). Base context stands; goalnet.pt unchanged.
**Phase-1 result in one line:** harness reproduces production bit-for-bit (seed-7 TEST rps 0.2134);
pooled reference registered (eval_natl grid_info +0.127, wc_slate +0.146); the model adds genuine
off-modal score info on national/WC (EV-picks 1-0 not the modal 1-1) but sits at/below the prior on
club-heavy fixtures (eval_all grid_info −0.032) — key motivation for the Phase-2 β sweep.
**Session facts a fresh context needs:** bare `python` NOT on Bash PATH — always use the full
interpreter path above; NumPy2-vs-torch warning banner at import is pre-existing and non-fatal;
`data/players_imp.npz` = 69,053 matches ending 2026-06-14; DB has ZERO WC2026 matches (pre-WC
context is leakage-free for the slate); worldcup results.json = 104 finished games, all with
lineups; old `experiments/wc_cache.npz` (90 games, rates-only) is superseded for the harness by
`experiments/ablation/wc_inputs.npz` (raw inputs, built by splits.py) but kept for eval_harness.

## Frozen contracts (updated as phases complete)

- Registry: `experiments/ablation/registry.jsonl` — one JSON object per experiment run (schema
  frozen in Phase 1 Step 1 DESIGN.md; includes name, config, git_commit, data_hash, seeds,
  per-lane metric dict, timestamp).
- Report: `experiments/ablation/RESULTS_ABLATION.md` — regenerated from registry, never hand-edited.
- Metric suite (Phase 1): rps, acc, outcome_nll, grid_nll, grid_nll_prior (null baseline),
  ece_outcome, sharpness (mean grid entropy), exact_rate, exact_lift_vs_modal, pts_g_31 (reference
  only), per lane: ALL / NATL-pooled / WC2026-static-slate (104 games, frozen in wc_cache.npz).
- Existing conventions to follow: `build_X.py` → `data/X.npz` → `train_goals.py --X` flag;
  throttled `src/fetch.py` + disk cache for any scraping; logs to `data/_*.log`; results
  narrative appended to RESULTS_WC2026.md only for adopted changes.

## Later-phase skeletons

### Phase 2 — Loss-level levers
Inputs: Phase-1 harness + baseline registry rows. Goal: β∈{0,1,3} sweep (is decision term a useful
regularizer or a points-bias?), exponential time-decay sample weights (half-life sweep ~2/4/8y),
re-check national W under new metrics. Outputs: registry rows + adopt/reject verdicts; possibly new
default core loss. Files: run configs only (harness does the work) + train_goals.py flag for decay.
Gate: grid-NLL/calibration on pooled-natl + no WC-slate regression. Risks: β=0 may hurt acc as well
as points (then core keeps small β); decay interacts with Elo context (Elo already time-local).

### Phase 3 — Context features
Goal: (a) Elo-trajectory/momentum features (extend build_context.py, windowed Elo deltas);
(b) investigate stage/round backfill source (ESPN cached summaries / worldcup team_db) → stage +
rest-days features; knockout-shrinkage tested as stage-conditioned context. Outputs: context.npz v2
(versioned, back-compatible), registry rows. Risks: stage data may not exist for historical
nationals → scope to what's backfillable; rest-days already partially in ctx (verify first).

### Phase 4 — Market anchor
Goal: scrape BetExplorer closing 1X2 for internationals 2015→2026 (all confederations + friendlies),
join by date+normalized team names (martj42-style spine; reuse club_alias machinery), de-vig (Shin),
build odds feature (probs + source flag + missingness mask), A/B as ctx feature AND as
residual-anchor training. Outputs: new scraper src/scrape_betexplorer.py, data/natl_odds.csv,
feature npz, registry rows. Risks: site structure/anti-bot; coverage of pre-2018 friendlies;
name matching for national teams (should be easy — small fixed set).

### Phase 5 — Architecture
Goal: (a) cross-team attention variant (match-comparison block over both XIs, HIGFormer-style) in
train_goals model zoo; (b) plus-minus player ratings computed from our own 90k-match DB as
player-level or ctx features. Outputs: registry rows; adopt only on clear pooled-natl +
WC-slate win. Risks: CPU cost of cross-attention (keep d small); plus-minus needs careful
leakage-free chronological computation.

### Phase 6 — Replay backtest + selection + production
Goal: walk-forward WC2026 day-by-day replay (only-past-info context, real lineups, optional
incremental fine-tune); compare candidate models from Phases 2–5; select + retrain production
core (goalnet v2) + keep 3/1 head as separate layer; update docs (README/RESULTS/HOW_TO_PREDICT),
memory, and archive old checkpoints. Gate: replay pts + grid-NLL beat current production replay.

## Open questions / blockers
- none currently (paid Odds API explicitly deferred; stage-source investigation is Phase 3 Step 1).
