# Ablation harness v2 — frozen design (Phase 1, 2026-07-21)

Purpose: every experiment = one config + one command → train → evaluate on a frozen metric suite →
append `registry.jsonl` row → regenerate `RESULTS_ABLATION.md`. Core-model philosophy (user-decided):
the network's job is a **calibrated scoreline distribution** P(h,a); tournament scoring schedules are
downstream heads. Gates are distributional; 3/1 fantasy points is a reference column only.

## Environment facts (verified 2026-07-21)

- Interpreter: `C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe`
  (bare `python` is NOT on the Bash-tool PATH — always use the full path in commands/logs).
- Dataset `data/players_imp.npz`: 69,053 matches, (11,62) per side + roles + imputed masks Ih/Ia,
  dates 2020-01-01 → 2026-06-14. Context `data/context.npz`: (90279, 10).
- WC source of truth `D:\Programming\claude\worldcup\team_db\{results,lineups}.json`:
  **104 finished games, all with lineups** (tournament complete).
  `experiments/wc_cache.npz` currently holds 90 games (stale) → Step 3 rebuilds it to 104 and it is
  then FROZEN (final ARG-ESP included).
- Reused code: model + scoring from `src/train_goals.py` (imported, never modified in Phase 1);
  grid/ensemble utilities from `experiments/eval_harness.py` (`make_grid`, `ens_grid`,
  `empirical_grid` pattern); season labels from `src/eval_seasons.py:season_of`.

## Split policy (DECISION)

Problem: the historical canonical split early-stops on val (2024-08→2025-07) and reports natl on
test only (n=203) — three features (value, FIFA, venue) won that n=203 lane and reversed on the WC
slate. Decision — **option (b)**, two named splits supported by the runner:

- `canonical` (continuity): train < 2024-08-01; early-stop on val [2024-08, 2025-08); eval on test
  ≥ 2025-08-01. Used ONCE per baseline for comparability with all RESULTS_WC2026.md history.
- `pooled` (DEFAULT for all new experiments): train < 2024-08-01 with early-stop on the
  **chronological tail of train** (last 10% of train matches by date); eval on
  **val ∪ test restricted to < 2026-06-11** (pre-WC cutoff so the eval lane and the WC slate are
  disjoint). Pooled natl n ≈ 400 (≥350 required), roughly doubling the national lane.

Leakage rules (asserted in `splits.py`): earlystop ⊂ train date-range; eval ∩ train = ∅;
eval ∩ WC-slate dates = ∅; WC slate is never trained on in Phases 1–5 (Phase 6 replay relaxes this
deliberately, day-by-day, walk-forward only).

## Metric suite (frozen names & formulas; implemented in `metrics.py`)

All computed per lane from per-match score grids P (10×10, MAXG=9, after DC-ρ and seed-averaging —
production convention `ens_grid`). Truth: (hg, ag) capped at MAXG; outcome y ∈ {H,D,A}.

| name | definition |
|---|---|
| `n` | matches in lane |
| `acc` | argmax outcome accuracy from grid-derived H/D/A probs |
| `rps` | mean ranked probability score over H/D/A (lower better) |
| `outcome_nll` | mean −log P(true outcome) |
| `grid_nll` | mean −log P(true scoreline cell) — the primary score-level metric |
| `grid_nll_prior` | same, using the **train-split empirical score grid** (the null model; computed on the experiment's own train mask only — no eval leakage into the null) |
| `grid_info` | grid_nll_prior − grid_nll (nats of score-level information added; >0 = model beats prior) |
| `ece_outcome` | 10-bin expected calibration error on the max-prob outcome (max-prob bin convention) |
| `sharpness` | mean grid entropy in nats (lower = sharper; only meaningful alongside calibration) |
| `exact_rate` | EV-pick exact-score hit rate (pick = `ev_pick`, production 3/1 EV) |
| `exact_lift` | exact_rate ÷ always-modal exact rate (modal scoreline of the train empirical grid scored on the lane; 1.0 = no better than always guessing the mode) |
| `pts_g_31` | fantasy pts/game under exact=3/outcome=1 (REFERENCE ONLY, never a gate) |
| `exact_n` | raw exact-hit count |

Diagnostics (Step 6, per named run): per-scoreline lift table — for each predicted cell: count,
precision, prior precision; for each true common scoreline: top-3-grid-mass recall; outcome and
exact-score reliability tables (bin, predicted, observed, n).

## Lanes

- `eval_all`, `eval_natl` — the split's eval set (pooled by default), natl = competition_id ∈ {9..15}.
- `canonical_test_all`, `canonical_test_natl` — only when `--split canonical`.
- `wc_slate` — the frozen 104-game WC2026 cache; scored with the SAME checkpointing as the run
  (train-split seeds — honest out-of-tournament view) — note this differs from the historical
  "full-data goalnet.pt" WC numbers (which trained through 2026-06-14); both facts recorded in
  RESULTS_ABLATION.md preamble.

## Registry (`experiments/ablation/registry.jsonl`, append-only)

One JSON object per line:
```json
{"name": str, "ts": iso8601, "git_commit": str, "dirty": bool,
 "config": {"npz": str, "split": "pooled|canonical", "beta": float, "W": float,
            "seeds": int, "epochs": int, "rho_policy": "val-tuned|fixed:<x>",
            "ctx_extra": [str], "flags": {…}, "notes": str},
 "data": {"npz_mtime": iso8601, "n": int, "ctx_dim": int},
 "metrics": {"<lane>": {<metric suite>}, …},
 "wall_min": float}
```
Rules: duplicate `name` → runner refuses unless `--force-rerun` (then both rows kept; report shows
latest per name with a rerun marker). Rows are never edited or deleted. `RESULTS_ABLATION.md` is
fully regenerated from the registry on every run — never hand-edited.

## Runner CLI (`run_ablation.py`)

```
run_ablation.py --name <id> [--npz players_imp.npz] [--split pooled|canonical]
                [--beta 3] [--w 15] [--seeds 5] [--epochs 150] [--decay-halflife <years>]
                [--ctx-extra <file.npz> ...] [--notes "..."] [--force-rerun]
run_ablation.py --diagnose <name>      # writes the diagnostics section for a registered run
run_ablation.py --report               # regenerate RESULTS_ABLATION.md only
```
`--decay-halflife` and `--ctx-extra` are declared now (schema stability) and implemented in
Phases 2–3. Per-seed rates for every run are cached to `experiments/ablation/rates/<name>.npz`
(same idea as eval_harness grids_cache) so pick-layer/diagnostic work never retrains.

## Report (`RESULTS_ABLATION.md`, generated)

Preamble (metric key + lane definitions + WC-slate caveat) → one table row per run (latest per
name): name, split, config summary, then per-lane `grid_info / grid_nll / rps / acc / ece /
exact_lift / pts_g_31`, with Δ-vs-baseline columns on the pooled lanes → per-run diagnostics
sections appended by `--diagnose`.

## Baseline anchoring (Step 5)

Two runs: `baseline-beta3-w15-canonical` (must reproduce TEST-all rps 0.2130–0.2145, pg 0.703–0.716
within seed noise — harness-bug tripwire) and `baseline-beta3-w15` (pooled — the reference row all
Phase 2+ experiments diff against). WC-slate lane additionally cross-checked against production
goalnet.pt's known realized score on the same 104 games (computed once via compare_models
convention) — recorded in the report preamble, not as a registry row.

## Phase-2 adopted defaults (frozen 2026-07-22)

Phase 2 tested loss-level levers against `baseline-beta3-w15`. **Adopted core-training config
for all Phase-3+ ablation experiments: `--beta 0 --w 1`** (pure Poisson, no national upweight).

Rationale (registry evidence, pooled eval_natl grid_info vs baseline +0.1268):
- **β=0** (drop the EV-points decision term): +0.2522 (Δ+0.125) — strongest single lever. The β=3
  decision term was a points-bias that hurt the calibrated-distribution objective on every lane
  (baseline eval_all grid_info was NEGATIVE, −0.032).
- **W=1** (drop the 15× national upweight): +0.2126 (Δ+0.086) — the upweight was the same kind of
  points-bias; removing it helps both lanes. W=40 (more upweight) fails the gate (wc pg −0.058).
- **Combo β0+W1 (ADOPTED)**: natl +0.2432, wc **+0.2992** (best), all **+0.0762** (best, now POSITIVE),
  exact_lift **1.33** (recovers the pick ability β0-alone gave up), wc pg **0.933** (best). The two
  levers don't stack additively (β0 already captures most of the natl gain) but together give the
  best-rounded model AND drop an arbitrary hyperparameter. Robust at 10 seeds (natl +0.2414).
  Canonical split confirms wins on every lane + better WC points (pg 0.971 vs 0.942).
- **Time-decay: DEFERRED to Phase 3.** hl∈{2,4,8} each beat the β3/W15 baseline standalone
  (+0.045..+0.080 natl) but non-monotonically (half-life is noise-level), and adding decay-hl8 on top
  of β0+W1 gives nothing (Δnatl +0.004, wc pg −0.019) — its gain was subsumed by removing the biases.
  Recency is thus already handled; Phase-3 Elo-momentum/trajectory features must earn their keep by
  adding signal BEYOND time-decay, not just recency.

`pts_g_31` remained reference-only throughout (never gated); notably the adopted config improves it too.
The historical production `goalnet.pt` (β=3, W=15, --natl-finetune) is NOT retrained here — production
retrain is a Phase-6 decision. `src/train_goals.py` is unchanged; the adopted config lives as the
harness default for experiments.

## Phase-3 adopted context (frozen 2026-07-22)

**Adopted context = NONE NEW.** The base 10-feature `context.npz` (Elo level, form level, goal-diff
level, rest-days) stands; no `--ctx-extra` bundle joins the default. Phase-3 experiment config is
unchanged from Phase 2: `--beta 0 --w 1`.

Evidence:
- **Elo-momentum / form-trajectory (`ctx_momentum.npz`, REJECTED):** vs `combo-beta0-w1`, eval_natl
  grid_info −0.0044 (gate needs ≥+0.02), eval_all −0.0028, rps flat. Trajectory/direction adds no
  independent score-level signal once the base context supplies Elo/form LEVEL and Phase-2's β0+W1
  already subsumes recency. Elo-delta correlates with level; form-trend with mean-form. Informative
  null at 69k matches / 397 national eval, as the Phase-2→3 handoff predicted.
- **Stage/knockout (DEFERRED — needs a data-collection step, NOT rejected):** no stage/round column
  exists in the DB today (`match_kind` ∈ {league, national} only), but the data IS collectable — WC
  team_db already carries stage, and historical tournament rounds are scrapeable (ESPN summaries /
  martj42 / transfermarkt) via the project's throttled fetch+cache. So this feature is OWED an honest
  ablation after a backfill sub-phase; it was not tested here only for lack of *already-collected*
  data, not on the merits. (Standing rule: never permanently skip a feature for missing data —
  collect-then-test.) Rest-days is already in the base context.

Takeaway: re-derived context features (level/recency/trajectory) are at their ceiling for this model
+ data scale. The next lever must bring **genuinely new information** — Phase 4 (de-vigged bookmaker
odds anchor), which the literature (see memory) flags as the strongest untried signal.
`src/build_momentum.py` is kept (leakage-free trajectory builder) in case a larger dataset revisits it.

## Phase-4 adopted market/stage config (frozen 2026-07-22)

**ADOPTED: the market signal is real (+~0.024 nats on national games where odds exist), captured
equivalently by either the odds FEATURE or the post-hoc BLEND — they do NOT stack (same signal).**

Covered-subset evidence (the 137 odds-covered pooled-eval national matches, identical for all runs):
| config | grid_info | rps | acc | ece |
|---|---|---|---|---|
| combo-beta0-w1 (base) | +0.1543 | 0.2007 | 0.511 | 0.100 |
| **ctx-odds (Arm A, feature)** | **+0.1783** | **0.1923** | **0.562** | **0.054** |
| base + B1 blend λ0.9 | +0.1790 | 0.1925 | 0.526 | 0.082 |
| ctx-odds + blend λ0.5 (best known) | +0.1803 | 0.1918 | 0.533 | 0.042 |
| anchor-kl01 / kl03 (B2) | +0.152 / +0.152 | ~base | ~base | — |
| ctx-stage (Arm S) | +0.1225 | 0.2011 | 0.533 | — |

Verdicts:
- **Arm A (ctx-odds feature): ADOPTED for market-aware configs.** Best single config on covered
  natl (rps/acc/ece all best); also improves the 56%-covered full club lane (eval_all grid_info
  +0.076→+0.084, rps −0.0025). Caveat: full-natl lane reads −0.012 vs base because 65% of that lane
  lacks odds (dilution + mild mask noise) — so it is NOT the Phase-5 experiment default (see below).
- **Arm B1 (post-hoc outcome-mass blend, λ*=0.9): ADOPTED as the zero-training production layer.**
  Equivalent gain to the feature, applies to ANY model without retraining, no full-lane downside
  (identity where odds are missing). λ tuned on the odds-covered earlystop subset; monotonic to 0.9
  — the market dominates outcome opinion (classic model≈market result, now on our own data).
- **Arm B2 (training-time KL anchor): REJECTED** — neutral everywhere (w∈{0.1,0.3}); the anchor
  does not transfer beyond what the feature/blend already capture.
- **Arm S (stage/knockout): REJECTED as tested** — the thin ET/pen-derived knockout flag (31 covered
  matches) adds nothing (covered −0.032). A richer stage feature would need real round labels
  (future data collection); the collect-then-test debt for THIS representation is paid.

**Phase-5 experiment default stays `combo-beta0-w1` (no odds feature)** so the frozen WC-slate lane
(no odds available for it) remains comparable across runs. Phase-5 candidates must ALSO be scored
with the odds layer (feature or blend) before any adoption; Phase-6 replay/production must include
the market layer — recommended production shape: ctx-odds feature + λ≈0.5 blend (best known:
+0.1803 / rps 0.1918 / ece 0.042).

**Coverage caveat + multi-source TODO:** natl odds coverage is 62% train / 35% eval (BetExplorer
lacks senior friendlies + 2026-cycle qualifiers; WC-slate odds unavailable → wc lane skipped on
odds runs). Next sources: oddsportal (free) then The Odds API (paid, 2022→) to fill eval-window
qualifiers/friendlies + the WC slate. Odds inventory: `data/ctx_odds.npz` = 38,403 matches
(37,665 club from DB football-data columns + 738 scraped national).
