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
