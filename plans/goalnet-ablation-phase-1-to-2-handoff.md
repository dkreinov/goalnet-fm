# Phase 1 → Phase 2 handoff — GoalNet ablation program

**Phase 1 (Reproducible ablation harness v2 + diagnostics + re-baseline): COMPLETE (2026-07-21).**
A fresh session can start Phase 2 from this file + `experiments/ablation/DESIGN.md` alone.

## What Phase 1 delivered

Every experiment is now one command → trains seeds → evaluates the frozen metric suite on all
lanes → appends a `registry.jsonl` row → regenerates `RESULTS_ABLATION.md`. The production recipe was
re-baselined under the new metrics and the "is the model coasting on the modal prior?" question is
answered quantitatively.

### Completed steps, validations, commits
| Step | What | Validation | Commit |
|---|---|---|---|
| 1 | DESIGN.md (schema/metrics/splits frozen) | doc keys present | 94921da |
| 2 | metrics.py (frozen suite) | `--selftest` PASS | 9354835 |
| 3 | splits.py (masks + leakage + frozen WC inputs) | `--report` LEAKAGE CHECKS PASS; pooled natl n=397 | f93d359 |
| 4 | run_ablation.py (runner + registry + report) | smoke row valid; dupe-name refused | 6e2d7e6 |
| 4-fix | build train/earlystop tensors once (memory) | 2-seed smoke exit 0 | e95c741 |
| 5 | two baseline rows (canonical anchor + pooled ref) | seed-7 == production bit-for-bit | 7db6727 |
| 6 | prior-coasting diagnostics (`--diagnose`) | report has Diagnostics + lift + Verdict | 275db2e |
| 7 | this handoff + phase-state + memory | checklist | (this commit) |

## Frozen contracts (all now implemented AND verified — do not silently change)

- **Registry** `experiments/ablation/registry.jsonl` (append-only, never edit/delete): one JSON row
  per run — `{name, ts, git_commit, dirty, config{npz,split,beta,W,seeds,epochs,rho_policy,
  ctx_extra,decay_halflife,flags,notes}, data{npz_mtime,n,ctx_dim,seed_earlystop_rps}, metrics{lane:{suite}}, wall_min}`.
- **Metric suite** (`metrics.py`, per lane): n, acc, rps, outcome_nll, grid_nll, grid_nll_prior,
  grid_info (=prior−model nats), ece_outcome (10-bin max-prob), sharpness (mean grid entropy),
  exact_rate, exact_lift (÷ always-modal), pts_g_31 (REFERENCE ONLY), exact_n. Prior null = train-mask
  empirical grid (no eval leakage). `lift_table` + `reliability` for diagnostics.
- **Splits** (`splits.py`): `pooled` (DEFAULT) train<2024-08 minus last-10%-by-date earlystop tail,
  eval = [2024-08, 2026-06-11); `canonical` (continuity) train<2024-08, earlystop=val[2024-08,2025-08),
  eval=test≥2025-08. Verified counts — pooled: train 42,628 / earlystop 4,758 / eval 21,663 (natl 397);
  canonical: train 47,386 / earlystop 11,210 / eval 10,457 (natl 203). Leakage asserts pass.
- **WC slate**: `experiments/ablation/wc_inputs.npz` — FROZEN raw inputs for all 104 finished WC2026
  games (197/2288 imputed starters). Scored, never trained on (Phases 1–5). DB still has zero WC2026
  matches (asserted at build).
- **Runner CLI**: `run_ablation.py --name <id> [--npz][--split pooled|canonical][--beta 3][--w 15]
  [--seeds 5][--epochs 150][--decay-halflife <y>][--ctx-extra f.npz ...][--notes][--force-rerun]`;
  `--diagnose <name>`; `--report`. Per-seed rates cached to `rates/<name>.s<k>.npz` (resume across
  kills) + consolidated `rates/<name>.npz` (diagnostics, no retrain). `rates/` + `diagnostics/` gitignored.
- `--decay-halflife` (exp time-decay sample weights) and `--ctx-extra` (concat extra context npz) are
  WIRED but inert by default — Phase 2 uses decay, Phase 3 uses ctx-extra.

## Baseline row values (the numbers Phase 2+ diff against)

**baseline-beta3-w15 (pooled reference):**
| lane | grid_info | grid_nll | rps | acc | ece | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|
| eval_all | −0.0317 | 3.0218 | 0.2102 | 0.496 | 0.018 | 0.92 | 0.718 |
| eval_natl | +0.1268 | 3.0683 | 0.1817 | 0.577 | 0.038 | 1.31 | 0.816 |
| wc_slate | +0.1463 | 3.0516 | 0.1595 | 0.654 | 0.122 | 1.17 | 0.923 |

**baseline-beta3-w15-canonical (anchor):** canonical_test_all rps 0.2108 pg 0.718 (5-seed);
canonical_test_natl grid_info +0.2337 rps 0.1752 exact_lift 1.35 pg 0.798; wc_slate grid_info +0.1978
rps 0.1553 pg 0.942. Tuned rho: canonical −0.05, pooled +0.05.

**Harness-faithfulness proof:** harness with seed 7 == production `train_goals.py` seed 7 to the digit:
val rps 0.2109 / TEST rps 0.2134 / pg 0.706 / exact 1125. The historical band (0.2130–0.2145) was the
single-seed-7 number; the 5-seed ensemble legitimately improves it. WC-slate ~98 pts ≈ production 97.

## Diagnostic verdict (the answer to "is it just guessing 1-0/1-1?")

Modal prior scoreline = 1-1 (P=0.123), but the model's EV-pick is **1-0** for 241/397 national games
and **never 1-1** — it does not echo the mode. On the **national/WC** lanes it adds genuine off-modal
score information: +0.127 / +0.146 nats over the prior, exact_lift 1.31× / 1.17×, off-modal EV-pick
precision 0.118 vs prior 0.058, outcome ECE 0.038. On the **broad all-competitions** lane grid_info is
**−0.032** — the model does NOT beat the prior on club-heavy fixtures; its scoreline edge is
concentrated on the national/WC lanes it is W-upweighted for (the intended target). Exact-cell
calibration is slightly over-confident. Conclusion: not coasting on the games that matter; little
scoreline info on club games (acceptable — not the target).

## Phase 2 — Loss-level levers (updated skeleton)

Goal unchanged: β∈{0,1,3} sweep, exponential time-decay half-life sweep, re-check national W under new
metrics. Concrete leads from Phase 1:
- **β sweep is well-motivated by data**: β=3 (decision-focused, EV-points) pushes EV-picks to 1-0 and
  yields eval_all grid_info=−0.032 / exact_lift 0.92. Hypothesis: **β=0 (pure Poisson) should improve
  grid_nll/calibration**, especially on eval_all, at some cost to pts_g_31 (reference only). Test whether
  a small β is a useful regularizer vs a points-bias that hurts the calibrated-distribution objective.
- **Decay**: `--decay-halflife` already implemented (0.5^(age_years/hl) sample weights). Sweep ~2/4/8y.
  Watch interaction with Elo context (already time-local). Gate on pooled eval_natl grid_info + no
  wc_slate regression.
- **W recheck**: eval_all being at/below prior while eval_natl is well above suggests W=15 trades club
  calibration for national. Sweep W and read grid_info on both lanes (not just pts).
- Gate: grid-NLL / calibration on pooled eval_natl + no wc_slate regression; pts_g_31 never a gate.
- Each experiment = `run_ablation.py --name <id> --beta <b> [--decay-halflife <h>] [--w <W>]`; compare
  in RESULTS_ABLATION.md Δ columns vs baseline-beta3-w15.

## Environment / operational notes for the next session

- Interpreter (bare `python` NOT on Bash PATH): `C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe`.
- NumPy2-vs-torch warning banner at import is pre-existing, non-fatal (prints an import-stack that
  looks like a traceback — ignore).
- **Background jobs are idle-killed here.** A 5-seed/150-epoch run is ~1h CPU. Options: keep the session
  active with back-to-back monitoring loops, OR rely on per-seed resume (relaunch same `--name`; the
  dupe-check keys on the registry, which has no row until the run fully finishes, so partial runs
  resume from `rates/<name>.s<k>.npz`). val_rps is vectorized; transformer training on CPU dominates.
- Data: `data/players_imp.npz` = 69,053 matches → 2026-06-14; `data/context.npz` = 10-dim ctx.

## Resume pointer
Phase state updated: phase 1 → COMPLETE, current phase = 2. Next action: plan Phase 2 (β/decay/W
sweep) as its own plan file before executing, per PHASE_MODE=pause-between-phases.
