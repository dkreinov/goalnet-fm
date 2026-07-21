# Phase 1 — Reproducible ablation harness v2 + diagnostics + re-baseline

**Consumes:** phase state `plans/goalnet-ablation-phase-state.md` (read it first — user-decided
principles + frozen contracts live there). No prior phase handoff (this is Phase 1).
**Produces for Phase 2:** working harness, frozen metric/registry contracts, baseline registry rows,
the prior-coasting diagnostic answer, handoff file.

**Goal:** Every future experiment = one config + one command → trains, evaluates on the frozen metric
suite across all lanes, appends a registry row, regenerates a shareable report. Then re-baseline the
production recipe under the new metrics and answer: "is the network genuinely predicting scorelines,
or coasting on the modal-score prior?"

**Execution mode:** not chosen yet. **Model:** Opus (harness design within established patterns).
**Thinking budget:** High. **Context:** RECOMMENDED /clear before execution (session holds two large
research dumps). **Subagents:** NOT allowed (user hasn't authorized; long trainings use background
Bash instead).

**Assumptions (stated, not asked):**
- 3/1 pts stays in the metric suite as a *reference* column, never a gate.
- The WC2026 slate (wc_cache.npz, committed cefa470) is the frozen on-target benchmark; it is
  scored, never trained on, in Phase 1.
- Existing `eval_harness.py` stays untouched (pick-layer experiments still work); new harness lives
  in `experiments/ablation/` and imports from train_goals rather than duplicating model code.
- Free data only; no new scraping in Phase 1.

---

## Step 1: Inventory + freeze the harness design (DESIGN.md)

**Goal**: Read the code the harness must reuse; freeze the registry schema, metric definitions, and
split policy in a design doc. Key open design point to resolve HERE: the national-lane eval set.
Current test natl n=203 caused three held-out→WC reversals. Options to evaluate in the doc:
(a) keep canonical split, report natl on test only; (b) merge val+test for eval (natl n≈400) and
early-stop on the chronological tail of train instead; (c) leave-one-season-out (expensive).
Recommendation to write up: (b), with the canonical-split numbers ALSO reported once for continuity
with all historical RESULTS_WC2026.md tables.

**Files**: `experiments/ablation/DESIGN.md` (new). Read-only: `experiments/eval_harness.py`,
`src/train_goals.py`, `src/eval_seasons.py`, `src/build_player_dataset_imp.py` (data shapes).

**Commands**: None (reads + one doc write)

**Success criteria**: DESIGN.md contains: registry JSONL schema (field names + types), each metric's
exact formula (incl. grid-NLL null = empirical prior computed on TRAIN split only — no test leakage
into the null), split policy decision with rationale, runner CLI contract, report format.

**Validation**: `python -c "import json,pathlib; t=pathlib.Path('experiments/ablation/DESIGN.md').read_text(encoding='utf-8'); assert all(k in t for k in ['registry.jsonl','grid_nll','split policy','exact_lift']); print('DESIGN ok')"`

### After completing this step:
- [ ] Run validation, report result
- [ ] Update work_log.md
- [ ] Commit: `git add experiments/ablation/DESIGN.md && git commit -m "docs: Step 1 - freeze ablation harness design (registry schema, metrics, splits)"`
- [ ] Summary + Next: Step 2 builds metrics.py
- [ ] STOP (step-by-step) / proceed (autonomous)

---

## Step 2: Metrics module with self-test

**Goal**: `experiments/ablation/metrics.py` implementing the frozen suite: rps, acc, outcome_nll,
grid_nll, grid_nll_prior, ece_outcome (10-bin), sharpness (mean grid entropy), exact_rate,
exact_lift_vs_modal (exact-rate ÷ always-modal-score exact-rate), pts_g_31 (reference), plus a
per-scoreline lift table function (predicted-bucket precision & top-3-recall vs prior). Pure
numpy, takes (grids, y, hg, ag) like eval_harness.score_grids — reuse/extend, don't duplicate.

**Files**: `experiments/ablation/metrics.py` (new)

**Commands**: `python experiments/ablation/metrics.py --selftest`

**Success criteria**: Self-test with synthetic grids where answers are known analytically:
uniform grid (max entropy, zero lift), delta-on-truth grid (nll→0, exact_rate=1), prior-as-grid
(grid_nll == grid_nll_prior exactly, lift==1.0). All asserts pass.

**Validation**: `python experiments/ablation/metrics.py --selftest` prints `SELFTEST PASS`

⚠️ Risk: ECE on 3-way outcomes has binning subtleties (use max-prob bin convention); document
convention in the docstring so future rows are comparable.

### After completing this step:
- [ ] Run validation, report result
- [ ] Update work_log.md
- [ ] Commit: `git add experiments/ablation/metrics.py && git commit -m "feat: Step 2 - ablation metric suite with analytic self-test"`
- [ ] Summary + Next: Step 3 builds splits.py
- [ ] STOP / proceed

---

## Step 3: Split/eval-set module

**Goal**: `experiments/ablation/splits.py` — canonical masks (train/val/test as today) + the new
eval policy from DESIGN.md (e.g. earlystop-tail-of-train + pooled val∪test eval), natl mask,
and the WC-slate loader (from wc_cache.npz / lineups+results). One function returns named masks +
metadata (counts, date ranges) for a given npz.

**Files**: `experiments/ablation/splits.py` (new)

**Commands**: `python experiments/ablation/splits.py --report`

**Success criteria**: Printed table of each eval set: n, natl n, date range; leakage assertions pass
(earlystop set ⊂ train dates; eval sets disjoint from train; WC slate size == expected count from
results.json finished games).

**Validation**: `python experiments/ablation/splits.py --report` prints `LEAKAGE CHECKS PASS` and a
counts table; natl pooled eval n ≥ 350.

### After completing this step:
- [ ] Run validation, report result
- [ ] Update work_log.md
- [ ] Commit: `git add experiments/ablation/splits.py && git commit -m "feat: Step 3 - eval splits with leakage checks and pooled national lane"`
- [ ] Summary + Next: Step 4 builds the runner
- [ ] STOP / proceed

---

## Step 4: Experiment runner + smoke test

**Goal**: `experiments/ablation/run_ablation.py` — takes a config (JSON inline or file: name, npz,
ctx flags, beta, W, decay params [future], seeds, epochs, notes) → trains train-split seeds via
train_goals components → evaluates all lanes with metrics.py → appends one row to registry.jsonl
(config + git commit + data file mtimes/hash + metrics + wall time) → regenerates
RESULTS_ABLATION.md (sortable table, one row per run, baseline-delta columns).

**Files**: `experiments/ablation/run_ablation.py` (new), `experiments/ablation/registry.jsonl`
(created), `experiments/ablation/RESULTS_ABLATION.md` (generated)

**Commands**: smoke run: `python experiments/ablation/run_ablation.py --name smoke --seeds 1 --epochs 3`

**Success criteria**: Smoke run completes in <10 min, appends a valid JSONL row (parses, has all
schema fields), report regenerates with the row, rerunning with same name warns (no silent dupes).

**Validation**: `python -c "import json; rows=[json.loads(l) for l in open('experiments/ablation/registry.jsonl')]; r=rows[-1]; assert r['name']=='smoke' and 'grid_nll' in r['metrics']['test_all']; print('REGISTRY ok', len(rows))"`

⚠️ Risk: train_goals.main() is monolithic — import its pieces (GoalNet, data loading, exp_points)
rather than subprocessing; if refactor is needed, extract a `load_data()`/`train_one()` into the
runner WITHOUT modifying train_goals.py (production script stays byte-identical in Phase 1).

### After completing this step:
- [ ] Run validation, report result
- [ ] Update work_log.md
- [ ] Commit: `git add experiments/ablation/run_ablation.py experiments/ablation/registry.jsonl experiments/ablation/RESULTS_ABLATION.md && git commit -m "feat: Step 4 - one-command ablation runner with registry + generated report"`
- [ ] Summary + Next: Step 5a kicks off the real baseline
- [ ] STOP / proceed

---

## Step 5a: Kick off baseline runs (background)

**Goal**: Launch the production-recipe baseline under the harness: `--name baseline-beta3-w15
--seeds 5 --epochs 150` (matches production: β=3, W=15, imputed 68k npz). Long-running (~2-4h CPU).

**Files**: None (runner invocation; log to `data/_ablation_baseline.log`)

**Commands**: background Bash: `python experiments/ablation/run_ablation.py --name baseline-beta3-w15 --seeds 5 --epochs 150 > data/_ablation_baseline.log 2>&1` (run_in_background)

**Success criteria**: Process launched, log shows epoch progress within 5 min.

**Validation**: `tail data/_ablation_baseline.log` shows training output; process alive.

⚠️ Risk: multi-hour run — the harness background task should survive, but if the session ends,
relaunch is idempotent (registry warns on dupe name; use --force-rerun flag). Do NOT wait
synchronously; Step 5b verifies after completion notification.

### After completing this step:
- [ ] Run validation, report result
- [ ] Update work_log.md
- [ ] Commit: skipped — no file changes
- [ ] Summary + Next: Step 5b verifies baseline vs known numbers
- [ ] STOP / proceed (in autonomous mode, wait for completion notification, then 5b)

---

## Step 5b: Verify baseline reproduces known results + WC-slate row

**Goal**: Sanity-anchor the harness: canonical-split metrics from the baseline row must reproduce
the historical numbers within seed noise (TEST-all rps ≈ 0.2130-0.2145, pg ≈ 0.703-0.716; natl pg
in historical range). WC-slate lane scored from production goalnet.pt must match the known final
(compare with compare_models.py output). Any mismatch = harness bug — STOP and debug before
trusting any future row.

**Files**: None — verification (registry/report updated by the run itself)

**Commands**: `python -c` registry assertions + read RESULTS_ABLATION.md

**Success criteria**: Baseline row within tolerance bands above; WC-slate pts matches production's
known score on the full played slate.

**Validation**: scripted assertion on the registry row (tolerances as above), printed comparison
table old-vs-new.

### After completing this step:
- [ ] Run validation, report result
- [ ] Update work_log.md
- [ ] Commit: `git add experiments/ablation/registry.jsonl experiments/ablation/RESULTS_ABLATION.md && git commit -m "feat: Step 5 - harness-verified baseline rows (reproduces historical metrics)"`
- [ ] Summary + Next: Step 6 writes the prior-coasting diagnostic
- [ ] STOP / proceed

---

## Step 6: Prior-coasting diagnostic report

**Goal**: Answer the user's question with baseline numbers: (1) grid-NLL vs empirical-prior null —
how much score-level information does the model add? (2) per-scoreline lift table — precision when
predicting off-modal scores, top-3 recall per true scoreline; (3) calibration reliability (outcome
+ exact-score); (4) sharpness. Written as a section in RESULTS_ABLATION.md (generated from a
diagnostics function in the runner, `--diagnose <name>`), plus a plain-language conclusion.

**Files**: `experiments/ablation/run_ablation.py` (add --diagnose), `experiments/ablation/RESULTS_ABLATION.md`

**Commands**: `python experiments/ablation/run_ablation.py --diagnose baseline-beta3-w15`

**Success criteria**: Report section exists with all four diagnostics + a stated verdict (e.g.
"model adds X nats over prior; off-modal precision Y vs prior Z").

**Validation**: grep RESULTS_ABLATION.md for the diagnostics section header + lift table present.

### After completing this step:
- [ ] Run validation, report result
- [ ] Update work_log.md
- [ ] Commit: `git add experiments/ablation/run_ablation.py experiments/ablation/RESULTS_ABLATION.md && git commit -m "feat: Step 6 - prior-coasting diagnostics (grid-NLL vs null, lift, calibration)"`
- [ ] Summary + Next: Step 7 hands off to Phase 2
- [ ] STOP / proceed

---

## Step 7: Prepare Next Phase Handoff

**Goal**: Write `plans/goalnet-ablation-phase-1-to-2-handoff.md` (completed steps, validation
results, commits, frozen contracts: registry schema + metric suite + split policy + baseline row
values; diagnostic verdict; updated Phase-2 skeleton with anything learned). Update
`plans/goalnet-ablation-phase-state.md` (phase 1 → COMPLETE, current phase 2). Update auto-memory
roadmap file with harness location + baseline verdict (context-clear-proof). Per PHASE_MODE:
print canonical handoff prompt and STOP (pause-between-phases) or expand Phase 2 plan and continue
(all-phases-autonomous).

**Files**: `plans/goalnet-ablation-phase-1-to-2-handoff.md` (new),
`plans/goalnet-ablation-phase-state.md`, memory file.

**Commands**: None

**Success criteria**: Handoff file complete per skill checklist; phase state current; a fresh
session could resume Phase 2 from files alone.

**Validation**: checklist scan of handoff file (all required sections present).

### After completing this step:
- [ ] Run validation, report result
- [ ] Update work_log.md
- [ ] Commit: `git add plans/goalnet-ablation-phase-1-to-2-handoff.md plans/goalnet-ablation-phase-state.md && git commit -m "docs: Step 7 - phase 1→2 handoff"`
- [ ] Summary + Next: Phase 2 (plan it first)
- [ ] STOP at phase boundary (Hard Rule #9) unless all-phases-autonomous
