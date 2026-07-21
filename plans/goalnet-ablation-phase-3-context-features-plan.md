# Phase 3 — Context features: Elo momentum/trajectory + stage/rest

**Consumes:** `plans/goalnet-ablation-phase-2-to-3-handoff.md` (read first — adopted config + why decay
was subsumed) + `experiments/ablation/DESIGN.md`. **Produces for Phase 4:** versioned context feature
bundle(s), registry rows, adopt/reject verdicts, handoff.

**Goal:** Add *trajectory* and *situational* context the current 10-feature `context.npz` misses, and
measure each with the harness. The bar is high and specific: Phase 2 showed **recency is already
captured** (time-decay was subsumed by β0+W1), so a momentum feature that only encodes "recent form
level" will be redundant. Phase 3 features must add **direction/change** (is a team rising or falling?)
and **situation** (knockout vs group, rest) — signal orthogonal to level + recency.

**Adopted baseline (all diffs are vs this):** `combo-beta0-w1` (config `--beta 0 --w 1`, pooled).
Every Phase-3 run passes `--beta 0 --w 1` so feature effects aren't confounded by the old biases.
Reference numbers: eval_natl grid_info +0.2432, wc_slate +0.2992, eval_all +0.0762, exact_lift 1.33.

**Gate (frozen):** ADOPT a feature only if pooled **eval_natl grid_info ≥ combo-beta0-w1 +0.02 nats**
(the Phase-2 seed-noise yardstick) AND no wc_slate regression (grid_info ≥ −0.02, pts_g_31 ≥ −0.05).
Report canonical + eval_all too, but national is the gate. `pts_g_31` reference-only.

**Execution mode:** autonomous, PHASE_MODE pause-between-phases. **Subagents:** not allowed.
**Ops (unchanged):** full interpreter path; ONE run at a time (4 cores/4GB); long runs as **detached
Windows scheduled tasks** (`experiments/ablation/run_phase2_*.ps1` are templates — copy for Phase 3);
per-seed caches resume; ~55 min/5-seed pooled run. Feature bundles enter via `--ctx-extra <file.npz>`
(a `mids` key + one feature array; `_load_extra` concatenates to context, standardised on train).

**Assumptions (stated):** context.npz already holds Elo level, mean form (last 5), goal-diff form,
rest-days (10 feats total — VERIFY exact contents in Step 0 before building, to avoid duplicating).
No new scraping unless Step 1 finds a free already-cached source. `src/build_context.py` gets a v2 that
EMITS A SEPARATE bundle (back-compatible; does not mutate context.npz) so `--ctx-extra` composition is
clean and the base model is untouched.

---

## Step 0: Re-confirm baseline + audit existing context (~15 min, no training)

**Goal**: Prevent redundant features. Read `src/build_context.py` and dump the exact 10 columns of
`data/context.npz` (names + a few example rows). Confirm which of {Elo level, form level, goal-diff
form, rest-days} already exist and at what window. Re-confirm `combo-beta0-w1` is in the registry as
the reference row.

**Files**: read-only `src/build_context.py`, `data/context.npz`; note findings in work_log.

**Validation**: work_log lists the 10 existing context columns and states what NEW signal each Phase-3
feature must add beyond them (the anti-redundancy contract).

### After: work_log; no commit; proceed.

---

## Step 1: Stage/round + rest-days backfill investigation (~30–60 min, no training)

**Goal**: Determine what situational context is BACKFILLABLE for the 69k training matches (not just
WC). Investigate, in order: (a) `match` table columns (is there a round/stage/competition_stage
field?); (b) `D:\Programming\claude\worldcup\team_db` stage data (WC only); (c) cached ESPN summaries
if present; (d) rest-days already in context.npz (verify coverage). Decide scope = what actually
exists for a usable fraction of matches. Knockout-shrinkage is the hypothesis (tighter, lower-scoring
knockouts) — only testable if stage backfills.

**Files**: read-only DB schema probe, `worldcup/team_db`, existing context builder; findings → work_log.

**Success criteria**: a written table of candidate situational features × coverage%, with a
build/skip decision each (scope to what's backfillable — DESIGN principle from the phase state).

**Validation**: work_log has the coverage table + explicit build-list for Steps 4–5.

### After: work_log; no commit; proceed.

---

## Step 2: Build Elo-momentum/trajectory bundle (`data/ctx_momentum.npz`)

**Goal**: Extend the Elo/form machinery in `build_context.py` (v2, emits a SEPARATE npz) with
leakage-free **trajectory** features computed chronologically over all matches: candidate set —
windowed Elo delta (Elo now − Elo N matches ago, N∈{5,10}), form slope (linear trend of last-5
result points, not mean), Elo "acceleration" (delta of deltas), unbeaten/winless streak length. Emit
`data/ctx_momentum.npz` = {`mids`, `feats` (n × k)}. STRICT no-leakage: every value uses only matches
strictly BEFORE the current one (same discipline as the existing Elo build).

**Files**: `src/build_context.py` (add a `--momentum` emit path OR a new `src/build_momentum.py`),
`data/ctx_momentum.npz` (generated).

**Commands**: `python src/build_momentum.py` (or the v2 flag).

**Success criteria**: npz built; a leakage self-check (feature for match i uses only dates < date_i);
feature distributions sane (deltas centered ~0, streaks non-negative ints).

**Validation**: scripted assertion — bundle has `mids` aligning to players_imp; leakage check passes;
print feature means/stds.

⚠️ Risk: momentum correlates with Elo level; standardisation + the ablation will reveal if it adds
*independent* signal. If eval_natl gain < +0.02, it's redundant with level+recency → reject (expected
possibility given Phase-2's recency finding).

### After: work_log; commit `src/build_momentum.py` + note (npz is data, gitignore-check first); proceed.

---

## Step 3: Ablate momentum (1 run + optional feature-subset run, ~1–2 h)

**Goal**: `run_ablation.py --name ctx-momentum --beta 0 --w 1 --ctx-extra ctx_momentum.npz` vs
`combo-beta0-w1`. If the full bundle passes but is marginal, one follow-up with the single strongest
sub-feature to avoid noise from weak columns.

**Commands** (detached scheduled task): `... --name ctx-momentum --beta 0 --w 1 --seeds 5 --epochs 150 --ctx-extra ctx_momentum.npz`

**Success criteria**: registry row; Δ-vs-combo table on all lanes.

**Validation**: registry assertion (row present, ctx_extra recorded); gate applied in work_log.

### After: work_log; commit registry + report; proceed.

---

## Step 4: Build + ablate stage/rest feature (conditional on Step 1, ~1–2 h)

**Goal**: IF Step 1 found backfillable stage/rest: build `data/ctx_stage.npz` (stage one-hot / rest-days
delta / knockout flag for the backfillable subset + missingness mask, same npz shape) and ablate
`--name ctx-stage --beta 0 --w 1 --ctx-extra ctx_stage.npz`. Test knockout-shrinkage as a
stage-conditioned effect. IF not backfillable beyond WC: SKIP with a logged reason (scope to reality).

**Commands**: build script + one detached run.

**Validation**: registry row (if built) or a work_log skip-rationale; gate applied.

### After: work_log; commit; proceed.

---

## Step 5: Combine winners + diagnose (~0–1 h)

**Goal**: IF ≥2 features pass, one combined run (`--ctx-extra momentum.npz stage.npz`) to check they
stack. Diagnose the best config (`--diagnose <winner>`). Write adopt/reject verdicts per feature vs the
gate.

**Validation**: work_log verdict block per feature; report has diagnostics for the winner.

### After: work_log; commit registry + report; proceed.

---

## Step 6: Adopted context + handoff

**Goal**: Freeze the Phase-3 adopted context config (which `--ctx-extra` bundles, if any, join the
β0+W1 default) in DESIGN.md ("## Phase-3 adopted context" section). Write
`plans/goalnet-ablation-phase-3-to-4-handoff.md` (feature table + verdicts + adopted config + what
Phase 4 (market anchor) should know). Update phase-state (phase 3 → COMPLETE, current 4) + memory.
If NOTHING passes (plausible — recency already captured), that is a valid, documented verdict: context
features beyond level+recency don't help at this data scale; Phase 4 (market anchor) is the next lever.

**Validation**: handoff checklist; phase-state shows phase 4.

### After: work_log; commit (`git add -f` the handoff); **STOP at phase boundary (pause-between-phases)**.

---

## Notes carried from Phase 2
- Momentum must beat RECENCY, not just add it (time-decay was subsumed by β0+W1). A null result here is
  informative, not a failure.
- Detached-scheduled-task pattern is the reliable way to run the sweeps unattended; hourly self-check
  via ScheduleWakeup worked well.
- Keep production `goalnet.pt` and `src/train_goals.py` untouched; adopted context is harness-only until
  the Phase-6 production retrain.
