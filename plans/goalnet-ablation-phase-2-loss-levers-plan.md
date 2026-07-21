# Phase 2 — Loss-level levers: β purity, time-decay, national-W recheck

**Consumes:** `plans/goalnet-ablation-phase-1-to-2-handoff.md` (read first — baseline rows, frozen
contracts, ops notes) + `experiments/ablation/DESIGN.md`. Harness is DONE and proven faithful;
Phase 2 writes **no new harness code** except the tiny W plumbing check in Step 0 — every experiment
is a `run_ablation.py` invocation.

**Goal:** Decide, with registry evidence, three questions about the core training recipe:
1. **β purity** — is the decision-focused term (β=3, EV-points softmax) a useful regularizer for the
   calibrated-scoreline objective, or a points-bias that hurts it? (Motivation: baseline
   eval_all grid_info = **−0.032** — below the empirical prior on club-heavy games.)
2. **Time-decay** — do exponential sample weights (half-life ~2/4/8y) beat equal-weight history?
3. **National W** — under the new distributional metrics, is W=15 still right, or does it trade too
   much club calibration for the national lane?

**Reference row (never rerun):** `baseline-beta3-w15` (pooled) — eval_natl grid_info +0.1268 /
rps 0.1817 / exact_lift 1.31 / pg 0.816; wc_slate grid_info +0.1463 / pg 0.923; eval_all grid_info
−0.0317. All Phase-2 runs use `--split pooled --seeds 5 --epochs 150` (comparability with baseline).

**Adopt/reject gate (frozen for this phase):** ADOPT a lever only if pooled **eval_natl grid_nll
improves (grid_info ≥ baseline +0.02 nats)** OR (eval_natl grid_info ≥ baseline − 0.01 AND eval_all
grid_info improves ≥ +0.03) — **AND** wc_slate shows no regression beyond noise (grid_info ≥
baseline − 0.02 and pts_g_31 ≥ baseline − 0.05). `pts_g_31` is NEVER a gate, reference only.
Seed-noise yardstick: baseline per-seed earlystop rps spread ≈ 0.0018; treat |Δgrid_info| < 0.02 as
noise-level unless consistent across lanes.

**Execution mode:** autonomous. **PHASE_MODE:** pause-between-phases. **Model:** any (runs are
mechanical; analysis needs care). **Subagents:** NOT allowed; long runs = background Bash.

**Ops (from Phase 1, non-negotiable):**
- Interpreter: `C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe` (bare `python`
  not on Bash PATH). NumPy2/torch import banner is benign noise — filter it, don't chase it.
- Each 5-seed/150-epoch pooled run ≈ **55–60 min CPU**. Run ONE experiment at a time.
- **Background jobs get idle-killed.** Keep the session active with monitoring loops
  (≤9.5-min timeouts) while a run is live. If killed anyway: relaunch the SAME `--name` — per-seed
  caches `experiments/ablation/rates/<name>.s<k>.npz` resume automatically (registry row only
  appears at full completion, so the dupe-check won't block resumes).
- Total budget: 6 sweep runs + up to 1 combo run ≈ **6–7 h**. Steps are ordered so an interrupted
  phase still leaves complete verdicts for whatever finished.

**Assumptions (stated, not asked):**
- β sweep at {0, 1} only — β=3 IS the baseline row; no rerun. If β=0 and β=1 straddle a sweet spot,
  a follow-up β=0.5 is allowed in Step 4 (counts against the combo-run budget).
- Decay sweep at β=3/W=15 (one lever at a time vs baseline). Combos only in Step 4 from WINNERS.
- W recheck at {1, 40} (bracket W=15 from both sides; W=1 = no upweight is the honest null).
  If both bracket ends lose, W=15 stands without testing intermediate values.
- No new data, no scraping, no changes to `src/train_goals.py` or production `goalnet.pt` in this
  phase. Adopted levers become the *harness default config for Phase 3+ experiments* — production
  retrain decisions belong to Phase 6.

---

## Step 0: Preflight — verify decay + W plumbing (~10 min)

**Goal**: `--decay-halflife` and `--w` were wired in Phase 1 but decay has NEVER run. Two 1-seed
3-epoch smoke runs to prove the flags reach the loss, then delete the throwaway rows.

**Commands**:
- `... run_ablation.py --name smoke-decay --seeds 1 --epochs 3 --decay-halflife 4`
- `... run_ablation.py --name smoke-w1 --seeds 1 --epochs 3 --w 1`
- Verify: registry rows carry `"decay_halflife": 4.0` / `"W": 1.0`; runs exit 0.
- Cleanup: remove both rows from registry.jsonl (they're the only permitted edit — throwaway smoke,
  same convention as Phase 1 Step 4), delete their `rates/` caches, `--report` to regenerate.

**Validation**: both smoke runs exit 0 AND a scripted check that decay actually changed training —
easiest: assert the two smoke rows' `seed_earlystop_rps` differ from each other and registry ends
clean (2 baseline rows only) after cleanup.

### After: work_log; no commit (no tracked-file changes after cleanup); proceed.

---

## Step 1: β sweep — the purity question (2 runs, ~2 h)

**Goal**: Quantify what the decision-focused term does to the calibrated-scoreline objective.

**Commands** (sequential, background Bash + active monitoring, logs `data/_ablation_beta<k>.log`):
- `... run_ablation.py --name beta0-w15 --beta 0 --notes "pure Poisson core; purity test"`
- `... run_ablation.py --name beta1-w15 --beta 1 --notes "small decision term; regularizer test"`

**Success criteria**: two registry rows, all lanes populated.

**Read-out (record in work_log)**: Δgrid_info / Δgrid_nll / Δece / Δsharpness on eval_all AND
eval_natl AND wc_slate vs baseline; exact_lift and pts_g_31 as reference. Explicit statement:
does β=0 fix the negative eval_all grid_info? What does it cost on eval_natl/wc_slate?

**Validation**: scripted registry assertion — both rows present, `config.beta` ∈ {0.0, 1.0},
`metrics.eval_all.grid_nll` present; print a Δ-vs-baseline table.

### After: work_log; commit `git add experiments/ablation/registry.jsonl experiments/ablation/RESULTS_ABLATION.md && git commit -m "feat: Phase 2 Step 1 - beta purity sweep (beta 0/1 vs baseline 3)"`; proceed.

---

## Step 2: Time-decay sweep (3 runs, ~3 h)

**Goal**: Exponential sample-weight decay `0.5^(age_years/hl)` at hl ∈ {2, 4, 8} vs equal-weight
baseline (β=3, W=15 held).

**Commands**:
- `... run_ablation.py --name decay-hl2 --decay-halflife 2 --notes "aggressive recency"`
- `... run_ablation.py --name decay-hl4 --decay-halflife 4 --notes "moderate recency"`
- `... run_ablation.py --name decay-hl8 --decay-halflife 8 --notes "gentle recency"`

**Read-out**: monotonicity across hl (a clean trend = real signal; a zig-zag = noise → reject);
same Δ table vs baseline. Note dataset spans 2020→2026 (max age ≈ 6.5y at train cutoff 2024-08:
hl=2 → oldest ≈ 0.21×; hl=8 → oldest ≈ 0.67×).

**⚠️ Risk**: decay shrinks the *effective* n of the 42.6k train set; watch per-seed earlystop rps
spread — if hl=2 spread balloons (>2× baseline's 0.0018), flag as unstable in the verdict.

**Validation**: scripted registry assertion — 3 rows, `config.decay_halflife` ∈ {2,4,8}; Δ table.

### After: work_log; commit (same file list) `"feat: Phase 2 Step 2 - time-decay half-life sweep"`; proceed.

---

## Step 3: National-W recheck (2 runs, ~2 h)

**Goal**: Bracket W=15 under distributional metrics: W=1 (no upweight — does the national edge
survive at all without it?) and W=40 (does more upweight buy more national grid_info, and what does
it cost eval_all?).

**Commands**:
- `... run_ablation.py --name beta3-w1 --w 1 --notes "no national upweight; null test"`
- `... run_ablation.py --name beta3-w40 --w 40 --notes "heavy national upweight"`

**Read-out**: eval_natl grid_info vs eval_all grid_info trade-off curve across W ∈ {1, 15, 40};
wc_slate as tiebreak. Explicitly answer: is eval_all's negative grid_info CAUSED by W (W=1 row
should show it) or by β (Step 1 answers that) or both?

**Validation**: scripted registry assertion — 2 rows, `config.W` ∈ {1.0, 40.0}; Δ table.

### After: work_log; commit `"feat: Phase 2 Step 3 - national weight recheck (W 1/40 vs 15)"`; proceed.

---

## Step 4: Verdicts + (conditional) combo run (~0–2 h)

**Goal**: Apply the frozen gate to every lever; write adopt/reject verdicts. IF ≥2 independent
levers pass the gate, run ONE combo (`--name combo-<levers>` with the winning settings) to check
they stack (interactions are real: decay reweights recent national-heavy years, so decay×W overlap).
IF the β sweep suggests a sweet spot between tested values, one β=0.5 run allowed INSTEAD of combo.
Diagnose the best run: `... run_ablation.py --diagnose <winner>` (does the winner fix eval_all
coasting? does off-modal precision improve?).

**Success criteria**: work_log verdict block per lever: ADOPT/REJECT + one-line numeric reason.
`RESULTS_ABLATION.md` regenerated; diagnostics section for the best candidate present.

**Validation**: scripted check — every Phase-2 registry row has a verdict line in work_log;
report contains diagnostics for the chosen winner (or baseline if nothing won).

### After: work_log; commit `"feat: Phase 2 Step 4 - lever verdicts + combo/diagnostics"`; proceed.

---

## Step 5: Adopted-config decision + narrative (~20 min)

**Goal**: Freeze the **Phase-3 default config** (the `--beta/--w/--decay-halflife` values all
Phase-3+ experiments will inherit): baseline values for rejected levers, winner values for adopted
ones. Record it in DESIGN.md under a new "## Phase-2 adopted defaults" section (this is an ALLOWED
DESIGN.md append — contracts above it stay frozen). Per repo convention, append a short narrative
to RESULTS_WC2026.md ONLY for adopted changes (skip entirely if everything is rejected).

**Validation**: DESIGN.md contains the new section with explicit values; grep passes.

### After: work_log; commit `git add experiments/ablation/DESIGN.md [RESULTS_WC2026.md] && git commit -m "docs: Phase 2 Step 5 - adopted defaults"`; proceed.

---

## Step 6: Prepare Next Phase Handoff

**Goal**: Write `plans/goalnet-ablation-phase-2-to-3-handoff.md` (runs table with key metrics,
verdicts + reasons, adopted default config, anything learned that changes Phase 3 — e.g. if decay
is adopted, Phase-3 Elo-momentum features may be partly redundant with it; say so). Update
`plans/goalnet-ablation-phase-state.md` (phase 2 → COMPLETE, current 3). Update auto-memory
`fm-modeling-roadmap.md` with the Phase-2 verdict line. **plans/ is gitignored — new plan files
need `git add -f`.**

**Validation**: handoff checklist scan (sections: runs+verdicts, adopted config, Phase-3 skeleton
update, ops notes, resume pointer); phase-state shows phase 3.

### After: work_log; commit `git add -f plans/goalnet-ablation-phase-2-to-3-handoff.md && git add plans/goalnet-ablation-phase-state.md work_log.md && git commit -m "docs: Phase 2 Step 6 - phase 2->3 handoff"`; **STOP at phase boundary (pause-between-phases)**.
