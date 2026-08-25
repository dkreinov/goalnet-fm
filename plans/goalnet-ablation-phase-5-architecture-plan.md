# Phase 5 — Architecture: cross-team attention + plus-minus ratings

**Program:** GoalNet post-WC2026 ablation (`plans/goalnet-ablation-phase-state.md`).
**Consumes:** `plans/goalnet-ablation-phase-4-to-5-handoff.md` + `experiments/ablation/DESIGN.md`
(Phase-2/3/4 adopted sections). **Produces:** `plans/goalnet-ablation-phase-5-to-6-handoff.md`.

## Goal

Test whether model-architecture changes add score-level information beyond (a) the current
per-team-encoder GoalNet at `combo-beta0-w1`, and (b) the **odds-informed bar (+0.1803 covered
natl = ctx-odds feature + λ0.5 blend)**. Two arms from the program skeleton:
- **Arm X — cross-team attention**: a match-comparison block that lets the two XIs attend to each
  other (HIGFormer-style), instead of encoding each XI in isolation.
- **Arm P — plus-minus player ratings**: leakage-free on-pitch goal-difference ratings computed
  from our own DB (match_player 3.4M + match_event 1.1M + match_sub 710k rows), tested cheap-first
  as a ctx feature, then as per-player input channels (the true "architecture" version).

## Assumptions and open decisions (none require user input — recorded per plan skill)

- **Both arms run regardless of each other's outcome** (independent information sources).
- **Exploratory seeds=3, confirmation seeds=5** on anything promising (phase-4 handoff decision).
- **Experiment default stays `combo-beta0-w1`** (β=0, W=1, base 10-dim ctx, pooled split); winners
  additionally scored against the odds layer before adoption (handoff frozen rule).
- Cross-attention variant keeps the exact GoalNet input contract (Xh,Rh,Xa,Ra,C) → **WC-slate lane
  stays ON for Arm X** (unlike --ctx-extra runs). Arm P2 changes A (62→64) → wc lane OFF for it.
- Plus-minus = segment-level on-pitch GD/90 where events+subs exist, full-match fallback,
  net-of-team, empirical-Bayes shrinkage toward 0, strict as-of-date expanding window.
  RAPM-style ridge is explicitly OUT of scope (cost); noted as future work if P shows signal.
- Long runs = detached Windows scheduled tasks + hourly ScheduleWakeup (established ops pattern).
- **Subagent policy: NOT allowed** (user rule, applies all phases).
- Production `goalnet.pt` / `src/train_goals.py` untouched (Hard Rule until Phase 6). Variant
  classes live in NEW `experiments/ablation/models.py`; `run_ablation.py` gets `--arch`/`--pm-channel`.

**Execution mode status:** not chosen yet (ask before Step 1). PHASE_MODE: pause-between-phases.
**Recommended model:** Opus — established harness patterns; the one design-heavy piece (cross22
block) is small and contract-frozen here. **Recommended thinking budget:** High.
**Context recommendation:** fresh context fine either way; plan is self-contained.
**Branch:** master (repo's working branch; no `main`).

## Frozen contracts (respect throughout)

- Registry schema, RESULTS regeneration, metric suite, pooled split, covered-subset methodology —
  all per DESIGN.md. New runs append rows; never edit old rows.
- `--arch goalnet` (the default) must remain **bit-for-bit identical** to today's code path: the
  model factory must not add RNG draws before `tg.GoalNet(...)` construction (train_one seeds RNG
  then immediately builds the net — run_ablation.py:190-195).
- New npz naming per convention: `build_X.py` → `data/X.npz`.
- Interpreter: `%LOCALAPPDATA%\Programs\Python\Python312\python.exe` (bare
  `python` not on PATH). Lazy-NpzFile gotcha: ALWAYS materialize arrays before loops.

## Steps

### Step 1 — Infra: `models.py` + `--arch` flag + parity tripwire
- Create `experiments/ablation/models.py`:
  - `build_model(arch, A, nctx)` → `"goalnet"`: return `tg.GoalNet(A, nctx)` verbatim (no extra
    RNG); `"cross22"`: new class (Step 2 spec).
- `run_ablation.py`: add `--arch` (default `goalnet`), thread through `train_one`, record in
  `config.flags.arch` (only when != goalnet, keeping old rows comparable), include in run name
  conventions. wc_ok logic: unchanged for arch (input contract preserved); OFF when `--pm-channel`.
- **Validation:** 2-epoch smoke run `--arch goalnet` vs the same 2-epoch run on current HEAD code
  path → identical epoch-loss printout (bit-for-bit). Registry untouched by smoke runs.
- After: commit (`feat: Phase 5 Step 1 - arch flag + models.py + parity`); work_log entry.

### Step 2 — Arm X: `cross22` variant + exploratory run (seeds=3)
- `Cross22GoalNet` in models.py: shared player MLP + role embedding (same shapes as Encoder,
  d=64) + **team embedding (2,d)**; concatenate both XIs → 22 tokens; 2-layer
  TransformerEncoder(d=64, nhead=4, ff=128, batch_first) over all 22 (cross-team attention comes
  free); pool per team per role (masked mean, 4·d each side); per-team head `team`→`ad` producing
  [attack, defence] as in GoalNet; same rate equation + home_adv + ctx head. Param count ≈ GoalNet.
- Detached scheduled-task run: `arch-cross22-s3` (`--arch cross22 --beta 0 --w 1 --seeds 3`).
- **Validation:** registry row appended; compare eval_natl + wc_slate grid_info vs combo-beta0-w1
  (and its s3 noise band — use combo-beta0-w1 per-seed spread to judge).
- After: commit; work_log with verdict-so-far.

### Step 3 — Arm X decision gate (confirm or fallback, seeds per outcome)
- If `arch-cross22-s3` ≥ combo-beta0-w1 on eval_natl grid_info (within/above noise): seeds=5
  confirmation run `arch-cross22`. If clearly below: ONE fallback exploratory variant
  `arch-latecross-s3` (keep per-team Encoder, add a single cross-attention block between the two
  pooled token sets before the ad head) — then stop Arm X regardless of its result.
- **Validation:** registry rows; verdict recorded in work_log.
- After: commit.

### Step 4 — Arm P data: `src/build_plusminus.py` → `data/ctx_pm.npz` + `data/players_pm.npz`
- Chronological single pass over all DB matches (ORDER BY match_date, match_id): maintain
  per-player accumulators (on-pitch GD, minutes). Per match: reconstruct on-pitch segments from
  match_event goal minutes + match_sub in/out minutes for the 22+ participants (match_player
  minutes as fallback when events/subs missing → full-match GD share). Rating used for a match =
  accumulator state BEFORE that match (leakage-free), net-of-team (player GD minus team-average GD
  over the same span), shrunk: `pm * n90 / (n90 + K)` (K≈20 90s; sensitivity-check K=10/40 on
  earlystop only).
- Outputs: `data/ctx_pm.npz` {mids, feats:[pm_diff_teammean, pm_cov]} for ALL 69,053 npz matches;
  `data/players_pm.npz` {mids, PMh (N,11,2), PMa (N,11,2)} with channels [pm_shrunk, has_pm],
  slot-aligned by replicating build_player_dataset_imp.py's ordering (graded-first-then-role sort,
  lines 100-113).
- **Validation:** alignment check — rebuild pipeline's mids ⊇ npz mids AND per-slot roles
  reproduce Rh/Ra EXACTLY for every kept match (hard assert); spot-check 3 known players' ratings
  for sanity (e.g. a star attacker > 0, relegation-fodder < 0); zero future leakage by
  construction (assert accumulator update happens after feature emission).
- After: commit (`feat: Phase 5 Step 4 - plus-minus builder + aligned npz`).

### Step 5 — Arm P1 (cheap): plus-minus as ctx feature
- Run `ctx-pm-s3`: `--beta 0 --w 1 --ctx-extra ctx_pm.npz --seeds 3` (wc lane auto-skipped).
- **Validation:** registry row; eval_natl + eval_all grid_info vs combo-beta0-w1.
- After: commit; work_log verdict.

### Step 6 — Arm P2 (architecture): plus-minus as player channels
- `run_ablation.py --pm-channel players_pm.npz`: loader appends the 2 per-slot channels to
  Xh/Xa (A 62→64) after standardization of the base 62 (pm channels standardized on TRAIN);
  wc lane skipped; works with any `--arch`.
- Run `pm-channel-s3` on the better of {goalnet, cross22-if-adopted}. seeds=3.
- **Validation:** registry row; compare vs same-arch baseline AND vs ctx-pm-s3 (does player-level
  beat team-aggregate?).
- After: commit.

### Step 7 — Confirmation + odds-bar scoring of any winner
- Anything promising at s3 → seeds=5 confirmation run.
- Every s5 winner then scored against the market: rerun `--ctx-extra ctx_odds.npz` variant AND/OR
  apply the B1 λ-blend (blend_market.py pattern) to the winner's cached rates; evaluate on the
  covered natl subset. **Adoption bar: beat +0.1803 covered natl grid_info** (or beat the arch-less
  odds config on identical matches). No winner → record null cleanly (a Phase-3-style null is a
  valid, useful result).
- **Validation:** covered-subset table in work_log; registry rows for all runs.
- After: commit.

### Step 8 — Verdicts + docs
- Append "Phase-5 adopted architecture" section to DESIGN.md (full verdict table, adopted config
  or explicit null); regen RESULTS_ABLATION.md; RESULTS_WC2026.md narrative ONLY if adopted.
- **Validation:** `--report` clean; registry row count consistent; git status clean after commit.
- After: commit (`feat: Phase 5 Step 8 - verdicts + adopted architecture docs`).

### Step 9 — Prepare Next Phase Handoff
- Write `plans/goalnet-ablation-phase-5-to-6-handoff.md` (runs+verdicts, adopted config, frozen
  contracts, Phase-6 replay skeleton updated with arch outcome + the still-open multi-source odds
  TODO); update phase-state (phase 5 → COMPLETE, current = 6); memory roadmap paragraph
  (`fm-modeling-roadmap.md`); work_log final entry; delete any scheduled tasks; print canonical
  handoff prompt; STOP (PHASE_MODE pause-between-phases).

## Risks

- **RNG parity break** in Step 1 (silent — would invalidate cross-run comparability): guarded by
  the bit-for-bit smoke test before anything else.
- **Slot misalignment** in players_pm.npz (silent feature garbage): guarded by the exact Rh/Ra
  reproduction assert.
- **Plus-minus confounding** (player rating ≈ team strength, which Elo ctx already has):
  net-of-team + shrinkage mitigate; the ctx-pm cheap arm measures exactly this redundancy before
  the expensive channel version.
- **cross22 undertrained on CPU** (22 tokens ≈ 2× attention cost): seeds=3 exploratory keeps cost
  ~35-45 min/run; d stays 64; no deeper sweeps unless s3 is promising.
- **DB drift**: DB may contain matches added after players_imp.npz was built — builder reindexes
  to the npz's mids, never regenerates players_imp.npz.

## Handoff prompt (for a fresh session executing this plan)

```text
--- HANDOFF PROMPT (paste into fresh session) ---
Continue plan from: plans/goalnet-ablation-phase-5-architecture-plan.md
Read first: that plan + plans/goalnet-ablation-phase-4-to-5-handoff.md + experiments/ablation/DESIGN.md
Resume at: Step 1 (arch flag + models.py + parity)
Execution mode: <as chosen>   PHASE_MODE: pause-between-phases
Model: Opus, thinking High
Context: OK to continue in this session if plan just written; else /clear first
Before executing:
1. Read the files above fully. 2. Check git status. 3. Reconfirm frozen contracts.
Important: subagents NOT allowed; long runs = detached scheduled tasks + hourly ScheduleWakeup;
full interpreter path (bare python not on PATH); plans/ needs git add -f; goalnet.pt untouched.
--- END HANDOFF PROMPT ---
```
