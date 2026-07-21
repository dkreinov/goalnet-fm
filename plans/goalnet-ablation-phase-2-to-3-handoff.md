# Phase 2 → Phase 3 handoff — GoalNet ablation program

**Phase 2 (Loss-level levers: β purity, time-decay, national-W recheck): COMPLETE (2026-07-22).**
A fresh session starts Phase 3 from this file + `plans/goalnet-ablation-phase-3-context-features-plan.md`
+ `experiments/ablation/DESIGN.md`.

## Headline result

The production recipe's two points-oriented knobs — the **β=3 EV-points decision term** and the
**W=15 national upweight** — were both hurting the calibrated-scoreline objective. Stripping them
(**β=0, W=1**) is a robust win on every lane and both splits, and it even keeps/improves the reference
point-scoring. **Adopted core-training config for all Phase-3+ experiments: `--beta 0 --w 1`.**

## Runs + verdicts (pooled eval_natl grid_info vs baseline `baseline-beta3-w15` = +0.1268)

| run | natl gInfo (Δ) | wc gInfo | all gInfo | exact_lift(natl) | wc pg | gate/verdict |
|---|---|---|---|---|---|---|
| baseline β3 W15 | +0.1268 | +0.1463 | −0.0317 | 1.31 | 0.923 | reference |
| beta0-w15 | +0.2522 (+0.125) | +0.2811 | +0.0723 | 1.22 | 0.913 | ADOPT (β) |
| beta1-w15 | +0.2435 (+0.117) | +0.2853 | +0.0550 | 1.31 | 0.904 | pass (not chosen) |
| beta3-w1 | +0.2126 (+0.086) | +0.2701 | +0.0053 | — | 0.913 | ADOPT (W) |
| beta3-w40 | +0.1725 (+0.046) | +0.1750 | −0.0041 | — | 0.865 | REJECT (wc pg −0.058) |
| decay-hl2 | +0.1925 (+0.066) | +0.2026 | +0.0077 | — | 0.933 | pass → deferred |
| decay-hl4 | +0.1714 (+0.045) | +0.1768 | +0.0103 | — | 0.904 | pass → deferred |
| decay-hl8 | +0.2072 (+0.080) | +0.2061 | +0.0195 | — | 0.894 | pass → deferred |
| **combo-beta0-w1 (ADOPTED)** | **+0.2432 (+0.116)** | **+0.2992** | **+0.0762** | **1.33** | **0.933** | **ADOPTED** |
| combo-beta0-w1-decay8 | +0.2477 (+0.121) | +0.3072 | +0.0779 | 1.31 | 0.913 | decay not additive |
| combo-beta0-w1-s10 (10-seed) | +0.2414 (+0.115) | +0.2937 | +0.0764 | — | 0.933 | robustness ✓ |
| combo-beta0-w1-canon | (canonical) | — | — | — | 0.971 | canonical ✓ |

Canonical (continuity): `combo-beta0-w1-canon` beats `baseline-beta3-w15-canonical` on every canonical
lane — canonical_test_natl grid_info +0.297 vs +0.234 (rps 0.169 vs 0.175), canonical_test_all +0.073
vs −0.000 (rps 0.209 vs 0.211), wc pg 0.971 vs 0.942.

## Verdict details

- **β=0 (ADOPTED)**: strongest lever (+0.125 natl). β=3's decision term is a points-bias; removing it
  adds score-level information on all lanes and flips the club lane positive. Cost = raw exact_lift
  (1.31→1.22) which the combo then recovers.
- **W=1 (ADOPTED)**: the 15× national upweight is the same kind of points-bias; W=1 beats W=15
  (+0.086 natl), W=40 is worse and fails. Removing it also drops an arbitrary hyperparameter and
  (in combo) restores exact_lift to 1.33.
- **Combo β0+W1 (ADOPTED)**: best-rounded — best wc/all grid_info, best exact_lift, best wc pg. The
  levers don't stack additively (β0 already captures most of the natl gain; combo ≈ β0 within noise)
  but the union is the cleanest model. 10-seed-robust; wins both splits.
- **Time-decay (DEFERRED)**: hl∈{2,4,8} each beat the β3/W15 baseline standalone but non-monotonically
  (half-life is noise-level), and adding hl8 on top of β0+W1 gives nothing (Δnatl +0.004, wc pg −0.019).
  Its gain was **subsumed** by removing the biases → recency is already handled.

## Frozen adopted config (for Phase 3+)

`run_ablation.py ... --beta 0 --w 1` is the new experiment baseline. The reference row Phase-3 diffs
against is **`combo-beta0-w1`** (pooled), NOT the old `baseline-beta3-w15`. Everything else in the
harness contract (metric suite, splits, lanes, registry schema, wc_inputs.npz, per-seed resume) is
unchanged from Phase 1. Full adopted-defaults rationale: `experiments/ablation/DESIGN.md` →
"Phase-2 adopted defaults" section.

## Implications for Phase 3 (context features)

- **Recency is already captured** by removing the biases (decay was subsumed). Phase-3
  Elo-momentum/trajectory features must add **trajectory** signal (form direction, windowed Elo
  deltas) BEYOND simple recency/time-decay — otherwise they'll be redundant. Gate them on eval_natl
  grid_info improvement over `combo-beta0-w1`, not over the old baseline.
- Phase-3 features enter via `--ctx-extra <file.npz>` (wired in Phase 1, inert until now). Build
  feature bundles as `data/<feat>.npz` with a `mids` key; `_load_extra` in run_ablation.py picks the
  first non-`mids` array (or a `feats`/`val`/`ctx` key) and concatenates to context.
- Every Phase-3 run must pass `--beta 0 --w 1` (the adopted config) so feature effects aren't confounded
  by the old biases.

## Ops notes (unchanged from Phase 1, reconfirmed)

- Interpreter: `C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe` (bare python not
  on Bash PATH). NumPy2/torch import banner is benign.
- 4 logical cores, ~4 GB free RAM → **one training run at a time** (concurrency OOMs/thrashes).
- **Background jobs are idle-killed inside the Claude session.** Reliable pattern used in Phase 2:
  a **Windows scheduled task** running a `.ps1` that invokes the sequential runs
  (`experiments/ablation/run_phase2_*.ps1` are templates) — fully detached, survives session idle.
  Per-seed caches (`rates/<name>.s<k>.npz`) make every run resumable; a killed/relaunched run resumes.
- ~55 min per 5-seed/150-epoch pooled run (early-stop governs, not the 150 cap).

## Resume pointer

Phase state: phase 2 → COMPLETE, current phase = 3. Next action: execute
`plans/goalnet-ablation-phase-3-context-features-plan.md` (already drafted) — but FIRST reconfirm the
adopted `--beta 0 --w 1` baseline and re-read the "Implications for Phase 3" above (momentum must beat
recency, not just add it).
