# Work Log

## Task: Standing-aware win-optimization for auto_bet
Started: 2026-07-09
Status: In Progress

---

## Step 1: Premise-verify the verdict (read-only)
- Status: ✅ Complete
- Summary: Confirmed effective verdict = CHASE (rank 4/10, gap 20 to leader, ~5-6 behind RIVAL_1/RIVAL_2). spain_in=True. Gate PASS.
- Files changed: none — read only
- Git commit: skipped — read-only
- Timestamp: 2026-07-09

## Step 2: Add effective_verdict() to auto_bet
- Status: ✅ Complete
- Summary: Added MULT/FUTURES/P_WIN constants + effective_verdict() (reuses bearer, no 2nd refresh). main() fetches picks+profiles, logs verdict. Dry run: verdict=CHASE spain_in=True (matches Step 1).
- Deviations: inlined constants instead of importing decide_risk — importing read_standings would be circular (it imports auto_bet). P_WIN/FUTURES kept in sync via comment.
- Files changed: auto_bet.py
- Git commit: pending
- Timestamp: 2026-07-09

## Step 3: Add win_flag() + thread into predict/model_pick/main
- Status: ✅ Complete
- Summary: win_flag(round,verdict,spain_in) → strategy argv. predict() now takes flag=. Threaded into main loop + model_pick (default CHASE). All 8 matrix cases pass; contrarian flag yields valid pick.
- Files changed: auto_bet.py
- Git commit: pending
- Timestamp: 2026-07-09

## Step 4: SF/Final notify + approval window
- Status: ✅ Complete
- Summary: notify_bigpick() pushes ntfy+toast on verified SF/Final write (pick + strategy reason, "override before KO"). Wired into main write-verify block. Test push fired OK.
- Files changed: auto_bet.py
- Git commit: pending
- Timestamp: 2026-07-09

## Step 5: Live verify
- Status: ✅ Complete
- Summary: Live run + scheduled task both log verdict=CHASE spain_in=True, health ok, no AUTH FAILED. Win-optimization live (scheduled task auto-picked up working-dir file).
- Files changed: none (verify)
- Git commit: skipped — verify only
- Timestamp: 2026-07-09

---

## Final Summary
- Total steps: 5, Completed: 5, Failed: 0
- Key decisions: CHASE=gamble default + contrarian escalation on SF/Final (β0.35); SF/Final ntfy approval-window; Spain-out->gamble-everywhere failsafe; verdict recomputed every cycle from single bearer (no 2nd refresh).
- Deviations: inlined MULT/FUTURES/P_WIN (circular import via read_standings avoided).
- Status: Complete

# Work Log — GoalNet ablation program, Phase 1 (harness)

## Task: Reproducible ablation harness v2 + diagnostics + re-baseline
Started: 2026-07-21
Status: In Progress
Plan: plans/goalnet-ablation-phase-1-harness-plan.md (SESSION_MODE=autonomous, PHASE_MODE=pause-between-phases)

---

## Step 1: Inventory + freeze harness design (DESIGN.md)
- Status: ✅ Complete
- Summary: Read eval_harness.py (full), eval_seasons.py, dataset/context/wc_cache shapes. Froze design: split policy = option (b) pooled eval (early-stop on train tail, eval val∪test <2026-06-11, natl n≈400) + canonical kept for continuity; metric suite (grid_nll vs train-only empirical prior null, grid_info, ece, sharpness, exact_lift, pts_g_31 reference); registry JSONL schema; runner CLI.
- Deviations: (1) wc_cache.npz found STALE (90 games; results.json has all 104 finished with lineups) — plan assumed it was final; Step 3 rebuilds+freezes to 104. (2) Bash PATH has no `python` — full interpreter path recorded in DESIGN.md and used everywhere. (3) validation grep capitalization fixed ('Split policy').
- Files changed: experiments/ablation/DESIGN.md
- Git commit: (next line)
- Timestamp: 2026-07-21

## Step 2: Metrics module with self-test
- Status: ✅ Complete
- Summary: experiments/ablation/metrics.py — frozen suite (suite(), empirical_prior(), ece_outcome(), lift_table(), reliability()). Analytic self-test: uniform grid (max entropy, negative grid_info vs prior), delta-on-truth (nll→0, exact=1, ece≈0), prior-as-grid (grid_info==0 exactly, lift==1 when EV-pick==modal). SELFTEST PASS.
- Deviations: none (NumPy2/torch compiled-against-1.x warning banner observed — pre-existing env condition, non-fatal, everything runs).
- Files changed: experiments/ablation/metrics.py
- Timestamp: 2026-07-21

## Step 3: Split/eval-set module
- Status: ✅ Complete
- Summary: experiments/ablation/splits.py validated — `--report` prints counts table + LEAKAGE CHECKS PASS. Eval lanes: canonical eval n=10,457 (natl 203, 2025-08-01..2026-06-14); pooled eval n=21,663 (natl 397 ≥350 ✓, 2024-08-01..2026-06-08); train/earlystop/eval disjoint. Built + froze wc_inputs.npz = 104 finished WC2026 games (== results.json finished count), 197/2288 imputed starters; DB-has-zero-WC2026 assertion held.
- Deviations: (1) Fixed two lazy-NpzFile bugs the written code shipped with — `cz["ctx"]` was re-decompressed inside the 90,279-item cmap comprehension (each access = full 3.44MB decompress), which crashed with ArrayMemoryError on first run; `z["Xh"]` (~188MB) was re-decompressed 4× in the role_mean comprehension. Both now materialize once. (2) Committed the frozen wc_inputs.npz (95KB) alongside splits.py — not in the plan's file list, but it is the frozen on-target benchmark (DESIGN.md "built once then FROZEN"), same treatment as the committed wc_cache.npz; version-controlling guarantees the freeze rather than relying on rebuild.
- Files changed: experiments/ablation/splits.py, experiments/ablation/wc_inputs.npz (new, frozen)
- Timestamp: 2026-07-21

## Step 4: Experiment runner + smoke test
- Status: ✅ Complete
- Summary: experiments/ablation/run_ablation.py — one config→train→eval→registry-row→report. Imports train_goals (GoalNet, score_matrix, ev_pick, grade, rps, national_context) and metrics/splits; replicates exp_points + T()/tonp() from train_goals.main (they are locals there) per the Phase-1 no-edit rule. Trains `seeds` nets on the split TRAIN mask (early-stop on earlystop lane by RPS), tunes one shared DC-rho on earlystop by league points (production convention), seed-averages grids, scores lanes {eval_all/eval_natl (or canonical_test_*), wc_slate} with metrics.suite, caches per-seed rates to rates/<name>.npz, appends JSONL row, regenerates RESULTS_ABLATION.md (per-lane grid_info/grid_nll/rps/acc/ece/exact_lift/pts_g_31 + Δ-vs-baseline on eval lanes). --report / --diagnose (Step-6 stub) / --force-rerun wired; --decay-halflife + --ctx-extra plumbing declared (schema-stable, inert until Phase 2/3).
- Validation: smoke run (--seeds 1 --epochs 3) completed in 1.15 min (<10 ✓); registry row valid with grid_nll in eval_all; full schema present (name/ts/git_commit/dirty/config/data/metrics/wall_min); dupe-name run refused without --force-rerun; rates cache + report regenerated. Smoke numbers sane (natl/wc grid_info ≈ +0.17 even at 3 epochs).
- Deviations: (1) Plan's inline validation string referenced r['metrics']['test_all']; the FROZEN DESIGN.md lane name is 'eval_all' (pooled) / 'canonical_test_all' (canonical). Honored the frozen contract and asserted against 'eval_all' — documented. (2) Cleared the throwaway smoke row from registry.jsonl (committed empty) so Step-5 baseline is the true row 1; removed rates/smoke.npz and gitignored rates/ (regenerable per-run caches; plan's Step-4 git-add list already excludes them).
- Files changed: experiments/ablation/run_ablation.py (new), experiments/ablation/registry.jsonl (new, empty), experiments/ablation/RESULTS_ABLATION.md (new, generated), experiments/ablation/.gitignore (new, rates/)
- Timestamp: 2026-07-21

## Step 5a: Kick off baseline runs (background)
- Status: ✅ Complete (launched)
- Summary: Background job bfx49cqf3 runs BOTH baselines sequentially (avoids CPU/mem contention that OOM'd earlier): (1) baseline-beta3-w15-canonical --split canonical (tripwire, must reproduce historical TEST rps 0.2130-0.2145) then (2) baseline-beta3-w15 --split pooled (reference row). Both --seeds 5 --epochs 150. Log: data/_ablation_baseline.log.
- Validation: launch confirmed within 5 min — "split=canonical A=62 nctx=10 train=47,386 earlystop=11,210 eval=10,457" (matches splits.py report); 14 python procs alive.
- Deviations: Launched two runs (canonical anchor + pooled reference) per DESIGN.md "Baseline anchoring", not just the single pooled run named in the plan's Step-5a command; both are required by Step 5b.
- Git commit: skipped — no file changes.
- Timestamp: 2026-07-21

## Step 5a (addendum): baseline crash + memory fix + relaunch
- First launch (bfx49cqf3) died: canonical seed 0/1 trained fine, then the process exited 1 with NO Python traceback between seed 1 and seed 2 (native torch crash / OOM), and the outer job was later externally killed while the pooled run was importing torch. No registry row written; no heavy orphan processes left (checked — all surviving python.exe <20MB).
- Fix: run_ablation.py rebuilt the large TRAIN+earlystop tensors (~130MB each, via the bytearray copy that NumPy-2/torch compat forces in T()) on EVERY seed. Refactored to build them ONCE (make_split_tensors) and reuse across seeds — lower peak memory + faster. Verified with a 2-seed/2-epoch smoke: exit 0, seeds now diversify (0.2130 vs 0.2129); the earlier identical 0.2098 across seeds was rounding coincidence at that convergence point.
- Relaunched canonical alone (b2xdvttws); seed 0 reproduced rps=0.2098 @e26. Canonical is slow (~18min/seed: 11,210-sample val loop each epoch); pooled will follow after it completes.
- Timestamp: 2026-07-21

## Step 5b: Verify baseline reproduces known results + WC-slate
- Status: ✅ Complete
- Summary: Both baseline rows registered — baseline-beta3-w15-canonical (59.8min) + baseline-beta3-w15 pooled (56.1min).
- TRIPWIRE (harness == production): ran production src/train_goals.py (its default IS the canonical seed-7 recipe) on current data → TEST-all rps=0.2134 pg=0.706 exact=1125 (inside historical band 0.2130-0.2145 / 0.703-0.716). Then ran the HARNESS with seed 7: val rps=0.2109, TEST rps=0.2134 pg=0.706 exact=1125 — BIT-IDENTICAL to production. Harness training faithfully reproduces production; no bug.
- Seed-variation note: harness seeds 0-4 clustered at TEST rps 0.2111-0.2121 (better than seed 7's 0.2134) — the historical band was seed-7 specifically; the 5-seed ensemble (canonical_test_all rps=0.2108, pg=0.718) is a legitimate grid-averaging improvement, so the registry canonical row reads slightly better than the single-seed band by design. rho choice (0.0 vs -0.05) affects rps negligibly (0.2111 vs 0.2112).
- WC-slate cross-check: harness wc_slate (5-seed) pts/g=0.942 ×104 ≈ 98 pts vs production seed-7 in-script WC total 97 pts (exact=14 correct=55) — match.
- Pooled reference highlights: eval_natl grid_info=+0.127 exact_lift=1.31; wc_slate grid_info=+0.146 exact_lift=1.17; eval_all grid_info=-0.032 (model does NOT beat the empirical prior on the broad all-competitions lane — edge concentrated on national/WC; a real prior-coasting signal for Step 6).
- Infra fixes folded in (required to complete the runs): vectorized val_rps (hda_batch, verified identical to hda_from_P∘score_matrix to 4e-16); per-seed rate checkpointing (rates/<name>.s<k>.npz) so kills resume; --diagnose implemented (dormant until Step 6). Background jobs get idle-killed here → drove runs with continuous active monitoring; per-seed resume made it robust.
- Files changed: experiments/ablation/registry.jsonl (2 rows), RESULTS_ABLATION.md (generated), run_ablation.py (val-speed + resume + diagnose), .gitignore (diagnostics/)
- Timestamp: 2026-07-21

## Step 6: Prior-coasting diagnostic report
- Status: ✅ Complete
- Summary: run_ablation.py --diagnose baseline-beta3-w15 → diagnostics/baseline-beta3-w15.md, folded into RESULTS_ABLATION.md by regen_report. Four diagnostics: (1) grid-NLL vs empirical-prior null per lane; (2) per-scoreline lift tables (eval_natl + wc_slate) — EV-pick precision vs prior cell prob + top-3-mass recall by true scoreline; (3) outcome+exact calibration reliability (eval_natl); (4) plain-language verdict.
- ANSWER to "is the network just guessing modal scores (1-0/1-1)?": Lane-dependent, and honest. Modal prior scoreline = 1-1 (P=0.123), but the model's EV-pick is 1-0 for 241/397 national games and NEVER 1-1 — it does not echo the mode. On the NATIONAL lane it adds +0.127 nats of score-level info over the prior (grid_nll 3.068 vs 3.195), exact-lift 1.31×, off-modal EV-pick precision 0.118 vs prior 0.058 → genuine off-modal signal. WC-slate: +0.146 nats, exact-lift 1.17×. BUT on the broad all-competitions lane grid_info=-0.032 (exact-lift 0.92×) — the model does NOT beat the prior on club-heavy fixtures; its edge is concentrated on the national/WC lanes it's W-upweighted for (the intended target). Calibration: outcome ECE 0.038 (national) — reasonably calibrated; exact-cell slightly over-confident.
- Deviations: enhanced the verdict to also report the all-lane (negative) grid_info for a complete answer, beyond the plan's national-only framing. --diagnose code itself landed in Step 5's commit (was pre-built during the training wait); this step adds the all-lane verdict lines + the generated diagnostics.
- Validation: report contains "# Diagnostics", per-scoreline lift table, and Verdict (asserted).
- Files changed: experiments/ablation/run_ablation.py (verdict), experiments/ablation/RESULTS_ABLATION.md (generated, +diagnostics)
- Timestamp: 2026-07-21
