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
