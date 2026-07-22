# Phase 3 → Phase 4 handoff — GoalNet ablation program

**Phase 3 (Context features: Elo momentum/trajectory + stage/rest): COMPLETE (2026-07-22).**
A fresh session starts Phase 4 from this file + `experiments/ablation/DESIGN.md`.

## Headline result — a clean, informative NULL

**No new context feature was adopted.** The base 10-feature `context.npz` stands, and the experiment
config is unchanged: **`--beta 0 --w 1`** (the Phase-2 adopted config; reference row `combo-beta0-w1`).

Re-derived context features have hit their ceiling for this model + data scale:
- **Elo-momentum / form-trajectory (built + ablated, REJECTED):** `data/ctx_momentum.npz` (7 feats: Elo
  delta over last 5 team-matches, form-trend slope, coverage) vs `combo-beta0-w1` → eval_natl grid_info
  **−0.0044** (gate needs ≥+0.02), eval_all −0.0028, rps flat, exact_lift +0.056 (not gated, noise).
  Trajectory adds no independent signal: the base context already carries Elo/form **level**, and
  Phase-2's β0+W1 already subsumed **recency** — so "direction/change" is redundant (Elo-delta ~ level,
  form-trend ~ mean-form). Exactly the null the Phase-2→3 handoff predicted.
- **Stage/knockout (DEFERRED to a data-collection step — NOT rejected):** no stage/round column exists
  in the DB today (`match_kind` ∈ {league, national}), but it IS collectable (WC team_db already has
  stage; historical tournament rounds scrapeable from ESPN/martj42/transfermarkt via the project's
  throttled fetch+cache). Owed an honest ablation after a backfill. Standing rule (user, 2026-07-22):
  never permanently skip a feature for missing data — collect-then-test. Rest-days already in base ctx.

## Runs table

| run | eval_natl grid_info (Δ vs combo-beta0-w1) | eval_all grid_info (Δ) | verdict |
|---|---|---|---|
| combo-beta0-w1 (reference) | +0.2432 | +0.0762 | adopted baseline |
| ctx-momentum | +0.2388 (−0.0044) | +0.0734 (−0.0028) | REJECT (null) |

(`--ctx-extra` runs skip the wc_slate lane — see harness change below.)

## Harness change this phase (committed)

`run_ablation.py` now **skips the wc_slate lane for `--ctx-extra` runs** (95e3822). The frozen
`wc_inputs.npz` carries only the 10-dim BASE context, so extra feature bundles can't standardise it.
Eval lanes (from `players_imp`, covered by the bundle's mids) are unaffected and carry the gate
(eval_natl). If a future phase needs WC-slate scoring WITH extra features, `splits.build_wc_inputs`
must be extended to also emit those features for the WC teams (pre-tournament, leakage-free).

## Frozen config for Phase 4

Unchanged: `--beta 0 --w 1`, base context, reference row `combo-beta0-w1`. Metric suite / splits /
lanes / registry schema / per-seed resume all unchanged. Full rationale:
`experiments/ablation/DESIGN.md` → "Phase-2 adopted defaults" + "Phase-3 adopted context".

## Phase 4 — Market anchor (what the next phase should know)

The context-feature well is dry; the next lever must bring **genuinely new information**, not re-derived
signal. Per the research synthesis (memory `fm-modeling-roadmap`), the strongest untried lever is a
**de-vigged bookmaker-odds anchor** for international matches.
- **Data**: free source = BetExplorer scrape (static HTML; only source covering 2015→ all confederations
  + friendlies); join to the martj42 international-results spine by date + normalized team names (reuse
  club_alias machinery). The Odds API is paid (deferred unless user approves). DB already has b365/avg
  odds columns but those are CLUB-only (0% national coverage in training) — the gap is national odds.
- **Feature**: de-vigged (Shin) closing 1X2 probs + source-quality flag + missingness mask, entering via
  `--ctx-extra` (same mechanism; remember the wc_slate-skip caveat, or extend wc_inputs to carry odds).
- **A/B**: test odds as a ctx feature AND as a residual-anchor (blend model logits toward market) — the
  literature says model≈market and the ensemble of the two is best.
- **Gate**: eval_natl grid_info improvement over `combo-beta0-w1` (same yardstick +0.02); this is where a
  real gain is most plausible since odds carry information the lineup + Elo can't see.
- **Risk**: national-team name matching across sources (small fixed set — should be easy); pre-2018
  friendly coverage; site anti-bot (throttle via src/fetch.py + disk cache, per repo convention).

Phase 4 is currently a skeleton in the phase-state — PLAN IT FIRST before executing.

## Ops notes (unchanged, reconfirmed)

- Interpreter `C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe`; NumPy2/torch
  banner benign; 4 cores / ~4GB free → one run at a time.
- **Detached Windows scheduled task** (`experiments/ablation/run_phase3.ps1` template) is the reliable
  way to run unattended (session bg jobs idle-killed); per-seed caches resume; hourly `ScheduleWakeup`
  self-check worked well across Phases 2–3.
- PowerShell gotcha found this phase: a **single-element** `$runs = @(@(...))` array gets flattened —
  use a direct `& $PY ...` call for one-run scripts (see run_phase3.ps1).

## Resume pointer

Phase state: phase 3 → COMPLETE, current phase = 4. Next action: PLAN Phase 4 (market anchor) as its own
plan file, then execute. Adopted config for all Phase-4 diffs = `combo-beta0-w1` (`--beta 0 --w 1`,
base context).
