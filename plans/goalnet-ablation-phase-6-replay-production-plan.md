# Phase 6 — WC2026 day-by-day replay + model selection + production retrain (FINAL phase)

Planned 2026-07-22 from `plans/goalnet-ablation-phase-5-to-6-handoff.md` +
`plans/goalnet-ablation-phase-state.md` + `experiments/ablation/DESIGN.md` (Phase-2..5 adopted
sections reconfirmed). PHASE_MODE: moot (final phase). SESSION_MODE: asked at plan approval.

## Goal

Walk-forward WC2026 day-by-day replay (only-past-info context, real lineups) over the 104-game
slate; compare the frozen candidate set (combo-beta0-w1 core ± market layer), each replayed BOTH
frozen and incrementally fine-tuned; select the winner; retrain production goalnet v2 (full
cutover); update docs + memory; archive old checkpoints; write the program retrospective.

## Gate decisions (user, 2026-07-22 — do not re-litigate)

1. **Odds collection: oddsportal.com scrape FIRST (free).** Same throttled `src/fetch.py`+cache
   style as BetExplorer; new parser + soft-404 handling. The Odds API (paid) NOT approved —
   fallback on scrape failure is covered-lane-only market scoring, not spend.
2. **Bench/subs feature: BACKLOGGED.** The collect-then-test debt stays recorded (named unused
   bench via ESPN scrape; `match_player` subs-who-appeared is leakage). Not in Phase 6 scope.
3. **Incremental fine-tune: ALL candidates.** Every candidate is replayed both frozen and
   fine-tuned (walk-forward update on finished WC games) — the frozen/fine-tuned delta is itself
   a reported result.
4. **Production cutover: FULL.** Retrain goalnet v2 (β0/W1 + winning market layer) on all data,
   update `src/train_goals.py` defaults, point `wc-predictor` at the new checkpoint + blend,
   update README / RESULTS_WC2026 / HOW_TO_PREDICT / memory, archive old `goalnet.pt`.

## Frozen contracts (respect throughout)

- Core = standard GoalNet per-team encoder, **β=0, W=1**, pooled split (`combo-beta0-w1` recipe).
  No architecture variants, no momentum/pm context (Phases 3+5 null).
- Market layer options: (a) ctx-odds feature, (b) post-hoc λ-blend (identity off-coverage),
  (c) both — best known (c) λ0.5 = +0.1803 covered natl / rps 0.1918 / ece 0.042.
- Registry `experiments/ablation/registry.jsonl` append-only (23 rows at phase start);
  RESULTS_ABLATION.md regenerated, never hand-edited; frozen metric suite (metrics.py);
  covered-subset (identical-matches) discipline for any partially-covered feature.
- `src/train_goals.py` / `goalnet.pt` no-edit rule is lifted ONLY in Step 7 (archive first).
- DC rho val-tuned per run; per-seed rate caches under `experiments/ablation/rates/`.

## Steps

### Step 1 — Oddsportal scrape: `src/scrape_oddsportal.py` → extend odds inventory
- Recon first (throttled): confirm oddsportal has (a) WC2026 finished-match closing 1X2,
  (b) 2024–2026 natl qualifiers/friendlies missing from BetExplorer (eval coverage 35%,
  137/397). If the site is unusable (anti-bot / JS-only), record it and fall back per gate 1.
- Scraper follows `scrape_betexplorer.py` conventions: `src/fetch.py` throttle + disk cache,
  soft-404 detection, log to `data/_oddsportal.log`. De-vig with the existing Shin machinery.
- Outputs: `data/wc_odds.npz` (104-game slate odds, `--wc-extra` pattern designed in Phase-4
  plan Step 4, unbuilt until now) + refreshed `data/ctx_odds.npz` (append new natl rows;
  keep source flag). Report new coverage numbers (train / eval / wc-slate).
- Gate to proceed with market-layer replay arms on the WC lane: wc-slate odds coverage ≥ ~70%.
  Below that, market arms still run but their wc-lane read uses covered-subset only.

### Step 2 — Replay driver: `experiments/ablation/replay_wc.py`
- Walk-forward day-by-day over the 104-game slate grouped by matchday date. For each day:
  context features rebuilt AS-OF that date (only-past-info — this is the leakage-critical
  piece), real lineups from worldcup results.json, predict → score with the frozen metric
  suite → then reveal results.
- Reuses `splits.py` inputs (`wc_inputs.npz`), `metrics.py`, rate caches. Supports:
  `--model` (checkpoint or harness config), `--odds` (feature npz), `--blend LAMBDA`,
  `--finetune` (post-matchday incremental update: small LR, few epochs on finished WC games
  appended to the natl pool; fine-tune state carried forward day to day).
- Emits one registry row per replay run (lane `wc_replay`) + a per-matchday CSV under
  `experiments/ablation/diagnostics/` for the narrative (cumulative pts, grid-NLL, calibration).
- **Production reference row first:** replay CURRENT production `goalnet.pt` (β3/W15,
  natl-finetune) on the same slate — this is the selection gate's bar. Sanity-check against
  eval_harness / old `wc_cache.npz` numbers before trusting the driver.
- Tripwire: replaying a candidate with the static-slate config must reproduce its registry
  wc_slate metrics bit-for-bit when walk-forward context is disabled (driver-correctness check).

### Step 3 — Candidate replays (the comparison matrix)
- Candidates (4): core; core+ctx-odds; core+blend λ0.5; core+ctx-odds+blend λ0.5.
- Each × {frozen, fine-tuned} = 8 replay runs (+1 production reference, +1 production
  fine-tuned optional for symmetry). Seeds: s3 mean for the replay (runs are cheap at eval
  time; fine-tune arms re-run per seed). Long runs → detached scheduled tasks per ops rules.
- Odds-coverage asymmetry handled by covered-subset companion metrics on every market arm.

### Step 4 — Selection
- Gate: winner must beat the production-reference replay on BOTH replay pts (pts_g_31 on the
  real schedule) and grid-NLL/grid_info, with calibration (ece, reliability) reported; ties
  broken toward the simpler layer (blend-only beats feature+blend if within noise ~0.002).
- Decide fine-tune's fate on its own evidence: it enters production procedure only if it wins
  clearly; otherwise production ships frozen weights.
- Write the verdict table into DESIGN.md ("Phase-6 selection" section) before touching production.

### Step 5 — Production retrain + cutover (no-edit rule lifted HERE only)
- Archive first: `models/archive/goalnet_v1_YYYYMMDD.pt` (+ the natl-finetune variant if
  separate) and note the archive path in README.
- Retrain goalnet v2 with the selected recipe on ALL data through 2026-06-14 (`players_imp.npz`)
  — same seeds/protocol as the winning registry row; tripwire: retrained model must reproduce
  that row's pooled-eval metrics within noise before cutover.
- Update `src/train_goals.py` defaults (β=0, W=1, selected market wiring); produce production
  blend config (λ, coverage-identity behavior) as a loadable artifact, not a hardcode.
- Point `wc-predictor` agent at the new checkpoint + blend (update its instructions/paths).

### Step 6 — Docs, memory, retrospective, close-out
- RESULTS_WC2026.md: adopted-change narrative (replay story, selection verdict, v2 config).
- README + HOW_TO_PREDICT: new checkpoint, blend step, odds-refresh instruction.
- Memory updates: fm-ratings-project.md (pipeline status), fm-modeling-roadmap.md (program
  outcome; bench-feature backlog with leakage note; architecture deprioritized-not-closed).
- Program retrospective appended to phase-state file (what moved the needle: β0/W1 + odds;
  what nulled: momentum, stage-thin, architecture, plus-minus). NO next-phase handoff.
- Final commits (plans/ needs `git add -f`); delete stray run_phase*.ps1 or commit them;
  confirm scheduled tasks unregistered.

## Risks

- **Replay leakage** — context rebuilt as-of each matchday is the one design-heavy piece; the
  Step-2 tripwire (static-config reproduction) is the guard.
- **Oddsportal unknowns** — anti-bot / JS rendering / historical WC2026 page coverage; recon
  before building the full parser; fallback is covered-lane-only (gate 1).
- **Fine-tune instability** — tiny per-day updates on ≤4 games can thrash; use small LR + few
  steps, carry per-seed state, and report the frozen/fine-tuned delta honestly (a null is fine).
- **Retrain regression** — guarded by archive-first + reproduce-registry-row tripwire.
- **Coverage asymmetry** — market arms judged on covered subsets where wc odds are partial.

## Ops (from handoff)

- Interpreter: `C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe` (bare
  `python` not on PATH). NumPy2-vs-torch banner non-fatal. Never index a lazy NpzFile in a loop.
- Long runs: detached scheduled tasks (see `experiments/ablation/run_phase5*.ps1` templates,
  direct `& $PY 'arg' ...`) + ScheduleWakeup checks. `plans/` gitignored → `git add -f`.
- Subagents: ALLOWED this phase (per the Phase-6 handoff prompt as pasted by the user, which
  supersedes the older "not allowed" line in the handoff file).
