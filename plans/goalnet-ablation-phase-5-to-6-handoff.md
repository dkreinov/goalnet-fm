# Phase 5 → Phase 6 handoff — GoalNet ablation program

**Phase 5 (Architecture: cross-team attention + plus-minus): COMPLETE (2026-07-22), NULL result.**
SESSION_MODE was autonomous; PHASE_MODE pause-between-phases. A fresh session plans Phase 6 from
this file + `plans/goalnet-ablation-phase-state.md` + `experiments/ablation/DESIGN.md` (the
"Phase-5 adopted architecture" section has the verdict table). Phase 6 is the FINAL phase:
replay backtest → model selection → production retrain → docs. It MUST be planned first
(plan-skill process → `plans/goalnet-ablation-phase-6-replay-production-plan.md`).

## Headline result (Phase 5)

**Architecture is not the current bottleneck.** All four arms scored below `combo-beta0-w1` on
eval_natl grid_info (baseline +0.2432, noise ~0.002): cross-team attention hurts whether fused
early (cross22, joint 22-token transformer, −0.0175) or late (latecross, one cross-attn block,
−0.0171); plus-minus adds nothing as team aggregate (−0.0184) or per-player channels (−0.0119).
Scope, stated honestly: two representative fusion points at ~150k params on 69k matches —
DEPRIORITIZE architecture search, don't declare it closed; reopen only with a concrete hypothesis
for information the market lacks, or 5–10× data. Combined with Phases 3–4: **only genuinely NEW
information moves the needle; the odds-informed bar (+0.1803 covered natl) was never threatened.**

| run | eval_natl Δ | eval_all Δ | wc_slate Δ | verdict |
|---|---|---|---|---|
| arch-cross22-s3 | −0.0175 | −0.0043 | −0.0053 | REJECT |
| arch-latecross-s3 | −0.0171 | −0.0043 | −0.0022 | REJECT |
| ctx-pm-s3 | −0.0184 | −0.0005 | skipped | REJECT |
| pm-channel-s3 | −0.0119 | −0.0023 | skipped | REJECT |

No s5 confirmations / odds-bar scoring (nothing passed the s3 gate — Step 7 skipped per plan).

## Frozen config for Phase 6 (the production candidate set)

- **Core: standard GoalNet (per-team encoder), β=0, W=1** (`combo-beta0-w1` recipe, pooled split).
- **Market layer options: (a) ctx-odds feature, (b) post-hoc λ-blend (identity off-coverage),
  (c) both** — best known (c) with λ0.5 = +0.1803 covered natl / rps 0.1918 / ece 0.042.
- No architecture variants survive. DC rho val-tuned per run; frozen metric suite; registry
  append-only (23 rows).
- Contracts that persist: `--arch`/`--pm-channel`/`--ctx-extra` runner flags; per-seed rate caches
  under `experiments/ablation/rates/`; `blend_market.py` λ-blend pattern; covered-subset
  (identical-matches) comparison methodology for any partially-covered feature.

## Phase 6 planning inputs — decisions the plan must gate on

Ask these BEFORE freezing Phase-6 steps (Open Questions Gate, one at a time, in this order):

1. **Odds collection first?** There are NO WC2026 odds (BetExplorer lacks them) and natl eval
   coverage is 35% (137/397). Without new sources, the replay compares candidates with a
   handicapped market layer. Options: (a) oddsportal.com scrape first (free, same fetch.py-cache
   style as BetExplorer, needs its own parser + soft-404 handling); (b) The Odds API (paid
   ~$30–100 — REQUIRES explicit user spend approval; snapshots 2022→, covers WC/qualifiers/
   friendlies); (c) replay without WC odds and score the market layer on the covered-natl lane
   only. Recommendation to present: (a) first, (b) only on user approval, (c) as fallback.
2. **Bench/subs feature — now (Phase 5b/6 side-arm) or backlog?** User-raised owed test
   (collect-then-test rule): the model only sees the starting XI; the ANNOUNCED bench (known ~1h
   pre-kickoff, like lineups) is real strength info (subs play ~30% of minutes). LEAKAGE TRAP:
   `match_player` only has subs who APPEARED (manager choice correlates with game state) — the
   clean feature needs the named unused bench, likely scraped from ESPN match pages (coverage
   unknown; check first). Cheap-first: ctx feature "mean FM rating of top-3 bench players";
   full version: 11+N tokens with a starter/bench embedding (`--arch benchnet` in models.py).
3. **Incremental fine-tune during replay?** The replay may optionally fine-tune day-by-day on
   finished WC games. Adds realism and cost/complexity; decide in or out before the plan freezes.
4. **Production cutover scope.** The retrain step lifts the `src/train_goals.py`/`goalnet.pt`
   no-edit rule for the FIRST time. The plan must archive the old checkpoint + specify what the
   `wc-predictor` agent should load afterward, and which docs (README / RESULTS_WC2026 /
   HOW_TO_PREDICT / memory) get the final narrative.

## Phase 6 skeleton (expand into the full plan)

Goal: walk-forward WC2026 day-by-day replay (only-past-info context, real lineups) over the
104-game slate; compare the candidate set; select; retrain production core (goalnet v2, β0/W1)
+ market layer as a separate production step; update docs + memory; archive old checkpoints.
- Inputs: frozen config above; `experiments/ablation/wc_inputs.npz` (104 games, raw inputs);
  worldcup results.json (lineups); rate caches for all 23 registry rows; ctx_odds.npz (38,403).
- Likely new files: replay driver under `experiments/ablation/` (walk-forward eval, reuses
  splits/metrics); optional `src/scrape_oddsportal.py` + wc_odds.npz (`--wc-extra` pattern,
  designed in Phase-4 plan Step 4, unbuilt); optional bench scrape + `ctx_bench.npz`.
- Validation gate: replay pts + grid-NLL beat CURRENT PRODUCTION goalnet.pt's replay on the same
  slate (eval_harness / old wc_cache.npz gives the production reference); calibration reported.
- Risks: replay leakage (context must be rebuilt as-of each matchday); odds-coverage asymmetry
  between candidates (use covered-subset discipline); retrain regression (archive + tripwire
  against the registry reference row before cutover); scraping unknowns (oddsportal anti-bot).
- Final step: production cutover + docs + memory + program close-out (no next-phase handoff —
  this is the last phase; write a program retrospective instead).

## Ops facts a fresh session needs

- Interpreter: `C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe` (bare
  `python` NOT on PATH). NumPy-2-vs-torch import banner is pre-existing and non-fatal.
- Long runs: detached Windows scheduled tasks (Register/Start/Unregister-ScheduledTask, see
  `experiments/ablation/run_phase5*.ps1` templates — direct `& $PY 'arg' ...` calls, never
  single-element PS arrays) + hourly ScheduleWakeup checks. Runs are resumable via per-seed caches.
- Gotchas: NEVER index a lazy NpzFile in a loop (materialize first — bit us 4×); `plans/` is
  gitignored (always `git add -f`); subagents NOT allowed (user rule); s3 exploratory ≈ 25 min/run.
- Repo state: clean at the addendum commit on master; registry 23 rows; scheduled tasks all deleted.

## Recommended setup for Phase 6

- **Planning + execution model: Fable** (user's current default) or Opus — the replay driver is
  the one genuinely design-heavy piece (walk-forward correctness = leakage risk); everything else
  follows established harness patterns. Thinking: High.
- Context: plan in a FRESH session (/clear) — this file + phase-state carry everything needed.
- SESSION_MODE: ask at plan approval (user chose autonomous for Phases 2–5). PHASE_MODE moot
  (final phase).

--- HANDOFF PROMPT (paste into fresh session) ---
Continue plan from: plans/goalnet-ablation-phase-state.md
Read first: plans/goalnet-ablation-phase-state.md + plans/goalnet-ablation-phase-5-to-6-handoff.md
Resume at: Phase 6 (WC2026 replay + selection + production retrain) — plan it first, then execute
Execution mode: not chosen yet   PHASE_MODE: pause-between-phases (final phase)
Model: Fable (or Opus), thinking High
Context: RECOMMENDED: run /clear before starting
Before executing:
1. Read the file(s) above fully.
2. Check git status.
3. Reconfirm frozen contracts (DESIGN.md Phase-2..5 adopted sections).
4. Run the plan-skill process: ask the 4 gate questions in "Phase 6 planning inputs" ONE AT A TIME,
   then save plans/goalnet-ablation-phase-6-replay-production-plan.md and ask execution mode.
Important:
- Candidates: combo-beta0-w1 core ± market layer only. The Odds API costs money — needs explicit user approval.
- Production retrain lifts the train_goals.py/goalnet.pt no-edit rule ONLY in its retrain step (archive first).
- Subagents not allowed; long runs = detached scheduled tasks; full interpreter path; plans/ needs git add -f.
--- END HANDOFF PROMPT ---
