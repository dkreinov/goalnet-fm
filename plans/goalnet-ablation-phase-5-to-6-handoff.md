# Phase 5 → Phase 6 handoff — GoalNet ablation program

**Phase 5 (Architecture: cross-team attention + plus-minus): COMPLETE (2026-07-22), NULL result.**
SESSION_MODE was autonomous; PHASE_MODE pause-between-phases. A fresh session starts Phase 6 from
this file + `plans/goalnet-ablation-phase-state.md` + `experiments/ablation/DESIGN.md` (the
"Phase-5 adopted architecture" section has the verdict table).

## Headline result

**Architecture is NOT the bottleneck.** All four arms scored below `combo-beta0-w1` on eval_natl
grid_info (baseline +0.2432, noise ~0.002): cross-team attention hurts whether fused early
(cross22, joint 22-token transformer, −0.0175) or late (latecross, one cross-attn block, −0.0171);
plus-minus ratings add nothing as team aggregate (−0.0184) or per-player channels (−0.0119).
Combined with Phase 3 (momentum null) and Phase 4 (market adopted), the picture is consistent:
**the per-team GoalNet at β0/W1 has extracted what the FM-attribute + context data offers; only
genuinely NEW information sources (market odds, and someday richer event data) move the needle.**
The odds-informed bar (+0.1803 covered natl) was never threatened.

## Runs + verdicts (registry rows 20–23)

| run | eval_natl Δ | eval_all Δ | wc_slate Δ | verdict |
|---|---|---|---|---|
| arch-cross22-s3 | −0.0175 | −0.0043 | −0.0053 | REJECT |
| arch-latecross-s3 | −0.0171 | −0.0043 | −0.0022 | REJECT |
| ctx-pm-s3 | −0.0184 | −0.0005 | skipped | REJECT |
| pm-channel-s3 | −0.0119 | −0.0023 | skipped | REJECT |

No seeds=5 confirmations or odds-bar scoring were run (nothing passed the s3 gate — Step 7
correctly skipped per plan).

## What was built (kept, committed)

- `experiments/ablation/models.py` — arch zoo: goalnet (bit-for-bit parity vs tg.GoalNet, verified
  on params/forward/train-step at seeds 0,7), Cross22GoalNet, LateCrossGoalNet.
- `run_ablation.py --arch <name>` and `--pm-channel players_pm.npz` (asserts mids alignment,
  appends per-slot channels pre-standardization, skips WC lane, records config.flags).
- `src/build_plusminus.py` → `data/players_pm.npz` (69,053×11×2 per-slot [pm_shrunk, has_pm],
  98.9% coverage, 1/69,053 slot-align failure) + `data/ctx_pm.npz` ([pm_team_diff, pm_cov]).
  Leakage-free by construction (emission before accumulator update). Segment-level goal
  attribution on 51,264 matches (own-goal `team_side` = benefiting side), minutes-weighted
  fallback otherwise.
- Finding worth remembering: net-of-club plus-minus is a rotation/transfer proxy — INVERSELY
  correlated with winning (−0.10). Not a bug; a selection effect (low-minute players fatten GD in
  blowouts; big-club signings carry weak-club histories).

## Frozen config going into Phase 6 (production candidate)

- **Core: standard GoalNet (per-team encoder), β=0, W=1** (`combo-beta0-w1` recipe).
- **Market layer: ctx-odds feature + λ≈0.5 outcome-mass blend** (best known: +0.1803 covered natl,
  rps 0.1918, ece 0.042; see Phase-4 handoff).
- DC rho stays val-tuned per run; pooled split; frozen metric suite; registry append-only (23 rows).

## Open questions / debts carried into Phase 6

- **Multi-source odds TODO (unchanged, most valuable next data work):** natl eval coverage is only
  35% (137/397); no WC2026 odds. oddsportal (free) → The Odds API (paid, needs user spend
  approval) → Betfair historic. Enables the wc_odds.npz `--wc-extra` extension (designed Phase-4
  plan Step 4, unbuilt) and the replay's live market layer.
- Stage/knockout still owed a fair test with REAL round labels (collect-then-test rule).
- Plus-minus: RAPM-style ridge (opponent-adjusted) noted as possible future work, low priority
  given the P1/P2 nulls.

## Phase 6 skeleton (from program map, updated)

Goal: WC2026 day-by-day replay backtest (walk-forward, only-past-info context, real lineups,
optional incremental fine-tune) + model selection + production retrain (goalnet v2) + docs/memory
update + checkpoint archive. Updated by Phases 2–5:
- Candidate set is now SMALL: combo-beta0-w1 core ± market layer (feature, blend, feature+blend).
  No architecture variants survive.
- Replay must handle odds coverage honestly (B1 blend is identity where odds missing) — with
  current sources there are NO WC2026 odds, so the replay's market layer only helps if multi-source
  collection runs first. Decide early in Phase 6 planning whether to collect odds first.
- Production retrain touches `src/train_goals.py` / `goalnet.pt` for the first time — the
  no-edit rule lifts ONLY in the retrain step, with the old checkpoint archived beforehand.
- Gate: replay pts + grid-NLL beat current production's replay on the same 104-game slate.

## Resume pointer

Phase state: phase 5 → COMPLETE (null), current phase = 6 (NOT STARTED — plan it first per
PHASE_MODE pause-between-phases). Registry 23 rows; working tree clean at the Step-9 commit.
