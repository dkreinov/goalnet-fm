# How to predict WC2026 games with the model

The model (**GoalNet**) predicts the **scoreline** of a match from the two starting XIs' Football Manager
grades + national-team Elo/form context, then picks the scoreline that maximises expected fantasy points
(exact score = 3, correct outcome = 1, wrong = 0).

## TL;DR — predict a game

```bash
python D:/Programming/claude/FM/src/predict_game.py NED-SWE
# multiple at once:
python D:/Programming/claude/FM/src/predict_game.py CIV-GER ESP-CPV ARG-ALG
```

Output per game:
```
=== GER (home) vs CIV (away)  [status=inprogress, imputed 1/22] ===
  xG: GER 2.24 - 0.71 CIV   |   GER win 72%  draw 18%  CIV win 10%
  EV pick: GER 2-0 CIV   top: 2-0 13%  1-0 11%  3-0 10%  2-1 9%  1-1 9%
```
- **EV pick** = what to enter in the fantasy game.
- **imputed N/22** = starters without a grade (filled with role-average). >6 means lower confidence.
- Home/away labels are auto-corrected from squad rosters if the source data swapped them.

This is **pure inference (~4s)** — it loads the trained checkpoint, never retrains.

## Game keys

Keys are `HOMECODE-AWAYCODE` (3-letter FIFA codes) from `worldcup/team_db/lineups.json`. Only games with
a confirmed lineup are predictable. Map team names→codes via `worldcup/team_db/teams/<CODE>.json`
(`team.name`). Results/scores are in `worldcup/team_db/results.json`.

## The production model — goalnet v2

The live checkpoint `data/goalnet.pt` is **goalnet v2** (shipped 2026-07-23 by the six-phase ablation
program; see `experiments/ablation/DESIGN.md`). Recipe: **GoalNet(β=0, W=1)** — a pure calibrated
scoreline distribution with the old EV-points bias and national upweight removed — **plus a de-vigged
market-odds context feature**, as a 5-seed ensemble. On the WC2026 slate it more than doubles the
score-level information of the old β3/W15 model (grid_info +0.35 vs +0.13). The previous model is
archived at `models/archive/goalnet_v1_20260723.pt`.

The odds feature (`data/wc_odds.npz`, de-vigged 1X2, 100% WC-slate coverage) is applied automatically;
games with no odds fall back gracefully to the core model (the feature is masked). `predict_game.py`
also accepts `--market-blend` for an optional post-hoc λ0.5 blend toward the market (marginal on top of
the baked-in feature; default off).

### (Re)building it — only when missing or after changing the data
```bash
python D:/Programming/claude/FM/src/train_goals.py --full --odds --ensemble 5
```
- `--npz players_imp.npz` (default) — the 68k imputed training set (beats strict 48k).
- β=0 / W=1 are now the **defaults** (Phase-2 result: both were points-biases that hurt calibration).
- `--odds` — bake the de-vigged market context (`data/ctx_odds.npz`, 56% train coverage; masked else).
- `--full` — retrain on ALL matches (no held-out split) for the strongest production model, then save.
- `--ensemble 5` — 5-seed ensemble (predict averages the per-match grids across seeds).

## Full data pipeline (only if rebuilding from the DB)

1. `build_player_dataset_imp.py --max-imp 1` → `data/players_imp.npz` (68k matches, ≤1 imputed/side).
2. `build_context.py` → `data/context.npz` (Elo + form over all matches).
3. `train_goals.py --full` → trains + saves `data/goalnet.pt`.
4. `predict_game.py KEY` → predict.

FM26 grades for WC squads: `scrape_wc2026_clubs.py` (nationality pass) then `--by-club` (players abroad);
coverage check `wc2026_squads.py --coverage`. The model falls back to older FM editions when FM26 is
missing (≈89% real-grade coverage).

## How to read / trust it

- **Enter the EV pick verbatim.** The model deliberately picks near-zero draws — that is *correct* under
  this scoring (forcing draws loses points; proven in `draw_diag.py`). Don't override it toward a draw.
- The edge over manual play is grabbing **exact scores** via sensible modal scorelines, not outcome skill.
- Confidence is lower when imputed count is high, for minnow nations, and in general — national accuracy is
  ~0.52–0.60 and varies by period. Treat predictions as a modest edge, not certainties.

## Evaluation / diagnostics (optional)

- `eval_seasons.py` — leave-one-season-out robustness (don't trust a single adjacent split).
- `eval_imp.py` — strict-48k vs imputed-68k training comparison.
- `eval_natl.py` — club vs national performance + national upweighting sweep.
- `draw_diag.py` — draw-pick behaviour vs Dixon-Coles ρ.
- `train_goals.py --full` prints points on all already-played WC2026 games (vs the fantasy leaderboard).

## Agent

Ask the **wc-predictor** agent ("predict NED-SWE", "what should I pick for Germany?") to run this for you
and explain the pick.
