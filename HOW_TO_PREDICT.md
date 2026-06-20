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

## (Re)building the trained model — only when needed

The checkpoint `data/goalnet.pt` already exists. Rebuild it only if missing or after changing the data:
```bash
python D:/Programming/claude/FM/src/train_goals.py --w 5 --epochs 150 --full
```
- `--npz players_imp.npz` (default) — the 68k imputed training set (beats strict 48k).
- `--w 5` — national-team matches upweighted 5× (helps nationals, no cost to clubs).
- `--full` — retrain on ALL matches (no held-out split) for the strongest production model, then save.
- Add `--game KEY` to also print a detailed prediction at the end.

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
