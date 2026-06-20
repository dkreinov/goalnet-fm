# WC2026 fantasy model — results & statistics

Goal: predict WC2026 scorelines (not just who-wins) from FM 11v11 grades + national context, to beat the
fantasy league (scoring: exact score = 3, correct outcome = 1, wrong = 0).

## Models
- **GoalNet** (`src/train_goals.py`) — the production model. Same grade encoder as the result net (xfmr
  over the 11 + role embedding + national Elo/form context), but the head emits two Poisson **expected-goals
  rates** via an attack/defence structure (logλ_home = home_adv + att_h − def_a + ctx_h; logλ_away = att_a −
  def_h + ctx_a). Trained with Poisson NLL on real goals; who-wins is derived. Inference: full scoreline
  distribution (double-Poisson + tunable Dixon-Coles ρ) → EV-optimal scoreline under the 3/1 scoring.
- Result-only net (`src/train_pos2.py`, H/D/A) kept for comparison; goal model strictly better for this game.
- National matches upweighted (`--w 5`) in the loss — improves nationals at no cost to clubs.

## Datasets (`src/build_player_dataset*.py`)
- **strict 48,355** matches — both XIs fully FM-graded (11v11), per-starter 62 attrs + role.
- **imputed 68,761** matches (`players_imp.npz`, ≤1 imputed starter/side filled with role-mean) — +42% data.
- National Elo/form context computed over 2,707 international matches (no leakage).

## Key statistics

### Season-holdout robustness (leave-one-season-out) — the honest eval
Contiguous recent splits flatter the model, *especially nationals*. Per held-out season:

| test season | ALL acc | ALL rps | NATL acc | NATL pts/g |
|---|---|---|---|---|
| 2020-21 | 0.469 | 0.217 | 0.532 | 0.80 |
| 2021-22 | 0.484 | 0.211 | 0.503 | 0.76 |
| 2022-23 | 0.484 | 0.213 | 0.426 | 0.57 |
| 2023-24 | 0.492 | 0.208 | 0.515 | 0.80 |
| 2024-25 | 0.486 | 0.213 | 0.518 | 0.66 |
| 2025-26 | 0.481 | 0.214 | **0.651** | **1.02** |
| non-adjacent (20-21+22-23) | 0.480 | 0.215 | 0.473 | 0.68 |

- **Overall model is stable** across seasons (ALL acc 0.469–0.492). 
- **National accuracy swings 0.43–0.65**; 2025-26 is the lucky outlier. Honest national ≈ **0.52 acc / 0.77 pts/g**;
  non-adjacent estimate 0.473. *Evaluate with season-holdout / non-adjacent, not adjacent years.*

### Imputed-data ablation (same clean 7,061 test games)
| training set | ALL acc | ALL rps | ALL pts/g | NATL rps |
|---|---|---|---|---|
| strict 48k | 0.481 | 0.2141 | 0.703 | 0.1703 |
| **imputed 68k** | **0.487** | **0.2124** | **0.714** | **0.1678** |
Adding the 10+1-imputed games is a small but consistent overall win → adopt the 68k set.

### Played WC2026 games (32, group stage)
GoalNet (full-data, W=5): **exact=5, correct=14, wrong=13 → 29 pts**, ~3rd on the leaderboard
(top 31; YOU/you 19). CAVEAT: 29% of starters imputed (grade coverage incomplete) and these games sit
in the favourable 2025-26 window — treat 29 as a hopeful-case, not the cross-season expectation.

## Feature-ablation (2026-06-21) — does extra data help? Mostly NO.
Held-out season 2024-25 (n=11,171), GoalNet, national-weighted. Each feature added to baseline:

| config | acc | rps | pts/g | exact% | natl acc |
|---|---|---|---|---|---|
| baseline (4-role) | 0.490 | 0.2113 | 0.709 | 10.9 | 0.547 |
| + detailed position (9) | 0.484 | 0.2123 | 0.703 | 10.9 | 0.558 |
| + competition embedding | 0.483 | 0.2117 | 0.698 | 10.7 | 0.537 |
| + formation embedding | 0.490 | 0.2107 | 0.705 | 10.8 | 0.563 |
| + metadata (kickoff/attendance) | 0.491 | 0.2109 | 0.711 | 11.0 | 0.526 |
| **+ attendance auxiliary** | 0.491 | 0.2112 | **0.720** | **11.4** | 0.563 |
| ALL combined | 0.486 | 0.2117 | 0.710 | 11.2 | 0.568 |

- **Detailed position embedding and competition embedding HURT** — the FM attributes already encode position
  (a winger's pace/crossing vs a striker's finishing) and the Elo context already encodes league strength.
- **Formation / metadata: marginal/neutral.**
- **Attendance-as-auxiliary** is the only positive (pts/g 0.709→0.720, exact 10.9→11.4%) — a mild regulariser —
  but it's a single-season result and ALL-combined washed back to baseline, so it's not robust. Not adopted.
- Verdict: the attributes + Elo/form context are near the information ceiling; match metadata adds little.
  No `round/stage` column exists (match_kind is only league/national), so "final vs group" couldn't be tested.
  Scripts: `build_player_dataset_pos.py`, `build_meta.py`, `train_enr.py`.

## FM26 grade coverage (WC2026 squads)
1,248 players across 48 squads (source: worldcup project). FM26-graded **849/1,248 (68%)** after the
nationality club-route scrape. ~399 still missing (mostly players based abroad → `--by-club` pass pending).

## Scraper fix (fminside throttle)
Root cause: session `database_version` reverts 7→5 under `update_filter` rate → stale/wrong clubs.
Fix (`src/scrape_wc2026_clubs.py`): squad pages are URL-driven (`/clubs/7-fm-26/{id}`, no filter) → filter-free
+ cached; `update_filter` used once/nation for club-ID discovery with db7-verification + cooldown-retry +
resumable JSON cache.
