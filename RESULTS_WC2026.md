# WC2026 fantasy model — results & statistics

Goal: predict WC2026 scorelines (not just who-wins) from FM 11v11 grades + national context, to beat the
fantasy league (scoring: exact score = 3, correct outcome = 1, wrong = 0).

## Production model: `data/goalnet.pt` (the ONLY live model)
- **GoalNet** (`src/train_goals.py`) — the production model (everything else is a superseded experiment).
  Grade encoder (xfmr over the 11 + role embedding + national Elo/form context); head emits two
  **expected-goals rates** via an attack/defence structure (logλ_home = home_adv + att_h − def_a + ctx_h).
- **Loss: Poisson NLL − β·expected_points (β=3, decision-focused).** Trained directly for OUR 3/1 scoring
  (not just calibrated goals). National matches upweighted (`--w 5`). Saved as the single `goalnet.pt`;
  `predict_game.py` / the `wc-predictor` agent load only this — each retrain overwrites it.
- Inference: full scoreline distribution (double-Poisson + Dixon-Coles ρ=0.05) → EV-optimal scoreline.
- Superseded (deleted): the H/D/A result net (`train_pos2.py`) and early prototypes (posnet/model_nn).

## Improvement journey — how the model got better (held-out pts/game unless noted)
| step | what changed | result |
|---|---|---|
| baseline | majority/prior (who-wins) | RPS 0.2288 |
| 11v11 result net | xfmr over the 22 graded starters → H/D/A | RPS 0.2132 |
| + Elo/form context | team-strength priors the lineup can't see | RPS 0.2100 |
| → GoalNet (scoreline) | Poisson goals head → EV-pick under 3/1 | enables exact scores |
| + national upweight (W=5) | nationals weighted 5× | nat acc ~0.55→0.60 |
| + imputed 68k set | keep 10/11-graded games (impute 1) | pts/g +0.011 |
| + edition-fallback | FM26 else most-recent edition for WC squads | coverage 73%→90% |
| + WC2022 connection | club-anchored links | 33→54/64 matches graded |
| **+ decision-focused loss (β=3)** | **train for 3/1 points, not just goals** | **pts/g 0.699→0.714 (+2%), acc 0.487→0.496** |
- Feature metadata (position/competition/formation/attendance embeddings) tested — **none robustly helped**
  (model at its information ceiling; attributes + Elo already carry the signal). Tie-floor delta tested —
  **a points wash** (false positives cancel catches), dropped.

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

### Decision-focused loss — the one lever that worked (2026-06-21)
Train the model to maximize EXPECTED FANTASY POINTS, not just Poisson goal-fit. The EV-pick (argmax) is
non-differentiable, so the loss uses a soft-pick surrogate: build P(h,a) → EV(i,j) per scoreline →
π=softmax(EV/τ) → maximize Σ π·points(cell,truth). Anchored by Poisson: `loss = Poisson − β·exp_points`.
Held-out 2024-25 (`train_loss_ab.py`):

| loss | acc | RPS | pts/g | exact% | ties picked | ties caught |
|---|---|---|---|---|---|---|
| poisson (old production) | 0.487 | 0.2122 | 0.6993 | 10.6 | 1 | 0 |
| 3/1-weighted likelihood | 0.485 | 0.2119 | 0.7061 | 11.0 | 0 | 0 |
| **decision-focused (β=3)** | **0.493** | 0.2122 | **0.7137** | 11.0 | **28** | **9** |

- Decision-focused **beats both** Poisson (+2%) and the simpler weighted likelihood (+1%) — the soft-pick is
  genuinely better (invariant to points-irrelevant goal-count mass; reallocates capacity to outcome).
- It **learns to pick the worthwhile ties on its own** (28 vs 0) — exploiting that exact-ties concentrate on
  0-0/1-1/2-2. Adopted as production (β=3).
- **Tie-floor delta NOT adopted:** forcing more ties (ε slack) is a points wash — ε=0.03 adds 47 false
  positives to catch 16 extra draws, pts/g flat (0.7137→0.7136); higher ε loses. The model's natural ties
  are already EV-optimal.

## FM26 grade coverage (WC2026 squads)
1,248 players across 48 squads (source: worldcup project). After the overnight nationality + by-club scrape:
FM26-graded **1,047/1,248 (84%)**; **effective 90%** with edition-fallback (most-recent edition when FM26
missing). Remaining ~10% have no grade in ANY FM edition (obscure minnow-league players) = data floor;
they're role-mean imputed (low impact, weak teams). Played-game lineups run at ~0–1/22 imputed for major
sides. Scrapers: `scrape_wc2026_clubs.py` (`--by-club` for players abroad, `--fast` for paced bursts).

## Scraper fix (fminside throttle)
Root cause: session `database_version` reverts 7→5 under `update_filter` rate → stale/wrong clubs.
Fix (`src/scrape_wc2026_clubs.py`): squad pages are URL-driven (`/clubs/7-fm-26/{id}`, no filter) → filter-free
+ cached; `update_filter` used once/nation for club-ID discovery with db7-verification + cooldown-retry +
resumable JSON cache.
