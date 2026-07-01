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

## Ablations: national weight + data scaling (2026-06-21, held-out 2024-25)
**National-weight sweep** (decision-focused; national subset n=194, noisy but trend clear):

| W | ALL acc | ALL pts/g | NATL acc | NATL rps | NATL pts/g |
|---|---|---|---|---|---|
| 1 | 0.495 | 0.716 | 0.546 | 0.191 | 0.773 |
| 5 (was production) | 0.493 | 0.714 | 0.531 | 0.191 | 0.727 |
| 15 | 0.495 | 0.714 | 0.572 | 0.184 | 0.763 |
| 30 | 0.493 | 0.715 | 0.577 | 0.189 | 0.804 |
- Higher national weight (15–30) lifts NATIONAL accuracy 0.53→0.58 and RPS 0.191→0.184 at **zero cost** to
  overall (ALL flat ~0.49/0.71 even at W=30). W=5 was sub-optimal for nationals → **use W≈15 for the WC model.**

**Data-fraction sweep** (random removal, W=5): ALL pts/g 100%→0.7137, 90%→0.7116, 80%→0.7105, 70%→0.7062,
60%→0.7101, 50%→0.7013; RPS 0.2122→0.2161.
- **The model is DATA-SATURATED** — flat to ~70%, only ~1.7% pts lost at 50%. More match data will NOT
  meaningfully help; the bookmaker gap is an information/feature problem (injuries/news), not sample size.

## Bookmaker benchmark (real Bet365 1X2)
- Club leagues (6,288 held-out): bookmaker beats us — acc 0.507 vs 0.486, RPS 0.202 vs 0.212, pts/g 0.743 vs 0.710.
- **WC2026 national games (36, real odds collected → `data/wc_odds.csv`): NEARLY LEVEL — our 29 vs bookmaker 32
  pts, 21 vs 22 outcomes, 4 vs 5 exact.** The club gap (~5%) shrinks to ~noise on nationals = our competitive lane.
- Odds as a FEATURE closes ~⅔ of the club gap (acc →0.499) but converges just below the market; as a TARGET
  (distillation) = noise. No correct-score market is archived (betexplorer lacks it) → bookmaker exact is MARKET-DC.

## Cheap wins + autoregressive (2026-06-22)
**Seed-ensemble + calibration** (5 seeds W=15, held-out test): ensembling is a small real win — RPS ALL
0.2145→0.2130, national acc 0.586→0.596, **national pts/g 0.798→0.813**. Temperature calibration (a=0.90)
improves RPS a touch more (0.2123) but is **points-neutral** (helps the distribution, not the argmax pick).
→ ensemble worth adopting (esp. nationals); calibration = free RPS, no points. (`train_ensemble.py`)

**Autoregressive score-effects model (Version B)** (`build_segments.py` → 38,947 valid-segment matches;
`train_autoreg.py`): a learned state-multiplier table modulates per-15-min scoring by goal difference,
rolled forward via DP to a final-score distribution. **Learned a REAL state effect** — mult by diff
[≤-2,-1,0,+1,≥+2] = [1.277, 1.091, 0.992, 0.910, 0.979] (trailing teams +28%, leading teams ease off ~9%,
after the encoder controls for strength). **But prediction impact is negligible**: vs static double-Poisson
(same base rates) pts/g +0.0015 ALL / +0.007 NATL, +11 exact/4144, RPS a hair worse. Confirms the model is
at the INFORMATION ceiling — richer model classes (and Dixon-Coles ρ already covers the first-order effect)
don't move the metrics. **Not adopted.** Net: the only lever left is new FEATURES (injury/lineup/market), not
architecture; for the WC, W=15 + ensemble is the practical sweet spot.

## 5-seed ensemble baked into production (2026-06-22)
`train_goals.py --full --ensemble N` trains N full-data seeds and stores a `states` list in `goalnet.pt`
(`state`=seed-0 kept for back-compat); `predict_game.py` loads all seeds and **averages the per-match score
grids**. Production `goalnet.pt` now holds **5 seeds** (3.0 MB). Prior held-out A/B (`train_ensemble.py`):
single seed national pts/g 0.798 → ensemble **0.813**, RPS better; temperature calibration was points-neutral
so it was dropped. Inference stays instant (5 forward passes, ~5 s). Single-seed checkpoints still load
(states falls back to [state]).

## FM-attribute category ablation — "does removing parts of the FM score help?" (2026-06-22)
`train_attrcat.py`: leave-one-category-out over the 62 attrs grouped into {technical(16), mental(14),
physical(8), goalkeeping(11), hidden(13)}. Each category neutralised (standardise → set its columns to 0 =
impute to dataset mean), then the W=15 decision-focused GoalNet retrained; A/B on held-out test (n=10,457;
natl=203). Lower RPS / higher pts/g = better.

| config | ALL rps | ALL pts/g | ALL acc | NATL rps | NATL pts/g |
|---|---|---|---|---|---|
| **full** | 0.2145 | 0.7108 | 0.493 | **0.1701** | 0.7980 |
| drop technical | 0.2142 | 0.7037 | 0.490 | 0.1704 | 0.8227 |
| drop mental | 0.2151 | 0.7041 | 0.493 | 0.1771 | 0.7685 |
| drop physical | 0.2146 | 0.7029 | 0.492 | 0.1753 | 0.7783 |
| drop goalkeeping | 0.2156 | 0.6998 | 0.490 | 0.1724 | 0.7783 |
| **drop hidden** | **0.2138** | **0.7145** | **0.497** | 0.1718 | 0.7783 |

Findings: removing **technical / mental / physical / goalkeeping each HURTS** overall pts/g → all four carry
real signal (FM grades' value isn't concentrated; no category is dead weight). The one exception: dropping the
**13 hidden/personality attrs** (ambition, consistency, controversy, dirtiness, loyalty, professionalism,
temperament, versatility, …) **improves the broad set** (rps 0.2145→0.2138, pts/g 0.7108→0.7145, acc +0.4pp)
— personality grades are net noise for match prediction. **BUT on the WC-nationals lane (the target) the full
set is better by both metrics** (NATL rps 0.1701 vs 0.1718, pts/g 0.798 vs 0.778; natl n=203 = high variance).
**Decision: keep the full 62-attr set in production** — the "drop hidden" gain is on the club-heavy ALL set,
not nationals. Worth a future revisit if a larger national test set confirms hidden hurts there too.

**Age feature: not testable.** dob coverage is 1% of 192k players and no age field was scraped into snapshots,
so the 68k training matches carry no age signal to ablate. Blocked on data, not modelling.

## Phase-1 pick/betting-layer experiments (2026-06-23, no retrain)
Eval harness `experiments/eval_harness.py`: 5 train-split seeds (no test leakage) → per-seed rates cached for
val/test; full-data goalnet.pt rates cached for the 41 played WC games. Baselines: TEST-all rps 0.2130 / pg
0.7105; TEST-natl pg 0.8128; **WC played 35/41 (exact 5)**. All experiments reuse the cache (instant A/B).

**E1 empirical score-prior blend** (`e1_empirical_blend.py`): P=(1-α)·model+α·empirical, α tuned on val.
α*=0.5 → TEST-all pg 0.7105→**0.7169**, rps 0.2130→0.2121, +38 exacts (helps the club-heavy broad set). BUT
**hurts the target**: TEST-natl pg 0.8128→0.7980, **WC 35→29**. Val is club-dominated so it over-blends toward
the club score distribution; nationals score differently. **Not adopted** (same shape as drop-hidden).

**E2 exact-score calibration** (`e2_calibration.py`): joint DC ρ + sharpening γ tuned on val for pts. (ρ,γ)*=
(0.05, 2.0) → sharpening **wrecks RPS** (0.2130→0.2229, expected), flat on WC (35→35), only +0.01 natl pg /
+2 natl exact. Net wash for the target. **Not adopted.**

**E4 market blend** (`e4_market_blend.py`): de-vig wc_odds.csv 1X2 → fit market Poisson rates → blend with
model. On 36 matched WC games **model = pure-market = 32/36** (5 exact); every blend weight 0.1–0.5 also 32,
w=0.7 worse. Model and bookmaker pick the **same** scores → no edge to extract. Confirms the model already
sits at bookmaker level on nationals.

**E3 game-theoretic / contrarian picks** (`e3_gametheory.py`) — **THE WIN.** The league is a contest vs other
players: maximising E(points) (chalk EV-pick) ≠ maximising P(finish #1). MC over a field of K opponents (chalk
w/ prob q, else informed-crowd ~model grid), optimise the pick for P(sole #1). At **K=20, q=0.6**:

| policy | P(sole #1) | P(top1) | meanPts |
|---|---|---|---|
| chalk (production EV-pick) | 0.052 | 0.108 | 38.73 |
| max_exact (argmax P) | 0.091 | 0.150 | 38.67 |
| **contrarian β=0.25** | **0.159** | **0.198** | 36.75 |

Differentiating gives **~3× the chance of winning the league** (0.159 vs 0.052) for ~2 fewer expected points.
Sensitivity: when the field is **diverse** (q=0.4) chalk ≈ contrarian (differentiation moot); when the field
**clusters on chalk** (q≥0.7) contrarian/max_exact win big. Rule: **differentiate more the more you expect
opponents to clump on obvious scores.** Realized single-world backtest on the actual 41 results still favours
chalk (35 vs 32) — that's n=1; the MC is over many possible worlds. **Actionable, field-model-dependent;**
next step = collect the real league's competitor picks to calibrate q. The pick layer (not the model) is where
league points are won.

## Phase-2/4 experiments (2026-06-23)

**E6 squad experience / cohesion feature** (`experiments/e6_cohesion.py`): new context features = mean prior
starts (career apps-to-date, leakage-free) of each XI + lineup CONTINUITY (fraction of XI that also started the
team's previous match). Single-seed train-split A/B on held-out test:

| context | ALL pg | ALL rps | NATL pg | NATL exact |
|---|---|---|---|---|
| base ctx(10) | 0.7108 | 0.2145 | 0.7980 | 21 |
| **ctx+exp(14)** (apps+continuity) | 0.7024 | 0.2151 | **0.8276** | **24** |
| ctx+apps(12) (apps only) | 0.7111 | 0.2145 | 0.7980 | 21 |

Continuity carries national signal — **+0.030 natl pg, +3 exacts, the largest national-lane lift from any
feature tried** — but HURTS the broad club set (−0.008 pg, n=10,457 reliable; natl n=203 noisy). apps-only is
flat. **Candidate, not adopted**: needs WC-eval confirmation (the natl gain is small-sample, the club
regression is solid).

**E11 tournament-forward sim + leaderboard-aware adaptive risk** (`experiments/e11_adaptive.py`) — **best
strategic result.** Treat the 41 WC games as sequential rounds vs a field (K=20, q=0.6); hero adapts contrarian
β to its standing (behind the leader → raise β / gamble; ahead → β→0 / protect):

| policy | P(sole #1) | P(top1) |
|---|---|---|
| chalk (production EV-pick) | 0.056 | 0.121 |
| fixed contrarian β=0.25 | 0.156 | 0.195 |
| **ADAPTIVE (risk by rank)** | **0.206** | **0.290** |

Adaptive ≈ **4× chalk's P(win league)** and +32% over the best fixed contrarian. Confirms and extends E3: not
only differentiate, but *modulate* differentiation by your leaderboard position. Same caveat — depends on the
field model and on observing live standings (available in a real league).

**E13 national specialisation** (`experiments/e13_national_specialize.py`) — **confirms synthesis #2, biggest
model-side national gain of the night.** Single-seed train-split, held-out test:

| config | ALL pg | NATL rps | NATL pg | NATL exact |
|---|---|---|---|---|
| A baseline W=15 (production recipe) | 0.7108 | 0.1701 | 0.7980 | 21 |
| **B all → national fine-tune** (lr 5e-4) | 0.6991 | 0.1712 | **0.8424** | **25** |
| C national-only from scratch | 0.6329 | 0.1675 | 0.8030 | 21 |

Fine-tuning the all-data model on the 972 national train matches lifts the WC lane **+0.044 pg / +4 exacts**
(largest model-side national gain found), sacrificing only the club set (irrelevant for WC). Natl-only-from-
scratch (C) is worse → **transfer (pretrain-all → finetune-natl) is the recipe**, not pure specialisation.
Caveat: natl test n=203 (variance) — but the effect size is the biggest of the night and consistent with the
whole club≠national pattern. **Top productionisation candidate** for a WC-specific model (path: add a
`--natl-finetune` stage to train_goals.py --full). Pairs with the contrarian/adaptive pick layer.

### Synthesis of the night
1. **The pick layer, not the model, is where the league is won.** Chalk EV-pick is points-optimal but
   league-suboptimal; contrarian (E3) ~3× P(#1), adaptive (E11) ~4×. This is the single biggest lever found.
   Shipped as `predict_game.py --strategy chalk|exact|contrarian`; adaptive needs live standings.
2. **Club and national lanes want different things.** Drop-hidden + empirical-blend help the club-heavy set;
   continuity + sharpening help nationals. Production is measured on a club-dominated mix but the TARGET is
   nationals → a **national-specialised model / per-lane tuning** is the most promising untested model-side
   lever (beyond the existing W=15 upweight).
3. **Model is information-capped, confirmed again.** Market = model on WC (E4); every new feature either hurts
   the broad set or only nudges the noisy natl slice. Stop chasing RPS; optimise the pick policy and collect
   the real league's competitor picks to calibrate the field (q).

## Market-value & FIFA ablation study (2026-07-01) — the first feature that BEAT the baseline

Question (user): join FIFA grades / market value with FM, A/B both lanes. Data needed NO scraping — club
squad values are in `club_season_tm` (squad/top11 value, avg_age, squad_size per club-season, joins to matches
by club_id+season), and the WC2026 team files already carry per-player `market_value_eur` + `fc_rating`
(EA FC / FIFA) + `fifa_rank`/`elo` (parsed to `data/wc_team_strength.csv`, all 48 teams).

**Study 1 — CLUB lane gate** (`experiments/value_ablation.py`, value as a context feature, retrain, held-out
test; value coverage 42% of the 69k matches):

| config | ALL rps | ALL pg | ALL ex | NATL rps | NATL pg | NATL ex |
|---|---|---|---|---|---|---|
| base ctx(10) | 0.2145 | 0.7108 | 1140 | 0.1701 | 0.7980 | 21 |
| **+squad_value** | **0.2133** | **0.7115** | 1142 | 0.1721 | 0.8128 | 23 |
| +top11_value | 0.2134 | 0.7089 | 1120 | **0.1680** | 0.8276 | 24 |
| +all_value (squad+top11+age+size) | 0.2144 | 0.7069 | 1116 | 0.1757 | 0.8276 | 24 |

**`+squad_value` is the first feature in the whole project to IMPROVE the broad ALL set** (rps 0.2145→0.2133)
— every prior add (value_eur, wage, attendance, embeddings, …) hurt or was neutral. So team-level transfermarkt
market value carries real, non-redundant signal beyond FM attrs + Elo/form. `top11_value` helps nationals more
(rps 0.1680, +3 exacts) but costs club points; `all_value` over-dilutes. **Gate PASSED.**

**Study 2 — NATIONAL/WC lane** (`experiments/value_national.py`, prediction-time strength priors blended with
GoalNet on the 41 played WC games, leave-one-out calibrated; n=41 → low power, read direction):

| signal | prior-only rps / pts / exact | GoalNet+signal rps / pts / exact |
|---|---|---|
| GoalNet (FM) baseline | — | 0.1753 / 35 / 5 |
| squad_value | 0.1699 / 35 / 5 | 0.1686 / 35 / 5 |
| **avg_fc (FIFA/EA rating)** | **0.1640 / 39 / 7** | 0.1648 / 35 / 5 |
| fifa_rank | 0.1761 / 30 / 3 | 0.1737 / 30 / 3 |
| elo (team_db) | 0.2179 / 28 / 3 | 0.1839 / 32 / 4 |

**A simple FIFA-rating strength prior BEATS the full FM-GoalNet on the WC lane on every metric** (rps 0.164 vs
0.175, 39 pts vs 35, 7 exacts vs 5); market value beats it on RPS too. On data-thin international games, FM
grades under-capture team strength relative to FIFA/value — consistent with Peeters (squad value > Elo/FIFA-rank
internationally). team_db Elo is a poor standalone prior.

**UPDATE 2026-07-01 — re-run on the current 79 played games (72 group + 7 R32 ×2), the 41-game result
LARGELY REVERSED** (`experiments/combined_model.py`, multiplier-weighted; LOO-calibrated mix of GoalNet +
FIFA-prior + value-prior):

| model | rps | pts | wtd-pts | exact |
|---|---|---|---|---|
| **GoalNet only (FM)** | 0.1641 | **75** | **82** | **12** |
| GoalNet + FIFA | 0.1574 | 72 | 78 | 11 |
| GoalNet + value | 0.1615 | 70 | 76 | 10 |
| GoalNet + FIFA + value (mix) | **0.1569** | 72 | 78 | 11 |
| (in-sample best mix, ceiling) | 0.1546 | 73 | 80 | 11 |

With 79 games (vs the earlier 41-game snapshot where a FIFA prior "beat" GoalNet 39-35 pts), **GoalNet caught
up and now scores the MOST points (75 / 82-weighted) and exacts (12)**. Mixing in FIFA+value **improves RPS
(0.1641→0.1569 = better-calibrated probabilities) but NOT points** — every blend scores fewer points/exacts on
the realised games. Value adds ~nothing on top of FIFA (0.1574→0.1569). Even the overfit in-sample best mix
(weights g/f/v = 0.25/0.5/0.25) doesn't beat GoalNet on points.

**REVISED verdict:** the 41-game "adopt a FIFA prior" call was small-sample noise. On the full slate GoalNet's
*picks* are best; FIFA/value only help *probability calibration* (RPS), which matters for a market model, not
for the fantasy point-picks. **Do NOT blend for the picks.** Club-lane squad_value gain (Study 1, rps
0.2145→0.2133) stands but is conditional on 42% value coverage (missing-flag imputed elsewhere) and doesn't
reach national teams. Net: market value is the one *feature* that beat baseline in-model on clubs; but for WC
point-scoring, plain GoalNet remains the best predictor.

## Per-player value pre-test (2026-07-01) — GATE PASSED, strongest national gain yet

Before committing to a multi-day scrape of true per-player FIFA+market values for the whole 60k-player dataset,
a FREE decisive pre-test (`experiments/playerval_ablation.py`): proxy per-player market value with each
starter's CLUB squad value (club matches = match_player.club_id; NATIONAL = the player's real club via
player_snapshot.club_id, since match_player.club_id is the national team). Per-match features = [mean home-XI
club value, mean away-XI club value, diff, home cov, away cov], added as context, retrain, held-out test.

| config | ALL rps | ALL pg | NATL rps | NATL pg | NATL exact |
|---|---|---|---|---|---|
| base ctx(10) | 0.2145 | 0.7108 | 0.1701 | 0.7980 | 21 |
| **+player_value(15)** | **0.2133** | **0.7130** | 0.1734 | **0.8522** | **25** |

Coverage: ALL both-sides 42%, NATIONAL 23%. **+player_value helps the broad set (rps 0.2145→0.2133, pg +0.002,
acc +0.4pp) AND gives the LARGEST national pg lift of the whole project (+0.054, +4 exacts) — at only 23%
national coverage.** Reconciles the earlier combined-model finding: value as a prediction-time BLEND didn't beat
GoalNet's picks, but value as a TRAINED in-model feature does extract signal (a national XI of players from
top-value clubs is strong — a signal FM grades + team context miss). **GATE PASSED, especially for the WC lane.**

**Recommendation — the scrape is justified, but stage it cheaply first:** the 23% national coverage is the
ceiling on this result; raising it should amplify the gain. Cheapest high-value step = expand `club_season_tm`
to the ~583 missing clubs (a ~1000s-of-club-season scrape, NOT 60k players) → re-run at higher coverage. Only
if that keeps paying, collect true per-player transfermarkt values + FIFA (the full 60k scrape) for precision.
Caveats: NATL n=203 (noisy); 23%-coverage feature lifting the whole subset is surprising (verify at higher cov).

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
