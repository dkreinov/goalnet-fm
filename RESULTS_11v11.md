# 11v11 result model — results

Predict match result (Home / Draw / Away) from the 22 starters' FM attribute vectors, ordered by role
(GK → DEF → MID → ATT). Built and swept 2026-06-20.

## Data
`data/players.npz` (62 attrs) / `data/players2.npz` (66: + log value_eur, has_value, log wage_eur, has_wage).
Both = **48,355 matches** where BOTH starting XIs are fully FM-graded. Per starter: attribute vector +
role id; players ordered GK→DEF→MID→ATT per team.
Builders: `src/build_player_dataset.py`, `src/build_player_dataset2.py`.

Time split (no leakage): train < 2024-08 (33,461) · val 2024-25 (7,833) · test ≥ 2025-08 (7,061).

## Model
Per-player MLP encoder + role embedding → pool per role → team vector → `[home, away, advantage]` head
→ 3-class result. AdamW, cosine LR, dropout, early-stop on val RPS. Code: `src/train_pos.py` (single),
`src/train_pos2.py` (sweep: `--arch mean,attn,diff,xfmr --seeds N --npz`).

Pooling variants: `mean` (per-role mean), `attn` (per-role learned-query attention), `diff` (mean + the
head sees `[h-a, h*a, h, a]`), `xfmr` (2-layer self-attention over the 11, then pool).

## Results (test set, RPS = ranked probability score, lower better)

| model                         | acc    | logloss | RPS    |
|-------------------------------|--------|---------|--------|
| majority/prior baseline       | 0.4421 | 1.0715  | 0.2288 |
| mean ensemble (62)            | 0.4824 | 1.0286  | 0.2140 |
| attn ensemble (62)            | 0.4883 | 1.0281  | 0.2138 |
| diff ensemble (62)            | 0.4863 | 1.0281  | 0.2140 |
| **xfmr ensemble (62)**        | 0.4896 | 1.0265  | **0.2132** |
| xfmr ensemble (66, +value)    | 0.4858 | 1.0269  | 0.2135 |

Ensemble = mean of softmax over the seeds.

## Findings
- **xfmr (transformer over the 11) wins**, but barely: all architectures cluster 0.213–0.214. The
  pooling choice matters little — lineup attributes are near their own information ceiling.
- **Seed-ensembling helps** ~0.0006–0.0008 RPS, consistently.
- **Market value / wage add nothing** (66-feat val improved to 0.2119 but test 0.2135 ≈ flat). The 62
  attrs already encode player quality; the sparse missing-indicators add source-noise.
- Best model (xfmr, 0.2132) beats majority (0.2288) clearly and sits a bit above the literature pre-match
  ceiling (~0.205 RPS). Remaining gap is real match-outcome variance, not model capacity.

## Where the gains likely are (next, not yet done)
1. **Team-strength priors** the lineup can't see: Elo / recent form / squad-value-from-`club_season_tm`
   (needs club-id bridging across spaces). This is the standard lever that moves football RPS most.
2. **Calibration** (temperature / Dirichlet) on val — cheap RPS win.
3. **Stadium / attendance / rest-days** context features (feature-ablation study, per roadmap).
