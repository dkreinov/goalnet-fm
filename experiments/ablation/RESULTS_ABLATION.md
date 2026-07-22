# Ablation results (generated — do not hand-edit; `run_ablation.py --report` regenerates)

Core model = calibrated scoreline distribution P(home,away); 3/1 points is a REFERENCE column, never a gate. Metrics per DESIGN.md. `grid_info` = nats of score-level information over the train-empirical-prior null (>0 = model beats the modal-score prior). `exact_lift` = EV-pick exact rate ÷ always-modal exact rate (1.0 = no better than always guessing the mode).

**Lanes.** `eval_*` = the split's pooled val∪test eval set (pooled split). `canonical_test_*` = the historical test lane (canonical split, for continuity). `wc_slate` = the frozen 104-game WC2026 benchmark, scored from TRAIN-split seeds (honest out-of-tournament view — differs from the full-data production goalnet.pt WC numbers). `_natl` restricts to national-team competitions (ids 9-15).

Δ columns (on pooled `eval_*` lanes) are vs baseline **baseline-beta3-w15**.

## baseline-beta3-w15-canonical  ·  canonical · β=3.0 W=15.0 seeds=5 ep=150 · 59.81min · `e95c7412` ⚠dirty
_canonical continuity anchor (reproduces historical TEST metrics)_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| canonical_test_all | 10457 | -0.0004 | 2.9892 | 0.2108 | 0.497 | 0.0258 | 0.92 | 0.718 |
| canonical_test_natl | 203 | 0.2337 | 3.0373 | 0.1752 | 0.571 | 0.0632 | 1.35 | 0.798 |
| wc_slate | 104 | 0.1978 | 2.9988 | 0.1553 | 0.673 | 0.1479 | 1.17 | 0.942 |

## baseline-beta3-w15  ·  pooled · β=3.0 W=15.0 seeds=5 ep=150 · 56.06min · `e95c7412` ⚠dirty
_pooled reference baseline (Phase 2+ diffs against this)_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | -0.0317 | 3.0218 | 0.2102 | 0.496 | 0.0184 | 0.92 | 0.718 |
| eval_natl | 397 | 0.1268 | 3.0683 | 0.1817 | 0.577 | 0.0379 | 1.31 | 0.816 |
| wc_slate | 104 | 0.1463 | 3.0516 | 0.1595 | 0.654 | 0.1217 | 1.17 | 0.923 |

## beta0-w15  ·  pooled · β=0.0 W=15.0 seeds=5 ep=150 · 50.93min · `edf64d8b` ⚠dirty
_pure Poisson core; purity test_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0723 (+0.1039) | 2.9178 (-0.1039) | 0.2091 (-0.0012) | 0.498 (+0.002) | 0.0139 (-0.0045) | 0.92 (-0.005) | 0.718 (+0.001) |
| eval_natl | 397 | 0.2522 (+0.1254) | 2.9428 (-0.1254) | 0.1777 (-0.0041) | 0.582 (+0.005) | 0.0344 (-0.0035) | 1.22 (-0.083) | 0.804 (-0.013) |
| wc_slate | 104 | 0.2811 | 2.9169 | 0.1528 | 0.683 | 0.1194 | 1.00 | 0.913 |

## beta1-w15  ·  pooled · β=1.0 W=15.0 seeds=5 ep=150 · 63.34min · `edf64d8b` ⚠dirty
_small decision term; regularizer test_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0550 (+0.0867) | 2.9351 (-0.0867) | 0.2096 (-0.0006) | 0.497 (+0.001) | 0.0096 (-0.0088) | 0.97 (+0.048) | 0.700 (-0.018) |
| eval_natl | 397 | 0.2435 (+0.1167) | 2.9515 (-0.1167) | 0.1798 (-0.0019) | 0.577 (+0.000) | 0.0475 (+0.0096) | 1.31 (+0.000) | 0.786 (-0.030) |
| wc_slate | 104 | 0.2853 | 2.9127 | 0.1542 | 0.702 | 0.1458 | 1.17 | 0.904 |

## decay-hl2  ·  pooled · β=3.0 W=15.0 seeds=5 ep=150 · 22.53min · `a8377f24` ⚠dirty
_time-decay half-life 2y_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0077 (+0.0394) | 2.9824 (-0.0394) | 0.2102 (+0.0000) | 0.497 (+0.001) | 0.0207 (+0.0023) | 0.92 (-0.006) | 0.717 (-0.001) |
| eval_natl | 397 | 0.1925 (+0.0658) | 3.0025 (-0.0658) | 0.1802 (-0.0015) | 0.579 (+0.003) | 0.0354 (-0.0025) | 1.28 (-0.028) | 0.809 (-0.008) |
| wc_slate | 104 | 0.2026 | 2.9953 | 0.1555 | 0.663 | 0.1117 | 1.17 | 0.933 |

## decay-hl4  ·  pooled · β=3.0 W=15.0 seeds=5 ep=150 · 37.69min · `a8377f24` ⚠dirty
_time-decay half-life 4y_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0103 (+0.0420) | 2.9798 (-0.0420) | 0.2099 (-0.0003) | 0.497 (+0.001) | 0.0184 (-0.0000) | 0.93 (+0.004) | 0.720 (+0.002) |
| eval_natl | 397 | 0.1714 (+0.0446) | 3.0236 (-0.0446) | 0.1819 (+0.0002) | 0.567 (-0.010) | 0.0403 (+0.0024) | 1.17 (-0.139) | 0.781 (-0.035) |
| wc_slate | 104 | 0.1768 | 3.0211 | 0.1607 | 0.654 | 0.1332 | 1.08 | 0.904 |

## decay-hl8  ·  pooled · β=3.0 W=15.0 seeds=5 ep=150 · 38.64min · `a8377f24` ⚠dirty
_time-decay half-life 8y_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0195 (+0.0512) | 2.9706 (-0.0512) | 0.2106 (+0.0004) | 0.496 (+0.000) | 0.0199 (+0.0015) | 0.95 (+0.025) | 0.713 (-0.005) |
| eval_natl | 397 | 0.2072 (+0.0804) | 2.9879 (-0.0804) | 0.1807 (-0.0010) | 0.572 (-0.005) | 0.0269 (-0.0110) | 1.19 (-0.111) | 0.768 (-0.048) |
| wc_slate | 104 | 0.2061 | 2.9919 | 0.1591 | 0.673 | 0.1495 | 1.08 | 0.894 |

## beta3-w1  ·  pooled · β=3.0 W=1.0 seeds=5 ep=150 · 44.36min · `a8377f24` ⚠dirty
_no national upweight; null test_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0053 (+0.0370) | 2.9848 (-0.0370) | 0.2094 (-0.0008) | 0.498 (+0.002) | 0.0111 (-0.0073) | 0.93 (+0.009) | 0.720 (+0.003) |
| eval_natl | 397 | 0.2126 (+0.0858) | 2.9825 (-0.0858) | 0.1797 (-0.0020) | 0.572 (-0.005) | 0.0492 (+0.0113) | 1.14 (-0.167) | 0.763 (-0.053) |
| wc_slate | 104 | 0.2701 | 2.9279 | 0.1538 | 0.663 | 0.1081 | 1.08 | 0.913 |

## beta3-w40  ·  pooled · β=3.0 W=40.0 seeds=5 ep=150 · 41.14min · `a8377f24` ⚠dirty
_heavy national upweight_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | -0.0041 (+0.0276) | 2.9942 (-0.0276) | 0.2111 (+0.0008) | 0.494 (-0.002) | 0.0206 (+0.0022) | 0.94 (+0.019) | 0.710 (-0.008) |
| eval_natl | 397 | 0.1725 (+0.0457) | 3.0225 (-0.0457) | 0.1842 (+0.0025) | 0.562 (-0.015) | 0.0419 (+0.0040) | 1.25 (-0.056) | 0.783 (-0.033) |
| wc_slate | 104 | 0.1750 | 3.0230 | 0.1610 | 0.663 | 0.1496 | 1.00 | 0.865 |

## combo-beta0-w1  ·  pooled · β=0.0 W=1.0 seeds=5 ep=150 · 41.41min · `0687a97c` ⚠dirty
_combo: pure-Poisson (beta0) + no-upweight (W1) - do the de-biasing levers stack?_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0762 (+0.1079) | 2.9139 (-0.1079) | 0.2089 (-0.0013) | 0.497 (+0.001) | 0.0152 (-0.0032) | 0.96 (+0.036) | 0.711 (-0.007) |
| eval_natl | 397 | 0.2432 (+0.1164) | 2.9519 (-0.1164) | 0.1809 (-0.0008) | 0.569 (-0.008) | 0.0540 (+0.0161) | 1.33 (+0.028) | 0.796 (-0.020) |
| wc_slate | 104 | 0.2992 | 2.8988 | 0.1548 | 0.683 | 0.1182 | 1.17 | 0.933 |

## combo-beta0-w1-decay8  ·  pooled · β=0.0 W=1.0 seeds=5 ep=150 · 46.84min · `0687a97c` ⚠dirty
_3-lever: beta0 + W1 + decay hl8_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0779 (+0.1096) | 2.9122 (-0.1096) | 0.2089 (-0.0013) | 0.496 (-0.000) | 0.0127 (-0.0057) | 0.97 (+0.046) | 0.715 (-0.003) |
| eval_natl | 397 | 0.2477 (+0.1209) | 2.9473 (-0.1209) | 0.1802 (-0.0016) | 0.574 (-0.003) | 0.0526 (+0.0147) | 1.31 (+0.000) | 0.793 (-0.023) |
| wc_slate | 104 | 0.3072 | 2.8908 | 0.1551 | 0.673 | 0.1233 | 1.08 | 0.913 |

## combo-beta0-w1-canon  ·  canonical · β=0.0 W=1.0 seeds=5 ep=150 · 46.26min · `0687a97c` ⚠dirty
_beta0 + W1 canonical continuity_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| canonical_test_all | 10457 | 0.0731 | 2.9157 | 0.2092 | 0.495 | 0.0069 | 0.89 | 0.710 |
| canonical_test_natl | 203 | 0.2968 | 2.9742 | 0.1692 | 0.591 | 0.0642 | 1.29 | 0.808 |
| wc_slate | 104 | 0.2905 | 2.9062 | 0.1564 | 0.683 | 0.1347 | 1.25 | 0.971 |

## combo-beta0-w1-s10  ·  pooled · β=0.0 W=1.0 seeds=10 ep=150 · 75.27min · `0687a97c` ⚠dirty
_beta0 + W1 robustness 10-seed_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0764 (+0.1081) | 2.9137 (-0.1081) | 0.2087 (-0.0015) | 0.498 (+0.002) | 0.0135 (-0.0049) | 0.95 (+0.031) | 0.712 (-0.006) |
| eval_natl | 397 | 0.2414 (+0.1146) | 2.9536 (-0.1146) | 0.1806 (-0.0011) | 0.572 (-0.005) | 0.0355 (-0.0024) | 1.31 (+0.000) | 0.791 (-0.025) |
| wc_slate | 104 | 0.2937 | 2.9043 | 0.1554 | 0.683 | 0.1256 | 1.17 | 0.933 |

## ctx-momentum  ·  pooled · β=0.0 W=1.0 seeds=5 ep=150 · 47.63min · `95e3822d` ⚠dirty
_Elo-momentum + form-trajectory bundle vs beta0-W1_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0734 (+0.1051) | 2.9167 (-0.1051) | 0.2090 (-0.0012) | 0.496 (-0.000) | 0.0181 (-0.0003) | 1.00 (+0.073) | 0.704 (-0.014) |
| eval_natl | 397 | 0.2388 (+0.1120) | 2.9563 (-0.1120) | 0.1809 (-0.0008) | 0.579 (+0.003) | 0.0414 (+0.0035) | 1.39 (+0.083) | 0.793 (-0.023) |

## market-blend-b1  ·  pooled · β=0.0 W=1.0 seeds=0 ep=0 · 0.17min · `ad349090` ⚠dirty
_Arm B1: inference blend of combo-beta0-w1 grids toward de-vigged market_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|

## ctx-odds  ·  pooled · β=0.0 W=1.0 seeds=5 ep=150 · 47.24min · `8eda7a9d` ⚠dirty
_Arm A: Shin-devigged closing 1X2 (38k club from DB + 738 natl scraped) vs beta0-W1_

| lane | n | grid_info | grid_nll | rps | acc | ece_outcome | exact_lift | pts_g_31 |
|---|---|---|---|---|---|---|---|---|
| eval_all | 21663 | 0.0842 (+0.1159) | 2.9059 (-0.1159) | 0.2063 (-0.0039) | 0.503 (+0.007) | 0.0165 (-0.0019) | 0.98 (+0.061) | 0.714 (-0.004) |
| eval_natl | 397 | 0.2309 (+0.1041) | 2.9642 (-0.1041) | 0.1803 (-0.0014) | 0.574 (-0.003) | 0.0650 (+0.0271) | 1.14 (-0.167) | 0.758 (-0.058) |

---

# Diagnostics

## Diagnostics: baseline-beta3-w15

Prior null = train-split empirical score grid; modal scoreline = **1-1** (P=0.123). DC rho=0.05. Does the model add score-level information, or coast on the modal-score prior?

**(1) Score-level information — grid-NLL vs the empirical-prior null**

| lane | n | grid_nll | grid_nll_prior | grid_info (nats) | sharpness (nats) | ece |
|---|---|---|---|---|---|---|
| eval_all | 21663 | 3.0218 | 2.9901 | -0.0317 | 2.607 | 0.0184 |
| eval_natl | 397 | 3.0683 | 3.1950 | +0.1268 | 2.552 | 0.0379 |
| wc_slate | 104 | 3.0516 | 3.1980 | +0.1463 | 2.624 | 0.1217 |

**(2) Per-scoreline lift — eval_natl.** EV-pick precision vs prior cell prob (off-modal picks with precision > prior_p = genuine score information):

| EV-pick | picked | hits | precision | prior_p | off-modal? |
|---|---|---|---|---|---|
| 1-0 | 241 | 31 | 0.129 | 0.100 | yes |
| 0-1 | 118 | 8 | 0.068 | 0.079 | yes |
| 2-0 | 19 | 3 | 0.158 | 0.073 | yes |
| 0-2 | 15 | 4 | 0.267 | 0.048 | yes |
| 0-0 | 2 | 1 | 0.500 | 0.076 | yes |
| 0-4 | 1 | 0 | 0.000 | 0.008 | yes |
| 0-3 | 1 | 0 | 0.000 | 0.022 | yes |

Top-3-mass recall by true scoreline — eval_natl:

| true score | n | top3_recall |
|---|---|---|
| 1-0 | 39 | 0.872 |
| 2-0 | 37 | 0.514 |
| 1-1 | 36 | 0.278 |
| 0-0 | 34 | 0.853 |
| 2-1 | 28 | 0.000 |
| 0-1 | 24 | 0.625 |
| 1-2 | 23 | 0.000 |
| 0-2 | 22 | 0.455 |
| 2-2 | 19 | 0.000 |
| 3-0 | 17 | 0.471 |

**(2) Per-scoreline lift — wc_slate.** EV-pick precision vs prior cell prob (off-modal picks with precision > prior_p = genuine score information):

| EV-pick | picked | hits | precision | prior_p | off-modal? |
|---|---|---|---|---|---|
| 1-0 | 68 | 7 | 0.103 | 0.100 | yes |
| 0-1 | 25 | 6 | 0.240 | 0.079 | yes |
| 2-0 | 8 | 0 | 0.000 | 0.073 | yes |
| 0-2 | 3 | 1 | 0.333 | 0.048 | yes |

Top-3-mass recall by true scoreline — wc_slate:

| true score | n | top3_recall |
|---|---|---|
| 1-1 | 12 | 0.333 |
| 2-1 | 9 | 0.000 |
| 2-0 | 8 | 0.375 |
| 0-1 | 8 | 0.875 |
| 0-0 | 8 | 0.125 |
| 1-0 | 7 | 1.000 |
| 3-1 | 5 | 0.000 |
| 3-0 | 5 | 0.600 |
| 3-2 | 5 | 0.000 |
| 1-2 | 5 | 0.000 |

**(3) Calibration reliability — eval_natl** (outcome max-prob & exact-cell):

| kind | pred_bin | mean_pred | observed | n |
|---|---|---|---|---|
| outcome | 0.01-0.15 | 0.097 | 0.094 | 149 |
| outcome | 0.15-0.22 | 0.185 | 0.134 | 149 |
| outcome | 0.22-0.27 | 0.245 | 0.201 | 149 |
| outcome | 0.27-0.30 | 0.283 | 0.318 | 148 |
| outcome | 0.30-0.32 | 0.307 | 0.255 | 149 |
| outcome | 0.32-0.42 | 0.363 | 0.396 | 149 |
| outcome | 0.42-0.57 | 0.492 | 0.483 | 149 |
| outcome | 0.57-0.97 | 0.694 | 0.785 | 149 |
| exact | 0.14-0.16 | 0.149 | 0.080 | 50 |
| exact | 0.16-0.16 | 0.160 | 0.082 | 49 |
| exact | 0.16-0.17 | 0.168 | 0.200 | 50 |
| exact | 0.17-0.18 | 0.175 | 0.082 | 49 |
| exact | 0.18-0.18 | 0.181 | 0.040 | 50 |
| exact | 0.18-0.19 | 0.188 | 0.102 | 49 |
| exact | 0.19-0.20 | 0.195 | 0.140 | 50 |
| exact | 0.20-0.23 | 0.208 | 0.220 | 50 |

**(4) Verdict.**
On the national lane the model adds **+0.1268 nats** of score-level information over the modal-score prior (grid_nll 3.068 vs prior 3.195); exact-score lift **1.31×** the always-modal rate; outcome ECE 0.038; sharpness 2.55 nats. Off-modal EV-picks (397 of them) hit at precision **0.118** vs mean prior cell prob 0.058 — the model is extracting genuine off-modal score signal. WC-slate grid_info +0.1463, exact_lift 1.17×. On the broad all-competitions lane, by contrast, grid_info is **-0.0317** (exact_lift 0.92×) — the model does NOT beat the empirical prior at the exact-cell level there: its score-level edge is concentrated on the national/WC lanes it is upweighted (W) for, which is the intended target. So the network is not merely echoing the modal score on the games that matter, but it adds little scoreline information on club-heavy fixtures.


## Diagnostics: combo-beta0-w1

Prior null = train-split empirical score grid; modal scoreline = **1-1** (P=0.123). DC rho=-0.1. Does the model add score-level information, or coast on the modal-score prior?

**(1) Score-level information — grid-NLL vs the empirical-prior null**

| lane | n | grid_nll | grid_nll_prior | grid_info (nats) | sharpness (nats) | ece |
|---|---|---|---|---|---|---|
| eval_all | 21663 | 2.9139 | 2.9901 | +0.0762 | 2.907 | 0.0152 |
| eval_natl | 397 | 2.9519 | 3.1950 | +0.2432 | 2.931 | 0.0540 |
| wc_slate | 104 | 2.8988 | 3.1980 | +0.2992 | 3.018 | 0.1182 |

**(2) Per-scoreline lift — eval_natl.** EV-pick precision vs prior cell prob (off-modal picks with precision > prior_p = genuine score information):

| EV-pick | picked | hits | precision | prior_p | off-modal? |
|---|---|---|---|---|---|
| 1-0 | 110 | 15 | 0.136 | 0.100 | yes |
| 2-0 | 75 | 11 | 0.147 | 0.073 | yes |
| 0-1 | 45 | 3 | 0.067 | 0.079 | yes |
| 1-1 | 40 | 2 | 0.050 | 0.123 |  |
| 0-2 | 40 | 8 | 0.200 | 0.048 | yes |
| 2-1 | 38 | 5 | 0.132 | 0.088 | yes |
| 1-2 | 34 | 2 | 0.059 | 0.070 | yes |
| 3-0 | 12 | 2 | 0.167 | 0.039 | yes |
| 0-3 | 3 | 0 | 0.000 | 0.022 | yes |

Top-3-mass recall by true scoreline — eval_natl:

| true score | n | top3_recall |
|---|---|---|
| 1-0 | 39 | 0.641 |
| 2-0 | 37 | 0.432 |
| 1-1 | 36 | 0.889 |
| 0-0 | 34 | 0.382 |
| 2-1 | 28 | 0.286 |
| 0-1 | 24 | 0.375 |
| 1-2 | 23 | 0.304 |
| 0-2 | 22 | 0.409 |
| 2-2 | 19 | 0.000 |
| 3-0 | 17 | 0.353 |

**(2) Per-scoreline lift — wc_slate.** EV-pick precision vs prior cell prob (off-modal picks with precision > prior_p = genuine score information):

| EV-pick | picked | hits | precision | prior_p | off-modal? |
|---|---|---|---|---|---|
| 2-1 | 27 | 4 | 0.148 | 0.088 | yes |
| 2-0 | 24 | 0 | 0.000 | 0.073 | yes |
| 1-2 | 15 | 4 | 0.267 | 0.070 | yes |
| 1-0 | 13 | 2 | 0.154 | 0.100 | yes |
| 0-2 | 11 | 2 | 0.182 | 0.048 | yes |
| 3-0 | 6 | 1 | 0.167 | 0.039 | yes |
| 0-1 | 4 | 1 | 0.250 | 0.079 | yes |
| 1-1 | 3 | 0 | 0.000 | 0.123 |  |
| 0-3 | 1 | 0 | 0.000 | 0.022 | yes |

Top-3-mass recall by true scoreline — wc_slate:

| true score | n | top3_recall |
|---|---|---|
| 1-1 | 12 | 0.750 |
| 2-1 | 9 | 0.778 |
| 2-0 | 8 | 0.500 |
| 0-1 | 8 | 0.250 |
| 0-0 | 8 | 0.125 |
| 1-0 | 7 | 0.429 |
| 3-1 | 5 | 0.000 |
| 3-0 | 5 | 0.400 |
| 3-2 | 5 | 0.000 |
| 1-2 | 5 | 1.000 |

**(3) Calibration reliability — eval_natl** (outcome max-prob & exact-cell):

| kind | pred_bin | mean_pred | observed | n |
|---|---|---|---|---|
| outcome | 0.02-0.16 | 0.103 | 0.054 | 149 |
| outcome | 0.16-0.23 | 0.196 | 0.188 | 149 |
| outcome | 0.23-0.26 | 0.249 | 0.201 | 149 |
| outcome | 0.26-0.29 | 0.277 | 0.243 | 148 |
| outcome | 0.29-0.33 | 0.299 | 0.315 | 149 |
| outcome | 0.33-0.43 | 0.371 | 0.383 | 149 |
| outcome | 0.43-0.56 | 0.481 | 0.523 | 149 |
| outcome | 0.56-0.92 | 0.691 | 0.758 | 149 |
| exact | 0.09-0.11 | 0.109 | 0.140 | 50 |
| exact | 0.11-0.12 | 0.116 | 0.122 | 49 |
| exact | 0.12-0.12 | 0.120 | 0.140 | 50 |
| exact | 0.12-0.13 | 0.124 | 0.061 | 49 |
| exact | 0.13-0.13 | 0.130 | 0.100 | 50 |
| exact | 0.13-0.14 | 0.134 | 0.122 | 49 |
| exact | 0.14-0.14 | 0.137 | 0.120 | 50 |
| exact | 0.14-0.14 | 0.141 | 0.060 | 50 |

**(4) Verdict.**
On the national lane the model adds **+0.2432 nats** of score-level information over the modal-score prior (grid_nll 2.952 vs prior 3.195); exact-score lift **1.33×** the always-modal rate; outcome ECE 0.054; sharpness 2.93 nats. Off-modal EV-picks (357 of them) hit at precision **0.129** vs mean prior cell prob 0.065 — the model is extracting genuine off-modal score signal. WC-slate grid_info +0.2992, exact_lift 1.17×. On the broad all-competitions lane, by contrast, grid_info is **+0.0762** (exact_lift 0.96×) — the model modestly beats the prior there: its score-level edge is concentrated on the national/WC lanes it is upweighted (W) for, which is the intended target. So the network is not merely echoing the modal score on the games that matter, but it adds little scoreline information on club-heavy fixtures.

