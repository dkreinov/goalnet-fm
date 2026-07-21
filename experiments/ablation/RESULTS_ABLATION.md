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

