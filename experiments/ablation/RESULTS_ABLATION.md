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
