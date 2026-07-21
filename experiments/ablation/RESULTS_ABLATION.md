# Ablation results (generated — do not hand-edit; `run_ablation.py --report` regenerates)

Core model = calibrated scoreline distribution P(home,away); 3/1 points is a REFERENCE column, never a gate. Metrics per DESIGN.md. `grid_info` = nats of score-level information over the train-empirical-prior null (>0 = model beats the modal-score prior). `exact_lift` = EV-pick exact rate ÷ always-modal exact rate (1.0 = no better than always guessing the mode).

**Lanes.** `eval_*` = the split's pooled val∪test eval set (pooled split). `canonical_test_*` = the historical test lane (canonical split, for continuity). `wc_slate` = the frozen 104-game WC2026 benchmark, scored from TRAIN-split seeds (honest out-of-tournament view — differs from the full-data production goalnet.pt WC numbers). `_natl` restricts to national-team competitions (ids 9-15).

Δ columns (on pooled `eval_*` lanes) are vs baseline **baseline-beta3-w15** (not yet registered).
