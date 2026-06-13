# Literature: predicting match outcomes from player ratings

(Compiled 2026-06-12 by research agent; links verified at compile time.)

## Headline conclusions

- Honest pre-match H/D/A prediction tops out at **50–56% accuracy / RPS ~0.205**. Bookmaker odds ≈ 52–55% and beat nearly all published models. Claims of 65–87% = leakage (in-match stats, post-match data) — treat as red flags.
- **GBDT (XGBoost/CatBoost) ≥ deep nets** in every careful comparison. Always build the GBDT baseline first.
- **Nobody has published a serious match-prediction model on Football Manager attributes** — the 2024 survey (arXiv:2403.07669) explicitly calls FM ratings "arguably better" than FIFA's but unexploited. Our angle is novel.
- Role-based aggregation (GK/DEF/MID/ATT) of player ratings beats overall team averages; team overall ratings alone ≈ random (AUC 0.50, Yeung 2023). Defense ratings carry outsized weight (Carpita et al.).
- Combining team-level ratings (Elo/pi-ratings) with player-level ratings is significantly better than either alone (Arntzen & Hvattum 2021).
- Draws are the hard class; use Poisson goal heads or ordinal models, evaluate with RPS + log-loss, never random splits.

## Key works

1. **Danisik, Lacko, Farkas (IEEE DISA 2018)** — LSTM on FIFA attributes of lineups, 5 leagues 2011-16: **52.5%** vs ~53% bookmakers. Closest prior art. https://www.researchgate.net/publication/328313044
2. **Arntzen & Hvattum (Stat. Modelling 2021)** — ordered logit; Elo + plus-minus player ratings of actual starting XIs; combination significantly best. https://journals.sagepub.com/doi/abs/10.1177/1471082X20929881
3. **Hubáček, Šourek, Železný (Mach. Learn. 2019)** — XGBoost + pi-ratings, won 2017 Soccer Prediction Challenge: RPS 0.2054, acc 51.9%. https://link.springer.com/article/10.1007/s10994-018-5704-6
4. **Yeung, Bunker, Fujii (PLOS One 2023)** — FIFA ratings grouped into 7 categories aggregated by role + formation → two-stage XGBoost; beat odds baseline on F1; team-overall-only ≈ random. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0284318
5. **Yeung et al. (Mach. Learn. 2024)** — CatBoost (RPS 0.2085) ≈ transformer deep model (0.2098); bookmaker consensus beat both. https://link.springer.com/article/10.1007/s10994-024-06608-w
6. **Bunker, Yeung, Fujii survey (arXiv:2403.07669, 2024)** — state of the art map; FM-data gap noted. https://arxiv.org/pdf/2403.07669
7. **ACM ICSTPA 2024** — EA ratings + odds, 5 leagues × 5 seasons, best RF 52.6%. https://dl.acm.org/doi/full/10.1145/3723936.3723980
8. **Graph transformer (arXiv:2507.10626, 2025)** — player+team heterogeneous graph → pooled team representations; enables counterfactual lineup swaps. https://arxiv.org/html/2507.10626v1
9. **Kaggle European Soccer Database (hugomathien)** — canonical FIFA-ratings+lineups dataset 2008-16; community models 45-55%. https://www.kaggle.com/datasets/hugomathien/soccer
10. **BradleyGrantham/pl-predictions-using-fifa** — TF NN on FIFA ratings of starting XI, EPL. https://github.com/BradleyGrantham/pl-predictions-using-fifa

## Design implications adopted (see PLAN.md §4)

- Per-player shared MLP encoder → permutation-invariant pooling (mean+max / attention) per team → [home, away, diff, context] → head.
- Output: Poisson goal-rate heads (H/D/A derived) + softmax variant for comparison.
- Baselines mandatory: majority class, Elo logistic, CatBoost/XGBoost on role-aggregated attrs, odds argmax.
- Context features: home flag, Elo/pi-rating diff, last-5 form, rest days; odds optional (separate experiment).
- Leakage rules: FM snapshot must predate match; time-based splits; no post-match ratings.
- FM-specific novel features: hidden attrs (consistency, injury proneness, important matches, pressure, professionalism) — not present in FIFA data.
- Expectation: 50-53% ratings-only; 53-56% with context = strong result. ~1140 matches is small; regularize, prefer aggregation.
