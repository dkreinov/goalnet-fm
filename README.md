# FM Ratings → EPL Match Prediction (POC)

Football Manager player attribute ratings (scraped from unofficial sources, versioned by date) +
real EPL match data (results, lineups, stats, odds, xG) for seasons **2023-24, 2024-25, 2025-26**,
stored in SQLite, with a baseline training pipeline that predicts H/D/A from the two lineups' ratings.

Companion docs: `PLAN.md` (architecture + status log), `LITERATURE.md` (prior work + design implications).

## Data

DB: `D:\Programming\claude\FM\data\fm.db` (SQLite, schema in `schema.sql`).

| Table | Content |
|---|---|
| `match` | 1,140 EPL matches: score, HT score, referee, shots/cards/corners, Bet365+avg odds, understat xG |
| `match_player` | 45,557 appearance rows from ESPN (starter flag, position, sub status) |
| `player` / `player_source_id` | canonical players + per-source IDs (in-game UID where available) |
| `player_snapshot` | dated rating snapshots per (player, source, fm_version); new row only when values change (`attrs_hash` dedupe) |
| `player_attribute` | EAV: technical / mental / physical / set_pieces / goalkeeping / hidden, raw 1-20 scale |
| `fm_version`, `season`, `club`, `club_alias`, `source`, `scrape_log` | dimensions + audit |

Rating sources and snapshot dates:

| Source | FM version | Snapshot date | Players | Notes |
|---|---|---|---|---|
| futek.io | pre-summer-2023 export ("FM24"-branded, actually FM23-era) | 2023-06-01 | ~1,860 | raw 0-200 CA/PA, 13 hidden attrs |
| fminside db4 | FM24 24.1.0 | 2023-11-06 | partial | original release DB |
| fminside db5 | FM24 24.3.0 | 2024-02-26 | ~1,900 | winter-update DB (in-season change point) |
| fminside db6 | FMU25 community 24.4 | 2024-10-01 | 1,516 | SI cancelled FM25; community update covers 2024-25 |
| fminside db7 | FM26 26.2.0 | 2026-03-01 | 1,653 | 2025-26 |

"Score changed → saved by date" = consecutive snapshots of the same player differ only when the
attribute hash changes; `snapshot_date` orders them.

## Pipeline (run order)

```
python src/load_matches.py        # football-data.co.uk results+odds (3 CSVs)
python src/load_lineups_espn.py   # ESPN hidden API lineups (resumable, disk-cached)
python src/load_xg_understat.py   # understat xG (3 requests)
python src/scrape_fminside.py 5 7 6 4   # FM attributes per db version (resumable)
python src/scrape_futek.py        # futek FM24-branded export (adaptive 50-cap slicing)
python src/gap_fill_futek.py      # by-name search for starters missed by division enumeration
python src/build_dataset.py       # join lineups->snapshots, aggregate, Elo+form -> data/dataset.parquet
python src/train.py [--with-odds] # baselines + GBDT + MLP, time-split eval
```

All scrapers: polite rate limits (1-2.5 s/req), disk cache (`data/cache/`), resume-safe, errors to `scrape_log`.
SQLite is opened in autocommit (multiple collector processes write concurrently).

## Joining lineups to FM players

`build_dataset.resolve()`: exact normalized name → initial+lastname (both orders, handles
"Son Heung-Min"/"Heung-Min Son") → unique-token (handles "Alisson") — each step disambiguated by club
when several candidates share a key. Coverage ≈ 93% of starter appearances (rising with db5/db4 completion).
A low-CA audit caught wrong-identity matches from name-search gap-fill (e.g. amateur namesakes) — deleted.

## Results (final POC run, test = 300 matches of 2025-26, time split, no leakage)

At 1,041 complete matches (≥8/11 starters matched per side), all 5 FM versions loaded, no-odds features:

| Model | Acc | Log-loss | RPS |
|---|---|---|---|
| Majority/prior | 43.0% | 1.076 | .2315 |
| Bookmaker (B365 implied) | 49.3% | 1.014 | .2097 |
| Elo + form logistic | 52.3% | 1.028 | .2109 |
| GBDT (all features) | 47.3% | 1.071 | .2230 |
| **MLP 64-32 (ratings+context)** | **50.0%** | 1.057 | .2180 |

Matches the literature (50-56% honest ceiling; bookmaker ≈ strongest probability calibration).
With only ~740 training matches the ratings models roughly tie Elo — more data needed to separate
them; the FM-attribute signal is there (MLP > GBDT > prior at this scale).

## Phase 2 model — per-player attention + Poisson (`src/train_nn.py`)

The aggregated tabular models above throw away per-player structure. Phase 2 keeps it:

- Each starter → 98-dim vector: 47 FM attributes (1-20 scale, `/20`) + 47-dim presence mask
  (so goalkeeper-vs-outfield is distinguishable from a genuine low rating) + 4-dim position one-hot.
- A shared MLP encodes each player; **masked attention pooling** (plus mean pooling) collapses the
  ≤11 starters into a team embedding — permutation-invariant, handles variable lineup sizes.
- `[home_emb, away_emb, home−away, context]` → MLP → **two Poisson goal rates** (λ_home, λ_away).
  H/D/A probabilities come from the Poisson score matrix (independent-Poisson approximation, 0-10 goals).
- Trained with Poisson NLL, early-stopped on a held-out slice of the training seasons.

Result (same 300-match 2025-26 test, time split):

| Model | Acc | Log-loss | RPS |
|---|---|---|---|
| Majority/prior | 43.0% | 1.074 | .2314 |
| Bookmaker (B365 implied) | 49.3% | 1.014 | .2097 |
| **PlayerAttn + Poisson (NN)** | **47.0%** | 1.025 | .2141 |

Goal-rate calibration is excellent: predicted mean λ_home=1.52 / λ_away=1.34 vs actual 1.53 / 1.27.
The NN beats the prior clearly and approaches the bookmaker on RPS/log-loss with only ~740 training
matches — exactly the data-starved regime the literature warns about. Trained weights: `data/model_nn.pt`.

## Next steps

1. **More data is the lever** (literature is unanimic that ~740 matches is too few): add leagues
   (the fminside league filter generalizes — Championship, La Liga, Bundesliga, Serie A, Ligue 1),
   and back-seasons via the Kaggle FM17/FM20-23 dumps (`PLAN.md` §2). Target 5k-10k matches.
2. Bivariate-Poisson / Dixon-Coles low-score correction (models the draw inflation the independent
   approximation misses); compare an ordinal-logit head.
3. Richer player features: sub minutes weighting, congestion, FM hidden attributes (futek-only — needs
   a futek scrape for the FM26 era), finer-grained snapshot dates via Wayback captures of fminside.
4. Blend with the market: stack NN + Elo + odds; evaluate calibration (reliability curves) not just RPS.
