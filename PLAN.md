# FM Ratings → Match Outcome Prediction: Plan

Goal: collect Football Manager (FM) player attribute ratings (unofficial web sources, up to 20 years back),
versioned by date, plus real match data (results, lineups, stats, odds, xG), to later train a neural network
that predicts match outcomes from the two teams' player ratings.

POC scope: English Premier League, seasons 2023-24 / 2024-25 / 2025-26 — all clubs, all players,
all attribute categories, attribute changes saved by date.

## 1. Data layer (DONE: schema)

SQLite at `D:\Programming\claude\FM\data\fm.db`, schema `D:\Programming\claude\FM\schema.sql`.

- `player_snapshot` + `player_attribute` (EAV): one dated snapshot per (player, source, fm_version);
  new row only when attribute hash changes → "score changed → saved by date" requirement.
- `match` (results + shots/cards/corners + Bet365/avg odds + xG columns) — LOADED: 1140 EPL matches, 3 seasons.
- `match_player` — lineups (starter/sub, minutes, position, post-match rating).
- Cross-source identity: `player.norm_name + dob`, `player_source_id` per-source IDs, `club_alias` for club name variants.

## 2. Sources (research DONE — 5 agents + direct probing)

| Purpose | Source | Verdict |
|---|---|---|
| FM attributes (PRIMARY) | **fminside.net** | Full 1-20 attrs + CA/PA (0-99 norm). 4 DB versions: db4=FM24 orig (2023-11), db5=FM24.3 winter (2024-02), db6=FMU25 community (2024-25), db7=FM26 26.2 (2025-26). Session-based filter API verified. SCRAPING. |
| FM attributes (SECONDARY) | **futek.io** | FM24 launch export, raw 0-200 CA/PA, 60 attrs **incl. 13 hidden** (consistency, injury proneness, pressure...). 50-row search cap → adaptive CA/age slicing. SCRAPING. |
| FM attributes (rejected) | sortitoutsi.net | NO numeric attrs/CA-PA publicly (stars only). Useful only for UID harvesting + dated transfer logs. |
| FM historical bulk | Kaggle: FM17, FM20-23 full CSVs (furkanuluta, siddhrajthakor, ajinkyablaze); GitHub FM23 JSON (downloaded, 469 players) | For 20-year extension. No FM24/FM26 dumps exist. Pre-FM17: no bulk source found anywhere. |
| Results+odds | football-data.co.uk | DONE (1140 matches, stats + B365/avg odds). |
| Lineups | **ESPN hidden API** (site.api.espn.com) | Verified: 11 starters + 9 bench per team, sub flags, formation place, referee, odds. LOADING. |
| xG | **understat.com** getLeagueData/getMatchData | DONE (xG for 1140/1140 matches). Per-player xG/xA available via getMatchData if needed. |
| Lineups (rejected) | fbref (Cloudflare 403), fotmob (signed-header auth), API-Football (100 req/day) | — |

Dated-snapshot story for "score changed → save by date": fminside db4 (2023-11) vs futek (2023-12) vs
fminside db5 (2024-02) give three dated FM24 points; db6 (2024-25); db7 (2025-26). Optional extension:
Wayback Machine captures of fminside player pages for finer-grained history.

## 3. Pipeline

1. Scraper per source (`src/scrape_<source>.py`) using `src/fetch.py` (cache, retries, rate limit ~1.5s/host).
2. For each (season, db version): enumerate EPL clubs → squads → player pages → parse all attribute
   categories (technical, mental, physical, goalkeeping; CA/PA; value/wage; position; physicals).
3. `db.save_snapshot` dedupes identical consecutive snapshots; changed values get new dated rows.
4. Lineups: per match, starting XI + subs + minutes; join to FM players via normalized name + club + dob fallback.
5. Entity resolution report: % of lineup players matched to FM snapshots (target >95%).

## 4. NN design (draft — refine after literature agent)

Features per match:
- Two teams × 11 starters (+ subs weighted by expected minutes), each player = vector of ~50 FM attributes
  taken from the snapshot **latest before match date** (no leakage from later db updates).
- Aggregations baseline: per-team mean/max by attribute group + positional buckets (GK/DEF/MID/ATT).
- Context features: home flag, rest days, recent form (last 5 W/D/L, goals), Elo, bookmaker odds (optional —
  with odds the model learns the market; without, tests pure ratings signal).

Models, in order:
1. Baselines: majority class (~43-46% home win), bookmaker odds argmax (~53-55%) — must beat or match.
2. Gradient boosting (XGBoost) on aggregated features — strong tabular baseline.
3. MLP on aggregated team vectors.
4. Set-based: per-player MLP → attention/mean pooling per team → match head (handles variable lineups).
Targets: 3-class H/D/A (primary); Poisson goals head (secondary).
Validation: time-based split (train 2023-25, test 2025-26). Never random split — leakage.

Realistic ceiling: literature reports ~50-58% accuracy on H/D/A; draws hardest. ~1140 matches is small —
regularize hard, prefer aggregated features over raw 22×50 input.

## 5. Risks

- Cloudflare/anti-bot on FM sites → fallback: Kaggle/GitHub dumps, browser automation.
- 2024-25 season has no dedicated FM game (FM25 cancelled) → use FM24 final db + FM26 initial db as brackets.
- Name matching FM↔lineups (diacritics, "J. Smith" forms) → norm() + dob + club context; manual alias table.
- Only ~1140 matches for POC → NN may not beat XGBoost; plan to extend leagues/years later (hence 20-year source hunt).

## 5b. Multi-league + national-team expansion (2026-06-13)

Goal expanded: collect as many leagues as possible by global rank, 6 seasons (2020-21..2025-26),
plus national teams; flag unmatched names for later resolution.

Key design decisions:
- **Season → FM database**: each season maps to its own fminside db (FM21=db1 … FM26=db7), one
  grade snapshot per season. `leagues.SEASON_DB`. build_dataset now joins a match to *its season's*
  FM database (not "latest snapshot before match date", which broke because FM26's snapshot date is
  mid-season-2026, after most 2025-26 matches).
- **Cross-source club identity via score-linking**: football-data abbreviates club names ("Ath
  Madrid"), ESPN/understat use full names → naive club-name matching fragments identity. `match_link.py`
  aligns foreign matches to football-data `match` rows by (competition, date±2d, exact score), with
  club-name token similarity only as a tiebreaker. xG re-match: 10,732 linked, 1 miss (was ~3,600 misses
  with name matching). ESPN lineups inherit football-data club identity the same way.
- **Registry-driven** (`leagues.py`): each league carries fminside league string (verified live) +
  nationality collision-filter + ESPN code + football-data code + understat name. 8 leagues enabled
  (EPL, LaLiga, Serie A, Bundesliga, Ligue 1, Championship, Eredivisie, Primeira Liga); ranks 9-15
  staged pending fminside-string verification.
- **Unmatched names**: `unmatched_name` table flags lineup players that don't join to an FM snapshot
  (recorded in build_dataset), for later manual/auto resolution.
- **National teams** (`load_national_espn.py`): ESPN is primary (no football-data internationals);
  creates national-team clubs, tags match_kind='national', maps each match to its football season so
  players join to that season's FM db by name. Covers WC/Euro/Nations League/Copa/qualifiers/friendlies.

Loaded so far: 17,717 league matches (8 leagues × 6 seasons), xG for top-5 (10,732 matches).
ESPN lineups + fminside grades scraping in background.

## 6b. Expansion progress log

- 2026-06-13 17:04: EPL FM grades complete for ALL 6 editions (FM21-FM26). Dataset rebuild:
  **2,252 complete matches** across 6 seasons (2020-21:343, 21-22:383, 22-23:382, 23-24:381,
  24-25:382, 25-26:381) — up from 1,041 (POC). Overall starter-join coverage 54.7% (low only because
  non-EPL leagues have lineups but FM grades still scraping). ESPN lineups at ~7,600 matches (through
  Ligue 1; Championship/Eredivisie/Primeira Liga remain). fminside on EPL FM21 (db1, last EPL edition),
  then La Liga/Serie A/Bundesliga/Ligue 1 × 6 editions.

- 2026-06-13 21:55: ESPN club lineups DONE (16,153 matches w/ lineups, 8 leagues × 6 seasons,
  only 66 unlinkable). National teams DONE: 2,707 matches, 2,375 w/ lineups (Friendlies 1478,
  Nations League 502, WC-Q UEFA 442, Euros 102, World Cup 64, Copa 60, WC-Q CONMEBOL 59).
  Verified national→FM-grade join: England (EPL players) 11/11 starters graded; smaller nations
  partial until their leagues scraped. fminside still on La Liga editions (FM24/db5).

- 2026-06-14 04:30: **7,240 complete matches** (>=8 graded starters/side), up from 2,252 (EPL-only)
  and 1,041 (POC). All 6 seasons: 2020-21:1031, 21-22:1095, 22-23:1219, 23-24:1294, 24-25:1327, 25-26:1274.
  By competition: EPL 2275 (full), La Liga 2272 (full), Serie A 1711 (db5 scraping), Championship 565
  (via cross-league players — own grades pending 2nd run), Ligue 1 89 / Bundesliga 29 / Eredivisie 14 /
  Primeira 5 (grades pending), national 205 (England/etc. with EPL+La Liga players). Overall starter→FM
  join 66.3%. 10,863 players with snapshots. fminside single worker on Serie A FM24; Bundesliga+Ligue1
  to follow, then 2nd run for Championship/Eredivisie/Primeira Liga, then rank 9+ leagues.

- 2026-06-14 08:00: ESPN match-context + event backfill DONE (from cache, zero new network): 20,388
  matches with kickoff_time + attendance, 19,399 venue, 19,623 formations, referee doubled to 15,316
  (incl. internationals). match_event table: 390,192 minute-stamped events (goals/cards/subs) — feeds
  the trajectory/autoregressive model. Schema v3 added: match context cols, match_event, manager +
  manager_snapshot/attribute, club_attribute tables. Manager + club-reputation scraper to run after
  player grades (fminside busy). Grade worker stalled on a timeout burst (Serie A db3) → restarted.

- 2026-06-14 12:00: KAGGLE BULK + FULL-ESPN-LEAGUE-DISCOVERY. Strategy finalized: FM grades =
  Kaggle bulk CSVs for FM21/22/23 (db1/2/3, all leagues at once, source=kaggle, joined to lineup
  players by name) + fminside for FM24/FMU25/FM26 (db5/6/7 only). Halves fminside work. Kaggle
  parser: right-anchored (name from front, 47 attrs from trailing block) to survive unquoted commas
  in money fields; 'nat1' recognized for the Natural-Fitness column. ESPN league universe = 244
  competitions, 48 domestic top divisions. We now collect match data (ESPN-primary: results+lineups+
  context+events) for 31 leagues with full lineups: 8 original + 17 (Brazil..Japan) + 7 (Chile/China/
  Ecuador/India/Paraguay/Peru/South Africa). 17 more top divisions exist on ESPN but lack lineups
  (Uruguay/Ireland/Uganda/Kenya/etc.) → would need other official sources (deferred, low value / often
  not in FM DB). Leagues without football-data odds use ESPN as primary results source. 4 jobs running:
  fminside grades (Ligue1/Bundesliga/Serie A db7,6,5), kaggle bulk, ESPN new-17, ESPN new-7.

## 6. Status log

- 2026-06-12 21:47: project start; schema + db layer + fetch util done; 1140 matches loaded; 5 agents launched.
- 2026-06-12 22:20: all research agents reported. ESPN lineups loaded (1140/1140 matches, 45,557 appearance rows, avg 22 starters ✓). understat xG 1140/1140. LITERATURE.md written.
- 2026-06-12 23:00: futek scrape DONE — 1,793 EPL players, 60 attrs each incl. 13 hidden, raw CA/PA. Discovery: futek export is pre-summer-2023 data (Vicario@Empoli) despite FM24 branding → snapshots relabeled to 2023-06-01.
- 2026-06-13 (Phase 2): mopped up 13 stale scrape_log errors (all transient timeouts that succeeded on
  resume; zero genuine gaps — every failed UID has its db5 snapshot). Built `train_nn.py`: per-player
  attention encoder (98-dim player vec: 47 attrs + presence mask + position) + masked attention/mean
  pooling + Poisson goal-rate head. Test (300 matches, 2025-26): RPS .2141, acc 47.0%, log-loss 1.025 —
  beats prior, approaches bookmaker (.2097); λ calibration near-perfect (1.52/1.34 vs 1.53/1.27).
  torch 2.2 ↔ numpy 2.4 bridge broken → all tensors built from Python lists (numpy used only for the
  Poisson score matrix). Model weights at data/model_nn.pt.
- 2026-06-13 04:30 ALL SCRAPING COMPLETE: 8,105 snapshots / 336,396 attribute rows across 5 dated FM
  versions (futek 2023-06: 1,859; FM24 24.1: 1,417; FM24 24.3: 1,669; FMU25: 1,514; FM26 26.2: 1,646).
  Lineup→FM join coverage 94.0%; 1,041/1,140 matches complete. DB 23 MB. 13 fminside errors remaining
  (transient timeouts on youth players — negligible; rerun scraper to mop up). Final eval
  (300-match 2025-26 test): bookmaker 49.3%/RPS .2097, Elo+form 52.3%/.2109, MLP ratings+context
  50.0%/.2180, GBDT 47.3%/.2230. POC done.
- 2026-06-13 02:50: db7 (FM26, 1653 players) + db6 (FMU25, 1516 players) DONE. Coverage 92.8%,
  1031 complete matches (train 739 / test 292). No-odds: Elo+form logistic 52.4% / RPS .2093
  (≈ bookmaker .2088); GBDT 48.3% (hyperparams need tuning on fuller data); MLP 48.0%.
  Club-aware name disambiguation added to resolver. db5 remainder scraping (~4:15 ETA), db4 queued.
- 2026-06-13 00:30: FIRST TRAINING RUN (861 complete matches, train 650 / test 211 = 2025-26 partial,
  time split). No odds: majority 41.7% / RPS .2359; bookmaker B365 49.3% / .2158; Elo+form logistic
  52.1% / .2176; **HistGradientBoosting 48.8% acc, logloss 1.0023, RPS .2128 — beats bookmaker on
  logloss+RPS**; MLP 47.4% (needs more data, as literature predicts). Futek gap-fill added 70 missed
  players (4 wrong-identity matches detected by low-CA audit and deleted). Coverage now 86.4% of
  starter appearances; will rise as fminside db7/db6/db4 land.
- 2026-06-12 23:45: fminside.net degraded (25-60s/page, list-API 504). Scraper restarted slow+patient, priority db7 (2025-26) → db6 → db5 → db4. db5 partial: ~550/1893 done. SQLite contention fixed via autocommit. Dataset builder + train.py written; name-resolve fallbacks (reversed order, single-token) added. Coverage with partial data: 2023-24 87%, 2024-25 76%, 2025-26 64% of starter appearances.
