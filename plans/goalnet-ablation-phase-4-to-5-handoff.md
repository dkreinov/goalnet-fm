# Phase 4 → Phase 5 handoff — GoalNet ablation program

**Phase 4 (Market anchor: BetExplorer odds → feature + blend A/B): COMPLETE (2026-07-22).**
A fresh session starts Phase 5 from this file + `experiments/ablation/DESIGN.md` ("Phase-4 adopted
market/stage config" section has the full numbers).

## Headline result

**The betting market carries real, non-redundant signal: ~+0.024 nats of score-level information on
national games where odds exist** — the first genuinely NEW information source to pass the gate since
the harness was built (Phases 2–3 established that re-derived signal was tapped out). Two equivalent
capture mechanisms; they do NOT stack (same signal):
- **ctx-odds feature** (trained): best single config on the covered natl subset — grid_info +0.1783
  (base +0.1543), rps 0.1923, acc 0.562 (+5pp), ece 0.054. Also improves the 56%-covered club lane.
- **B1 post-hoc outcome-mass blend λ*=0.9** (zero training): +0.1790 on the same matches; applies to
  ANY model; identity where odds are missing. λ monotonic to 0.9 → the market dominates outcome
  opinion (the literature's model≈market, reproduced on our data).
- Best known overall: **feature + λ0.5 blend = +0.1803 / rps 0.1918 / ece 0.042.**
- **REJECTED**: training-time KL anchor (w 0.1/0.3 — neutral everywhere); thin knockout-flag stage
  feature (covered −0.032; richer stage labels = future data collection).

## Runs + verdicts (all vs combo-beta0-w1; covered subset = 137 identical natl eval matches)

| run | covered natl grid_info (Δ) | full natl Δ | full all Δ | verdict |
|---|---|---|---|---|
| ctx-odds (Arm A) | +0.1783 (+0.0240) | −0.0123 (diluted, 65% uncovered) | +0.0080 | **ADOPT** (market-aware configs) |
| market-blend-b1 (λ0.9) | +0.1790 (+0.0248) | n/a (identity off-coverage) | n/a | **ADOPT** (production layer) |
| anchor-kl01 / kl03 (B2) | +0.152 (−0.002) | +0.002 / −0.002 | +0.001 | REJECT (neutral) |
| ctx-stage (Arm S) | +0.1225 (−0.0318) | −0.0091 | +0.0014 | REJECT (as tested) |

## Frozen config for Phase 5

- **Experiment default UNCHANGED: `combo-beta0-w1` (`--beta 0 --w 1`, base 10-feat context)** — keeps
  the odds-less frozen WC-slate lane comparable across architecture runs (odds runs skip that lane).
- **BUT: Phase-5 winners must ALSO be scored with the market layer before adoption** (either rerun
  with `--ctx-extra ctx_odds.npz`, or apply the B1 blend via `blend_market.py` pattern to the
  winner's cached rates). An architecture that wins without odds but adds nothing beyond the
  odds-informed baseline (+0.1803 covered natl) is not a real win.
- Production-path recommendation (for Phase 6): ctx-odds feature + λ≈0.5 blend.

## Data inventory (what exists now)

- `data/natl_odds_raw.csv` — 2,229 scraped national matches (comp, season, date, teams, 1X2, knockout).
- `data/ctx_odds.npz` — **38,403 matches** ×[pH,pD,pA,ln_overround,has_odds], Shin de-vigged:
  37,665 club (from DB football-data b365/avg columns — the model LEARNS odds-usage here) + 738
  national (scraped). Coverage: club train 56%; natl train 62% / **natl eval only 35%** (137 of 397).
- `data/ctx_stage.npz` — 738 ×[knockout,has_stage] (thin; rejected as tested).
- `src/scrape_betexplorer.py`, `src/build_odds_feat.py` — rerunnable (fetch.py disk cache).
- Registry: 19 rows; per-seed + consolidated rate caches for every run.

## Multi-source TODO (explicitly open — user-aligned "collect more when needed")

BetExplorer limits (why eval coverage is 35%): senior friendlies not listed (their friendly page is
nation-vs-club + youth), 2026-cycle qualifiers have no page yet (world-cup-2026 is a soft-404), no
WC2026 odds. Next sources, in order: **oddsportal.com (free, same scraping style)** → **The Odds API
(paid ~$30-100, snapshots 2022→, covers qualifiers/friendlies/WC)** → Betfair historic (free bulk).
Filling these enables: stronger natl eval lane (35%→~70%+), a WC-slate odds extension
(`--wc-extra` pattern, wc_odds.npz keyed by game — designed in the Phase-4 plan Step 4, unbuilt), and
the Phase-6 replay with a live market layer.

## Ops lessons from this phase (scraping)

- **Soft-404 trap**: wrong BetExplorer slugs return a generic 640k landing page with HTTP 200 —
  detect via `<title> == "Football Stats, Results, Tables & H2H stats | BetExplorer"`.
- **Stage expansion**: each competition page embeds its stages as `?stage=<id>` hrefs in
  `ul.list-tabs--secondary` (both Qualification and Final-tournament menus) — WC/Euro QUALIFIERS come
  free as parent-tournament stages. Base pages show only the latest stage.
- robots.txt disallows `?stage=/?page=/?year=` for bots — **user explicitly authorized** these for
  this personal throttled/cached scrape (2026-07-22); sitemap (`sitemap/football/results*.xml`) is
  the robots-clean enumerator but is incomplete (missing copa-2024, all qualifiers).
- Parser: home=`<strong>` span, away=last span (hyphenated-country safe); 3 `[data-odd]` = 1/X/2;
  `AfP|ET|pen` score suffix = knockout marker. Team-name aliases in build_odds_feat.ALIAS
  (Czechia/Türkiye/United States/Republic of Ireland/…).
- Repeated gotcha (3rd occurrence): NEVER index a lazy NpzFile in a loop — materialize arrays first.

## Phase 5 — Architecture (updated skeleton)

Goal (from program phase-state): (a) cross-team attention variant (match-comparison block over both
XIs, HIGFormer-style) in the harness's model zoo; (b) plus-minus player ratings from our own 90k-match
DB as player-level/ctx features. Updated constraints from Phase 4:
- The bar is now the **odds-informed** number (+0.1803 covered natl), not just combo-beta0-w1 —
  architecture must add signal the market doesn't already provide (player-level interactions are a
  plausible such source; outcome-level strength is NOT — the market has that).
- Cross-attention on CPU: keep d small (4 cores, ~1.3GB/run); ~55min/5-seed budget per run will grow —
  consider seeds=3 for exploratory sweeps with seeds=5 confirmation on the winner.
- Plus-minus ratings: leakage-free chronological computation over 90k matches (same discipline as
  build_context/build_momentum); enters via `--ctx-extra` or as player-feature channels (the latter
  needs a model change — that IS the architecture phase's remit; train_goals.py stays untouched,
  variants live in the harness).
- Runner note: model variants need a `--arch` flag in run_ablation (GoalNet import stays; variant
  classes can live in a new experiments/ablation/models.py) — registry schema already has flags.

## Resume pointer

Phase state: phase 4 → COMPLETE, current phase = 5 (NOT STARTED — plan it first per PHASE_MODE
pause-between-phases). SESSION_MODE was autonomous throughout Phase 4.
