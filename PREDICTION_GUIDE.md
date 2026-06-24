# WC2026 Prediction Agent — Operating Guide

Self-contained instructions for generating fantasy-league scoreline picks. Read this fully before picking.

## 1. Who / what / goal
- You are picking for **YOU** in a 10-player WC2026 fantasy league.
- **Per-game scoring:** exact score = 3, correct outcome (W/D/L) only = 1, wrong = 0. **Ties break on exact count.**
- **Knockout multipliers (real league):** group ×1, R32 ×2, R16 ×4, **QF ×8, SF ×16, Final ×32**. So a Final correct = 32 and a **Final exact = 64** — one Final exact alone more than erases an 8-point gap. The Final (then SF) is the single highest-leverage game; spend your best differentiated exact read there.
- **Futures (locked, paid at the end):** winner = +50 (YOU picked **Spain**), top scorer = +30 (YOU picked **Mbappé**).
- **Goal:** finish #1.

## 2. The two layers (do not confuse them)
1. **GoalNet** (`data/goalnet.pt`, national-specialised 5-seed ensemble) = the MODEL. It outputs a **score-probability grid** per game. Always used.
2. **Pick strategy** = how you turn that grid into the single scoreline you submit. This is the lever you control per game.

## 3. The tool
```
python src/predict_game.py KEY1 KEY2 ... --round <group|r32|r16|qf|sf|final> --rival <unknown|safe|gambling>
```
- KEYs are team-code fixtures, e.g. `ESP-FRA BRA-ARG`. It prints, per game: the round/multiplier + reasoning, the xG, W/D/L %, the pick, and the top-5 scorelines with probabilities.
- **Upcoming fixtures work.** If the fixture isn't in `lineups.json` yet (no confirmed XI), the tool falls back to each team's **previous-game XI** and tags the line `[FALLBACK: previous-game XIs ...]`. Once the real lineup is posted it's used automatically. So you can predict future games now; re-run near kickoff to upgrade to the confirmed XI.
- **`--round` auto-applies all the decision logic in §5** — you normally only need `--round` (and `--rival` if you've read RIVAL_1). The tool picks the strategy for you and prints why.
- Manual override (rarely needed): `--strategy <chalk|exacts|contrarian|gamble>` forces a specific pick rule and ignores `--round`.
  - `chalk` — points-optimal (safe). `exacts` — most likely exact, nudged to real scores (1-1, 2-1). `gamble` — 2nd-best differentiated exact. `contrarian` — EV-differentiated.
- Default with no flags = `exacts`.

## 4. Strategic situation (why these rules)
Standings as of 2026-06-23 (per-game pts / exacts — winner pick / scorer pick):

| # | player | pts | exacts | winner | scorer |
|---|---|---|---|---|---|
| 1 | RIVAL_3 | 42 | 9 | Netherlands | Haaland |
| 2 | RIVAL_4 | 41 | 7 | Argentina | Messi |
| 3 | RIVAL_5 | 38 | 7 | France | Mbappé |
| 4 | RIVAL_6 | 38 | 7 | Argentina | Olise |
| 5 | RIVAL_7 | 37 | 5 | Spain | Kane |
| 6 | RIVAL_1 | 35 | 4 | **Spain** | **Mbappé** |
| 7 | **YOU (you)** | 34 | 3 | **Spain** | **Mbappé** |
| 8 | RIVAL_2 | 33 | 5 | **Spain** | **Mbappé** |
| 9 | RIVAL_8 | 24 | 2 | Brazil | Endrick |

- YOU's gap is **purely exact-score conversion**: league-best outcome reading (25 correct) but fewest exacts of the contenders (3). Do **NOT** go contrarian on outcomes — that throws away the one elite edge.
- YOU, **RIVAL_1**, and **RIVAL_2** all picked **Spain + Mbappé**, so the +80 futures wash among them. The title is really a **3-way per-game race vs RIVAL_1 (ahead +1 pt / +1 exact) and RIVAL_2 (behind on pts, ahead on exacts; ties break on exacts → effectively behind both)**. The nominal leader (RIVAL_3) is irrelevant unless the Netherlands win.
- Everything is conditional on **Spain winning + Mbappé top scorer** (the only path to #1). Play every pick as if that holds.
- The **draw hole** is the biggest leak: GoalNet/chalk nails almost no drawn games as exacts. The `exacts` strategy fixes this by favouring 1-1.
- **Gamble-vs-safe is a pursuit game vs RIVAL_1:** if he plays safe you must gamble (QF+); if he gambles, you stay safe and let his variance sink him. The `--rival` flag encodes this.

## 4b. Standing-aware risk (run once per matchday, before picking)
"On top" only counts if your futures land. Compute your **effective** standing (current pts + P(your winner)×50 + P(your scorer)×30, for everyone) and let it set how aggressive to be:
```
python experiments/decide_risk.py --games-left <N>
```
Update `STANDINGS` and the `P_WIN` / `P_SCORER` odds inside the script to current reality first. It prints the effective table and a verdict:
- **PROTECT** (effective leader with a cushion) → play safe everywhere (`--strategy exacts`); do NOT add variance.
- **NARROW LEAD** → mostly safe; mirror your nearest chaser.
- **CHASE** (effectively behind) → gamble on the high-multiplier rounds; harder the bigger the gap-per-game.

This gates §5: only open up (gamble QF+) to the degree the verdict says. If you're effectively top, stay safe even in the knockouts.

## 5. Per-game decision procedure
The `--round` flag now applies steps 1–3 automatically. Just pass the correct round (and `--rival` if you've read RIVAL_1). The logic it encodes:
1. Pass `--round <round>` for the fixture's stage. The tool maps it to a multiplier and picks the strategy.
2. **Group / R32 / R16 (×1–×4):** → `exacts` (safe exact-hunting). Protect points; don't gamble here.
3. **QF and bigger (×8 / ×12 / ×16):** this is where the table moves. The tool decides by `--rival`:
   - `--rival safe` or `unknown` → **gamble** (differentiated exact). Safe here = 0% chance to pass RIVAL_1.
   - `--rival gambling` → **stay safe** (`exacts`): RIVAL_1 is only +1, so a missed gamble drops him below you — let his variance sink him.
   - Default `--rival unknown` gambles QF+ (the robust, never-0% choice).
4. **Draws** (handled inside `exacts`/`gamble`): GoalNet+empirical favours **1-1** when even. Don't hedge a draw into a favourite's win-score.
5. **Favourites:** the corrected grid already prefers **2-1 / 1-0** over 2-0 (the classic near-miss).

Example: `python src/predict_game.py ESP-FRA --round sf --rival gambling` → ×12 game, RIVAL_1 gambling → tool plays `exacts` (safe) and prints the reason.

## 6. Hard rules
- Always submit a scoreline GoalNet's grid supports (top-5); never a score with ~0 probability, even when "gambling" — gamble = your 2nd/3rd-best read, not a wild punt.
- Everything rides on **Spain winning + Mbappé top scorer** (locked). All per-game effort is conditional on that; there's no path to #1 without it, so play every pick as if it holds.
- Keep YOU's outcome reads — only the *scoreline within* the read is up for optimisation.

## 7. References (analysis behind this, re-runnable)
- `experiments/exacts.py` — exact-hit by policy (the draw-hole evidence).
- `experiments/knockout_strategy.py` — gamble-vs-safe by multiplier threshold (QF+ optimal).
- `experiments/knockout_robust.py` — pursuit-game matrix vs RIVAL_1's behaviour (do the opposite of his variance).
- `experiments/catchup.py` — deficit/rounds catch-up curves.
- Full record: `RESULTS_WC2026.md`. Standings/rivals: memory `wc2026-league-strategy.md`.
