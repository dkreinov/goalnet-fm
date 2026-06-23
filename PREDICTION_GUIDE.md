# WC2026 Prediction Agent — Operating Guide

Self-contained instructions for generating fantasy-league scoreline picks. Read this fully before picking.

## 1. Who / what / goal
- You are picking for **YOU** in a 10-player WC2026 fantasy league.
- **Per-game scoring:** exact score = 3, correct outcome (W/D/L) only = 1, wrong = 0. **Ties break on exact count.**
- **Knockout multipliers:** group games ×1; knockouts escalate — R32 ×2, R16 ×4, **QF ×8, SF ×12, Final ×16** (confirm the league's real values if shown).
- **Futures (locked, paid at the end):** winner = +50 (YOU picked **Spain**), top scorer = +30 (YOU picked **Mbappé**).
- **Goal:** finish #1.

## 2. The two layers (do not confuse them)
1. **GoalNet** (`data/goalnet.pt`, national-specialised 5-seed ensemble) = the MODEL. It outputs a **score-probability grid** per game. Always used.
2. **Pick strategy** = how you turn that grid into the single scoreline you submit. This is the lever you control per game.

## 3. The tool
```
python src/predict_game.py KEY1 KEY2 ... --strategy <chalk|exacts|contrarian>
```
- KEYs are team-code fixtures, e.g. `ESP-FRA BRA-ARG`. It prints, per game: the xG, W/D/L %, the pick, and the top-5 scorelines with probabilities.
- Strategies (all read the SAME GoalNet grid):
  - `chalk` — points-optimal pick (safe).
  - `exacts` — most likely EXACT after nudging to real common scores (1-1 draws, 2-1 over 2-0). **This is the default.**
  - `contrarian` — differentiated pick (for separating from rivals).

## 4. Strategic situation (why these rules)
- YOU's gap is **purely exact-score conversion**: league-best outcome reading (25 correct) but fewest exacts of the contenders (3). Do **NOT** go contrarian on outcomes — that throws away the one elite edge.
- Because YOU, **RIVAL_1**, and **RIVAL_2** all picked **Spain + Mbappé**, the +80 futures wash among them. The title is really a **3-way per-game race vs RIVAL_1 (ahead, +1 pt / +1 exact) and RIVAL_2 (behind on pts, ahead on exacts)**. The nominal leader (RIVAL_3, Netherlands+Haaland) is irrelevant unless the Netherlands win.
- The **draw hole** is the biggest leak: GoalNet/chalk nails almost no drawn games as exacts. The `exacts` strategy fixes this by favouring 1-1.

## 5. Per-game decision procedure
For each fixture:
1. Run `predict_game.py KEY --strategy exacts` to get GoalNet's grid + the exact-optimal pick.
2. **Group stage / R32 / R16 (×1–×4):** submit the `exacts` pick. Protect points; don't gamble here.
3. **QF and bigger (×8 / ×12 / ×16):** this is where the table moves. Decide gamble vs safe by watching RIVAL_1:
   - If RIVAL_1 is **playing safe / picks look like the obvious chalk** (gap staying frozen) → **gamble**: back your own confident, differentiated scoreline (or run `--strategy contrarian`). Safe here = 0% chance to pass him.
   - If RIVAL_1 is **gambling** (taking unobvious scores) → **play safe** (`exacts`): he's only +1, so a missed gamble drops him below you. Let his variance sink him.
   - If you can't tell → **default to gambling the QF+** (the robust choice; never leaves you at 0%).
4. **Draws:** when GoalNet leans even/tight, pick **1-1** (most common real scoreline). Don't hedge a draw into a favourite's win-score.
5. **Favourites:** prefer **2-1 or 1-0** over 2-0 (2-0 is the classic near-miss).

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
