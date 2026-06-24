"""Standing-aware risk decision: gamble or play safe based on your ACTUAL effective place on the table.
'On top' only counts if your futures land — so effective points = current per-game pts + P(your winner)*50 +
P(your scorer)*30, computed for every player (rivals' futures count too). Then:
  - if you're the effective leader with a cushion  -> PROTECT (safe picks; don't add variance).
  - if you're effectively behind                   -> CHASE (gamble on high-multiplier games).
  - also do the OPPOSITE of your nearest rival's variance (pursuit game, knockout_robust.py).
Maps the verdict to a predict_game --strategy. Edit STANDINGS / ODDS to current reality (or pass --odds).
Usage: python experiments/decide_risk.py [--me YOU] [--games-left 12]
"""
import sys

# current per-game standings (pts, exacts, winner-pick, scorer-pick) — update as the table moves
STANDINGS = [
    ("RIVAL_3", 42, 9, "Netherlands", "Haaland"),
    ("RIVAL_4",    41, 7, "Argentina",   "Messi"),
    ("RIVAL_5",38, 7, "France",      "Mbappé"),
    ("RIVAL_6",38, 7, "Argentina",   "Olise"),
    ("RIVAL_7",   37, 5, "Spain",       "Kane"),
    ("RIVAL_1",  35, 4, "Spain",       "Mbappé"),
    ("YOU", 34, 3, "Spain",       "Mbappé"),
    ("RIVAL_2",    33, 5, "Spain",       "Mbappé"),
    ("RIVAL_8",  24, 2, "Brazil",      "Endrick"),
]
# live futures probabilities — UPDATE to current odds (these are rough placeholders)
P_WIN = {"Spain": 0.24, "France": 0.18, "Argentina": 0.15, "Brazil": 0.12, "Netherlands": 0.08}
P_SCORER = {"Mbappé": 0.22, "Haaland": 0.12, "Messi": 0.06, "Kane": 0.10, "Olise": 0.04, "Endrick": 0.05}
WIN_PTS, SCORER_PTS = 50, 30


def main():
    def arg(k, d): return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    me = arg("--me", "YOU"); games_left = arg("--games-left", 12)
    rows = []
    for name, pts, ex, win, scorer in STANDINGS:
        eff = pts + P_WIN.get(win, 0.0) * WIN_PTS + P_SCORER.get(scorer, 0.0) * SCORER_PTS
        rows.append([name, pts, ex, win, scorer, eff])
    rows.sort(key=lambda r: -r[5])
    print("=== effective table (current pts + E[futures]) ===", flush=True)
    print(f"  {'#':>2} {'player':10s} {'now':>4} {'eff':>6} {'exact':>5}  futures", flush=True)
    my = None
    for i, r in enumerate(rows, 1):
        tag = "  <- you" if r[0] == me else ""
        if r[0] == me: my = (i, r)
        print(f"  {i:>2} {r[0]:10s} {r[1]:>4} {r[5]:>6.1f} {r[2]:>5}  {r[3]}/{r[4]}{tag}", flush=True)
    rank, r = my; eff_me = r[5]; leader = rows[0]
    gap = leader[5] - eff_me
    # nearest rival above you (the one to pass)
    rival = rows[rank - 2] if rank >= 2 else None
    print(f"\n  YOU are effective #{rank}/{len(rows)}; gap to effective leader ({leader[0]}) = {gap:.1f} pts", flush=True)
    # decision
    if rank == 1 and (eff_me - rows[1][5]) >= 5:
        verdict = "PROTECT — you're the effective leader with a cushion. Play SAFE (--strategy exacts/chalk); don't add variance."
    elif rank == 1:
        verdict = "NARROW LEAD — effective #1 but thin. Mostly safe; match (mirror) your nearest chaser, don't gamble first."
    else:
        per_game = gap / max(games_left, 1)
        aggr = "hard" if per_game > 1.0 else "moderate"
        verdict = (f"CHASE — effectively behind by {gap:.1f} with ~{games_left} games (~{per_game:.2f}/game needed). "
                   f"GAMBLE {aggr} on high-multiplier games (--round qf/sf/final differentiates). "
                   f"Do the OPPOSITE of {rival[0] if rival else 'your rival'}'s variance: safe if they gamble.")
    print(f"\n  >>> {verdict}", flush=True)
    # futures reality check
    if r[3] in P_WIN and P_WIN[r[3]] < 0.20:
        print(f"  NOTE: your winner pick ({r[3]}) is only ~{P_WIN[r[3]]*100:.0f}% — your +50 is unlikely; "
              f"discount your 'real' position and lean more aggressive.", flush=True)
    if rank > 1 and leader[3] == r[3]:
        print(f"  NOTE: the effective leader also bet {r[3]} — your futures WASH vs them; it's a pure per-game race.", flush=True)


if __name__ == "__main__":
    main()
