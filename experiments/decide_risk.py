"""Standing-aware risk decision: gamble or play safe based on your ACTUAL effective place on the table.
'On top' only counts if your futures land — so effective points = current per-game pts + P(your winner)*50 +
P(your scorer)*30, computed for every player (rivals' futures count too). Then:
  - if you're the effective leader with a cushion  -> PROTECT (safe picks; don't add variance).
  - if you're effectively behind                   -> CHASE (gamble on high-multiplier games).
  - also do the OPPOSITE of your nearest rival's variance (pursuit game, knockout_robust.py).
Maps the verdict to a predict_game --strategy. Edit STANDINGS / ODDS to current reality (or pass --odds).
Usage: python experiments/decide_risk.py [--me YOU] [--games-left 12]
"""
import sys, json
from pathlib import Path

# LIVE standings come from standings.json (run `python src/read_standings.py` first). Fallback below is a
# stale snapshot used only if that file is missing.
STANDINGS = [
    ("RIVAL_3", 42, 9, "Netherlands", "Haaland"), ("RIVAL_4", 41, 7, "Argentina", "Messi"),
    ("RIVAL_5", 38, 7, "France", "Mbappé"), ("RIVAL_6", 38, 7, "Argentina", "Olise"),
    ("RIVAL_7", 37, 5, "Spain", "Kane"), ("RIVAL_1", 35, 4, "Spain", "Mbappé"),
    ("YOU", 34, 3, "Spain", "Mbappé"), ("RIVAL_2", 33, 5, "Spain", "Mbappé"),
    ("RIVAL_8", 24, 2, "Brazil", "Endrick"),
]


def load_standings():
    """(rows, me_nick) from live standings.json if present, else the stale fallback above."""
    f = Path(__file__).resolve().parent.parent / "standings.json"
    if f.exists():
        d = json.load(open(f, encoding="utf-8")); me = "YOU"
        rows = []
        for r in d["rows"]:
            rows.append((r["nick"], r["pts"], r["ex"], r.get("winner"), r.get("scorer")))
            if r.get("is_me"): me = r["nick"]
        return rows, me, d.get("finished_fixtures")
    return STANDINGS, "YOU", None
# live futures probabilities — UPDATE to current odds (these are rough placeholders)
P_WIN = {"Spain": 0.24, "France": 0.18, "Argentina": 0.15, "Brazil": 0.12, "Netherlands": 0.08}
P_SCORER = {"Mbappé": 0.22, "Haaland": 0.12, "Messi": 0.06, "Kane": 0.10, "Olise": 0.04, "Endrick": 0.05}
WIN_PTS, SCORER_PTS = 50, 30
SCORER_ON = False   # PHASE 1: winner-only. Ignore the top-scorer +30 until a later phase.

# Reality guards — the current tactic is valid only while these hold. Set from the live bracket/scorer race.
STATUS = {"spain_in": True,        # Spain still in the tournament -> your +50 is live
          "france_in": True,       # France still in -> RIVAL_5 (France+Mbappé) is a live threat
          "mbappe_leading": False} # Mbappé NOT leading the golden boot yet


def main():
    def arg(k, d): return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    standings, me_default, nfin = load_standings()
    me = arg("--me", me_default)
    games_left = arg("--games-left", (103 - nfin) if nfin else 12)
    if nfin is not None:
        print(f"(live standings: {nfin} fixtures finished, ~{103 - nfin} left)", flush=True)
    rows = []
    for name, pts, ex, win, scorer in standings:
        eff = pts + P_WIN.get(win, 0.0) * WIN_PTS
        if SCORER_ON:
            eff += P_SCORER.get(scorer, 0.0) * SCORER_PTS
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
    if not SCORER_ON:
        print("  (phase 1: winner-only; top-scorer +30 ignored for now)", flush=True)
    # reality guards — the verdict above is only valid while these hold
    if not STATUS["spain_in"]:
        print("  *** CHANGE TACTICS: Spain is OUT — your +50 is gone. You're now effectively far behind; "
              "GAMBLE hard on every multiplier game. Re-run with real winner odds.", flush=True)
    else:
        flips = []
        if not STATUS["france_in"]:
            flips.append("France OUT (RIVAL_5's +50 collapses → you move clearly ahead → protect harder)")
        if STATUS["mbappe_leading"]:
            flips.append("Mbappé now leading the golden boot (re-enable scorer bonus next phase)")
        if flips:
            print("  RE-EVALUATE — a guard flipped: " + "; ".join(flips), flush=True)
        else:
            print("  guards OK: Spain in, France in, Mbappé not leading → current tactic stands.", flush=True)
    if rank > 1 and leader[3] == r[3]:
        print(f"  NOTE: the effective leader also bet {r[3]} — your futures WASH vs them; pure per-game race.", flush=True)


if __name__ == "__main__":
    main()
