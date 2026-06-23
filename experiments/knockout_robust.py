"""Robustness of the gamble-vs-safe conclusion to what RIVAL_1/RIVAL_2 actually do. Players pick a cell RANK per
game (0=most likely, 1=2nd, 2=3rd); different ranks -> different scorelines -> decorrelated outcomes (this is
'thinking differently'). Same rank -> identical pick -> that game can't change their gap. We sweep YOUR policy
against several RIVAL_1 behaviours (RIVAL_2 modelled as a volatile gambler throughout) and report P(you win the
3-way trio). The question: is 'gamble on high multipliers' your best response no matter what RIVAL_1 does?
Usage: python experiments/knockout_robust.py [--S 40000]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import e3_gametheory as E3

SCHED = [2] * 16 + [4] * 8 + [8] * 4 + [12] * 2 + [16] * 1
START = {"you": (34, 3), "pizzi": (35, 4), "yak": (33, 5)}


def topk(P, k=3):
    o = np.argsort(P.ravel())[::-1][:k]; M = P.shape[0]
    return [(int(x) // M, int(x) % M) for x in o]


def rank_for(policy, m, rng):
    """Return the cell-rank a player uses on a game with multiplier m. you-gamble uses rank1, pizzi-gamble
    rank2 (so you two differ when both gamble); diverge = stochastic; chalk = rank0 (safe/consensus)."""
    kind, T, grank = policy
    if kind == "chalk": return 0
    if kind == "gamble": return grank if m >= T else 0
    if kind == "diverge": return rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])      # natural different reads
    if kind == "diverge_hi": return rng.choice([0, 1, 2], p=[0.25, 0.4, 0.35])  # volatile gambler (RIVAL_2)
    return 0


def simulate(grids, you_pol, piz_pol, yak_pol, S, rng):
    M = grids[0].shape[0]; flats = [P.ravel() / P.sum() for P in grids]
    cells = [topk(P, 3) for P in grids]
    G = len(grids); first = 0
    yp0, ye0 = START["you"]; pp0, pe0 = START["pizzi"]; yk0, yke0 = START["yak"]
    for _ in range(S):
        yp, ye = yp0, ye0; pp, pe = pp0, pe0; yk, yke = yk0, yke0
        for m in SCHED:
            gi = rng.randint(G); o = rng.choice(M * M, p=flats[gi]); out = (o // M, o % M); ck = cells[gi]
            ry = rank_for(you_pol, m, rng); rp = rank_for(piz_pol, m, rng); rk = rank_for(yak_pol, m, rng)
            gy = E3.pts_of(ck[ry], out); gp = E3.pts_of(ck[rp], out); gk = E3.pts_of(ck[rk], out)
            yp += gy * m; ye += (gy == 3); pp += gp * m; pe += (gp == 3); yk += gk * m; yke += (gk == 3)
        def beats(ap, ae, bp, be): return ap > bp or (ap == bp and ae > be)
        if beats(yp, ye, pp, pe) and beats(yp, ye, yk, yke): first += 1
    return first / S


def main():
    def arg(k, d): return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    S = arg("--S", 40000)
    c = H.load_cache(); w = c["wc"]
    grids = [H.ens_grid(w["lh"][i], w["la"][i], 0.0) for i in range(len(w["keys"]))]
    yak = ("diverge_hi", 0, 0)   # RIVAL_2 = volatile gambler throughout
    you_pols = {"you-safe": ("chalk", 0, 0), "you-gambleQF+": ("gamble", 8, 1), "you-gamble-all": ("gamble", 0, 1)}
    piz_modes = {"RIVAL_1-safe": ("chalk", 0, 0), "RIVAL_1-gambleQF+": ("gamble", 8, 2), "RIVAL_1-diverge": ("diverge", 0, 0)}
    print(f"=== P(you win trio) by YOUR policy x PIZZI behaviour (RIVAL_2 = volatile gambler, S={S}) ===", flush=True)
    print("  " + " " * 16 + "".join(f"{pm:>17s}" for pm in piz_modes), flush=True)
    for yn, yp in you_pols.items():
        row = []
        for pn, pp in piz_modes.items():
            rng = np.random.RandomState(0); row.append(simulate(grids, yp, pp, yak, S, rng))
        print(f"  {yn:16s}" + "".join(f"{x:17.3f}" for x in row), flush=True)


if __name__ == "__main__":
    main()
