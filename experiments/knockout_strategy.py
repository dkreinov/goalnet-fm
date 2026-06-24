"""Gamble vs safe in the high-multiplier knockout phase, against your ACTUAL rivals (RIVAL_1, RIVAL_2) who share
your Spain+Mbappe futures (so the +80 washes -> frozen, it's a 3-way per-game race). Start (per-game/exacts):
you 34/3, RIVAL_1 35/4, RIVAL_2 33/5; ties break on exacts. Rivals mirror the model's best-exact pick (the leader's
optimal is to copy you). YOU choose: safe (= same pick, correlated -> gap frozen) or gamble (a decorrelated
2nd-best scoreline) on games with multiplier >= threshold T. Outcomes drawn from the model grids; knockout
multipliers escalate (R32 x2, R16 x4, QF x8, SF x12, F x16). Reports P(you finish #1 of the trio).
Usage: python experiments/knockout_strategy.py [--S 60000]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import e3_gametheory as E3

# remaining knockout multiplier schedule (one pick per fixture you make)
SCHED = [2] * 16 + [4] * 8 + [8] * 4 + [16] * 2 + [32] * 1     # R32..Final (real league: QF8/SF16/Final32)
START = {"you": (34, 3), "pizzi": (35, 4), "yak": (33, 5)}      # (points, exacts)


def top2_cells(P):
    flat = P.ravel(); o = np.argsort(flat)[::-1]
    M = P.shape[0]; return (int(o[0]) // M, int(o[0]) % M), (int(o[1]) // M, int(o[1]) % M)


def simulate(grids, T, S, rng):
    """You gamble (2nd-best cell) when multiplier>=T, else mirror chalk. Rivals always chalk. Returns P(#1)."""
    M = grids[0].shape[0]; flats = [P.ravel() / P.sum() for P in grids]
    chalk = []; gamble = []
    for P in grids:
        c, g = top2_cells(P); chalk.append(c); gamble.append(g)
    G = len(grids); first = 0
    yp0, ye0 = START["you"]; pp0, pe0 = START["pizzi"]; yk0, yke0 = START["yak"]
    for _ in range(S):
        yp, ye = yp0, ye0; pp, pe = pp0, pe0; yk, yke = yk0, yke0
        for t, m in enumerate(SCHED):
            gi = rng.randint(G); P = grids[gi]
            o = rng.choice(M * M, p=flats[gi]); oi, oj = o // M, o % M
            mine = gamble[gi] if m >= T else chalk[gi]
            rivalcell = chalk[gi]
            yp_g = E3.pts_of(mine, (oi, oj)); rp_g = E3.pts_of(rivalcell, (oi, oj))
            yp += yp_g * m; ye += (yp_g == 3)
            pp += rp_g * m; pe += (rp_g == 3)            # RIVAL_1 & RIVAL_2 mirror chalk (correlated)
            yk += rp_g * m; yke += (rp_g == 3)
        def beats(a_p, a_e, b_p, b_e): return a_p > b_p or (a_p == b_p and a_e > b_e)
        if beats(yp, ye, pp, pe) and beats(yp, ye, yk, yke): first += 1
    return first / S


def main():
    def arg(k, d): return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    S = arg("--S", 60000)
    c = H.load_cache(); w = c["wc"]
    grids = [H.ens_grid(w["lh"][i], w["la"][i], 0.0) for i in range(len(w["keys"]))]
    print(f"=== gamble-vs-safe in knockouts (3-way trio race, {len(SCHED)} KO games, S={S}) ===", flush=True)
    print(f"  start you 34/3ex, RIVAL_1 35/4ex, RIVAL_2 33/5ex (futures wash). Rivals mirror chalk.", flush=True)
    print(f"  {'your policy':28s} {'P(win trio)':>12s}", flush=True)
    rows = [("safe everywhere (mirror)", 999), ("gamble only Final (x16)", 16), ("gamble SF+ (x12+)", 12),
            ("gamble QF+ (x8+)", 8), ("gamble R16+ (x4+)", 4), ("gamble everything", 0)]
    for tag, T in rows:
        rng = np.random.RandomState(0)
        p = simulate(grids, T, S, rng)
        print(f"  {tag:28s} {p:12.3f}", flush=True)


if __name__ == "__main__":
    main()
