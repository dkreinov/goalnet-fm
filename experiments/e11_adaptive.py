"""E11 — tournament-forward sim with leaderboard-aware adaptive risk (crazy idea, extends E3, no retrain).
Treat the played WC games as sequential rounds. A field of K opponents accumulates points; the HERO adapts its
contrarian beta each round to its standing: when BEHIND the leader it raises beta (more differentiation =
higher variance to catch up); when AHEAD it lowers beta toward chalk (protect the lead). Compare against fixed
policies (chalk, fixed-contrarian) by P(finish #1) via Monte-Carlo. Tests whether *adaptivity* beats any fixed
beta. Usage: python experiments/e11_adaptive.py [--K 20] [--q 0.6] [--S 3000]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import e3_gametheory as E3


def adaptive_beta(gap, rounds_left, base=1.5):
    """Behind (gap<0) -> high beta (contrarian); ahead -> ~0 (chalk). Normalise the gap by rounds remaining."""
    x = -gap / (rounds_left + 1.0)            # >0 when behind
    return float(np.clip(base * x, 0.0, 4.0))


def sim_policy(grids, chalk_cells, flats, K, q, S, policy, fixed_beta=0.0):
    """MC over outcome draws + opponent picks. policy in {chalk, fixed, adaptive}. Returns P(sole#1), P(top1)."""
    G = len(grids); M = grids[0].shape[0]; rng = np.random.RandomState(0); np.random.seed(0)
    sole = top1 = 0
    evgrids = [E3.ev_points_grid(P) for P in grids]
    fmass = [E3.field_mass(P, q) for P in grids]
    for _ in range(S):
        hero = 0.0; opp = np.zeros(K)
        for g in range(G):
            if policy == "chalk": beta = 0.0
            elif policy == "fixed": beta = fixed_beta
            else: beta = adaptive_beta(hero - opp.max(), G - g)
            score = evgrids[g] - beta * fmass[g]
            hi, hj = np.unravel_index(np.argmax(score), score.shape)
            o = np.random.choice(M * M, p=flats[g]); oi, oj = o // M, o % M
            hero += E3.pts_of((hi, hj), (oi, oj))
            for k in range(K):
                c = chalk_cells[g] if rng.random() < q else np.random.choice(M * M, p=flats[g])
                opp[k] += E3.pts_of((c // M, c % M), (oi, oj))
        bestopp = opp.max()
        if hero > bestopp: sole += 1; top1 += 1
        elif hero == bestopp: top1 += 1
    return sole / S, top1 / S


def main():
    def arg(k, d): return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    K = arg("--K", 20); q = arg("--q", 0.6); S = arg("--S", 3000)
    c = H.load_cache(); w = c["wc"]
    grids = [H.ens_grid(w["lh"][i], w["la"][i], 0.0) for i in range(len(w["keys"]))]
    M = grids[0].shape[0]
    chalk = [np.ravel_multi_index(E3.chalk_pick(P), P.shape) for P in grids]
    flats = [P.ravel() / P.sum() for P in grids]
    print(f"=== E11 adaptive-risk tournament ({len(grids)} rounds, K={K}, q={q}, S={S}) ===", flush=True)
    print(f"  {'policy':22s} {'P(sole#1)':>10s} {'P(top1)':>9s}", flush=True)
    for tag, pol, fb in [("chalk", "chalk", 0.0), ("fixed contrarian b=0.25", "fixed", 0.25),
                         ("fixed contrarian b=1.0", "fixed", 1.0), ("ADAPTIVE (base1.5)", "adaptive", 0.0)]:
        s1, t1 = sim_policy(grids, chalk, flats, K, q, S, pol, fb)
        print(f"  {tag:22s} {s1:10.3f} {t1:9.3f}", flush=True)


if __name__ == "__main__":
    main()
