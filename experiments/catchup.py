"""Catch-up strategy (vectorised, high-S): you are DEFICIT points behind the leader with ROUNDS games left.
Find how aggressive to play (contrarian beta, or pure max-exact) to MAXIMISE P(finish #1). Field of K
opponents (chalk w/ prob q else informed crowd) with head-starts spread over [0, deficit] (leader = deficit
ahead). Future games drawn from a fixed schedule sampled from the played-WC grids. Fully numpy-vectorised over
sims so S can be large for accurate rare-event probabilities.
Usage: python experiments/catchup.py [--deficit 12] [--rounds 12] [--K 20] [--q 0.6] [--S 40000]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import e3_gametheory as E3


def points_matrix(M):
    """PM[p,o] = fantasy points for pick cell p vs outcome cell o (exact=3, correct outcome=1)."""
    idx = [(i, j) for i in range(M) for j in range(M)]
    PM = np.zeros((M * M, M * M), np.int8)
    for p, (pi, pj) in enumerate(idx):
        sp = (pi > pj) - (pi < pj)
        for o, (oi, oj) in enumerate(idx):
            if pi == oi and pj == oj: PM[p, o] = 3
            else: PM[p, o] = 1 if sp == ((oi > oj) - (oi < oj)) else 0
    return PM


def hero_pick_for(P, q, beta, kind):
    if kind == "max_exact": return int(np.argmax(P))
    EV = E3.ev_points_grid(P); F = E3.field_mass(P, q)
    return int(np.argmax((EV - beta * F).ravel()))


def simulate(schedule, PM, M, deficit, K, q, S, kind, beta, rng):
    """Vectorised MC. schedule = list of (flat, chalk_idx, hero_pick_idx). Returns P(sole#1), P(top1)."""
    R = len(schedule); MM = M * M
    hero = np.zeros(S); opp = np.tile(np.linspace(deficit, 0, K), (S, 1))   # (S,K) head-starts
    for flat, chalk_i, hp in schedule:
        O = rng.choice(MM, size=S, p=flat)                                  # outcomes (S,)
        hero += PM[hp, O]
        u = rng.random((S, K))
        samp = rng.choice(MM, size=S * K, p=flat).reshape(S, K)
        picks = np.where(u < q, chalk_i, samp)
        opp += PM[picks, O[:, None]]
    best = opp.max(1)
    return float((hero > best).mean()), float((hero >= best).mean())


def main():
    def arg(k, d): return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    K = arg("--K", 20); q = arg("--q", 0.6); S = arg("--S", 40000); rounds = arg("--rounds", 12)
    deficit = arg("--deficit", None)
    c = H.load_cache(); w = c["wc"]
    grids = [H.ens_grid(w["lh"][i], w["la"][i], 0.0) for i in range(len(w["keys"]))]
    M = grids[0].shape[0]; PM = points_matrix(M)
    sched_idx = np.linspace(0, len(grids) - 1, rounds).round().astype(int)   # fixed schedule of typical games
    sched_grids = [grids[i] for i in sched_idx]
    betas = [0.0, 0.25, 0.5, 1.0, 2.0]
    policies = [("chalk", "contra", 0.0)] + [("contra", "contra", b) for b in betas[1:]] + [("max_exact", "max_exact", 0.0)]
    defs = [int(deficit)] if deficit is not None else [3, 8, 15, 25]
    print(f"=== catch-up (vectorised, rounds={rounds}, K={K}, q={q}, S={S}) — P(sole #1) / P(top1) ===", flush=True)
    hdr = "  deficit | " + "  ".join(f"{('b='+str(b)) if name=='contra' else name:>9s}" for name, kind, b in policies)
    print(hdr + "   | best", flush=True)
    for D in defs:
        cells = []
        for name, kind, b in policies:
            sched = [(P.ravel() / P.sum(), int(E3.chalk_pick(P)[0] * M + E3.chalk_pick(P)[1]),
                      hero_pick_for(P, q, b, kind)) for P in sched_grids]
            rng = np.random.RandomState(0)
            s1, t1 = simulate(sched, PM, M, D, K, q, S, kind, b, rng); cells.append((s1, t1))
        best_i = int(np.argmax([x[0] for x in cells]))
        bn = policies[best_i]; blab = f"b={bn[2]}" if bn[0] == "contra" else bn[0]
        print(f"  {D:7d} | " + "  ".join(f"{s1:.3f}/{t1:.2f}" for s1, t1 in cells) + f"   | {blab}", flush=True)


if __name__ == "__main__":
    main()
