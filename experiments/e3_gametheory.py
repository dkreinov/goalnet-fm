"""E3 — game-theoretic / contrarian picks. The league is a contest vs other players, not vs the house:
maximising E(points) (chalk EV-pick) is NOT the same as maximising P(finish #1). If the field clusters on the
modal score, a differentiated pick with slightly lower EV but low overlap can win the league.

Setup: per-game ensemble grids = our outcome belief P_g. Field = K opponents; each opponent's pick per game
is chalk (the EV-pick) with prob q, else sampled ~ P_g (informed crowd). Monte-Carlo over S sims: draw true
outcomes ~P_g, draw opponents' picks, score hero policy vs same outcomes, record rank. Compare hero policies:
  - chalk      : production EV-pick (max E points)
  - max_exact  : argmax P(i,j)
  - contrarian : argmax over cells of EV(c) - beta * F(c), where F = expected field-pick mass on c; beta tuned
                 on MC to maximise P(sole #1). beta=0 reduces to chalk.
Reports P(sole #1), P(top-1 incl ties), mean points. Also a realized check on the 40 played WC games.
Usage: python D:/Programming/claude/FM/experiments/e3_gametheory.py [--K 20] [--q 0.6] [--S 4000]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import train_goals as tg


def pts_of(pick, out):
    pi, pj = int(pick[0]), int(pick[1]); oi, oj = int(out[0]), int(out[1])
    if pi == oi and pj == oj: return 3
    so = (oi > oj) - (oi < oj); sp = (pi > pj) - (pi < pj)
    return 1 if so == sp else 0


def chalk_pick(P): return tg.ev_pick(P)
def exact_pick(P): i, j = np.unravel_index(np.argmax(P), P.shape); return (int(i), int(j))


def field_mass(P, q):
    """Expected field-pick distribution over cells: q on the chalk cell + (1-q) * P."""
    F = (1 - q) * P.copy(); ci, cj = chalk_pick(P); F[ci, cj] += q; return F


def ev_points_grid(P):
    """EV(points) for every candidate pick cell (exact=3, correct outcome=1)."""
    ho = tg.hda_from_P(P); M = P.shape[0]; EV = np.zeros_like(P)
    for i in range(M):
        for j in range(M):
            oc = 0 if i > j else (1 if i == j else 2)
            EV[i, j] = 3 * P[i, j] + (ho[oc] - P[i, j])
    return EV


def contrarian_pick(P, q, beta):
    EV = ev_points_grid(P); F = field_mass(P, q); score = EV - beta * F
    i, j = np.unravel_index(np.argmax(score), score.shape); return (int(i), int(j))


def policy_picks(grids, name, q=0.6, beta=0.0):
    if name == "chalk": return [chalk_pick(P) for P in grids]
    if name == "max_exact": return [exact_pick(P) for P in grids]
    if name == "contrarian": return [contrarian_pick(P, q, beta) for P in grids]
    raise ValueError(name)


def simulate(grids, hero_picks, K, q, S, rng):
    """MC: returns (P_sole_1st, P_top1_incl_ties, mean_hero_pts). Opponents = chalk w/ prob q else ~P_g."""
    G = len(grids); M = grids[0].shape[0]
    flat = [P.ravel() / P.sum() for P in grids]
    chalk = [np.ravel_multi_index(chalk_pick(P), P.shape) for P in grids]
    hero_flat = [np.ravel_multi_index(p, (M, M)) for p in hero_picks]
    sole = top1 = 0; hpts_sum = 0.0
    for _ in range(S):
        outs = [np.random.choice(M * M, p=flat[g]) for g in range(G)]   # true outcomes
        out_ij = [(o // M, o % M) for o in outs]
        # hero
        hp = sum(pts_of((hero_flat[g] // M, hero_flat[g] % M), out_ij[g]) for g in range(G))
        hpts_sum += hp
        # opponents
        best_opp = -1; n_at_best = 0
        for _k in range(K):
            op = 0
            for g in range(G):
                if rng.random() < q: c = chalk[g]
                else: c = np.random.choice(M * M, p=flat[g])
                op += pts_of((c // M, c % M), out_ij[g])
            if op > best_opp: best_opp, n_at_best = op, 1
            elif op == best_opp: n_at_best += 1
        if hp > best_opp: sole += 1; top1 += 1
        elif hp == best_opp: top1 += 1
    return sole / S, top1 / S, hpts_sum / S


def main():
    def arg(k, d): return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    K = arg("--K", 20); q = arg("--q", 0.6); S = arg("--S", 4000)
    c = H.load_cache(); w = c["wc"]
    grids = [H.ens_grid(w["lh"][i], w["la"][i], 0.0) for i in range(len(w["keys"]))]
    acts = [(int(w["hs"][i]), int(w["as_"][i])) for i in range(len(grids))]
    rng = np.random.RandomState(0); np.random.seed(0)
    print(f"=== game-theory on {len(grids)} played WC games (K={K} opponents, q={q} chalk, S={S} sims) ===", flush=True)
    # tune contrarian beta on the MC objective P(sole #1)
    betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for name in ["chalk", "max_exact"]:
        hp = policy_picks(grids, name)
        np.random.seed(0); rng = np.random.RandomState(0)
        s1, t1, mp = simulate(grids, hp, K, q, S, rng)
        rows.append((name, 0.0, s1, t1, mp))
    for beta in betas:
        hp = policy_picks(grids, "contrarian", q, beta)
        np.random.seed(0); rng = np.random.RandomState(0)
        s1, t1, mp = simulate(grids, hp, K, q, S, rng)
        rows.append((f"contrarian", beta, s1, t1, mp))
    print(f"  {'policy':12s} {'beta':>5s} {'P(sole#1)':>10s} {'P(top1)':>9s} {'meanPts':>8s}", flush=True)
    for name, beta, s1, t1, mp in rows:
        print(f"  {name:12s} {beta:5.2f} {s1:10.3f} {t1:9.3f} {mp:8.2f}", flush=True)
    # realized check on actual results (single realization, n small -> noisy, reality check only)
    print("  --- realized points on ACTUAL results ---", flush=True)
    for name, beta in [("chalk", 0.0), ("max_exact", 0.0), ("contrarian", 1.0), ("contrarian", 2.0)]:
        hp = policy_picks(grids, name, q, beta)
        tot = sum(pts_of(hp[g], acts[g]) for g in range(len(grids)))
        ex = sum(hp[g] == acts[g] for g in range(len(grids)))
        print(f"    {name:12s} beta={beta:.1f} pts={tot}/{len(grids)} exact={ex}", flush=True)


if __name__ == "__main__":
    main()
