"""Exact-score hunting: the league gap is purely exact conversion (strong outcomes, few exacts). Compare
scoreline policies by EXACT-HIT COUNT (not total points) on the held-out national test and the played WC games.
Policies: chalk (EV-pick), max_exact (argmax P), and empirical/DC-corrected max_exact (shift 2-0->2-1 etc).
Also breaks exacts down by outcome type (home/draw/away) to expose the draw problem.
Usage: python experiments/exacts.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import train_goals as tg


def maxexact(P): return tuple(int(x) for x in np.unravel_index(np.argmax(P), P.shape))
def chalk(P): return tg.ev_pick(P)


def eval_policy(grids, hs, as_, pickfn):
    ex = co = 0; ex_by = {0: 0, 1: 0, 2: 0}; tot_by = {0: 0, 1: 0, 2: 0}
    for P, h, a in zip(grids, hs, as_):
        pi, pj = pickfn(P)
        oc = 0 if h > a else (1 if h == a else 2); tot_by[oc] += 1
        if pi == h and pj == a: ex += 1; ex_by[oc] += 1
        elif (pi > pj) == (h > a) and (pi < pj) == (h < a): co += 1
    return ex, co, ex_by, tot_by


def main():
    c = H.load_cache(); emp = H.empirical_grid()
    te = c["te_natl"].astype(bool)
    # national held-out grids (ensemble), and WC played grids
    natl_grids = [H.ens_grid(c["te_lh"][:, i], c["te_la"][:, i], 0.0) for i in np.where(te)[0]]
    nhs, nas = c["te_hg"][te], c["te_ag"][te]
    w = c["wc"]; wc_grids = [H.ens_grid(w["lh"][i], w["la"][i], 0.0) for i in range(len(w["keys"]))]
    whs, was = w["hs"], w["as_"]

    # empirical-corrected max-exact: argmax of (1-a)*P + a*empirical  (shifts toward real common scores)
    def emp_maxexact(alpha):
        def f(P):
            Q = (1 - alpha) * P + alpha * emp; return maxexact(Q)
        return f

    # draw-aware: if the model's draw probability is high enough, pick the most likely DRAW score (usually
    # 1-1); else empirical-corrected max-exact. Targets the draw hole directly.
    def draw_aware(thr, alpha=0.30):
        def f(P):
            ho = tg.hda_from_P(P)
            Q = (1 - alpha) * P + alpha * emp
            if ho[1] >= thr:                      # draw probability above threshold -> best draw cell
                d = np.array([Q[k, k] for k in range(P.shape[0])]); k = int(np.argmax(d)); return (k, k)
            return maxexact(Q)
        return f

    print("=== EXACT-hit count by scoreline policy (currency = exacts) ===", flush=True)
    for tag, grids, hs, as_ in [("NATL test (n=%d)" % len(natl_grids), natl_grids, nhs, nas),
                                 ("WC played (n=%d)" % len(wc_grids), wc_grids, whs, was)]:
        print(f"\n{tag}:", flush=True)
        for pname, pf in [("chalk (EV-pick)", chalk), ("max_exact", maxexact),
                          ("emp-corrected a=0.30", emp_maxexact(0.30)),
                          ("draw-aware thr=0.28", draw_aware(0.28)), ("draw-aware thr=0.33", draw_aware(0.33)),
                          ("draw-aware thr=0.38", draw_aware(0.38))]:
            ex, co, exby, totby = eval_policy(grids, hs, as_, pf)
            print(f"  {pname:22s} exact={ex:3d} correct={co:3d} | exact by H/D/A: "
                  f"{exby[0]}/{exby[1]}/{exby[2]}  (games H/D/A: {totby[0]}/{totby[1]}/{totby[2]})", flush=True)


if __name__ == "__main__":
    main()
