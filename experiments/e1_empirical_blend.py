"""E1 — empirical score-prior blend. Football scores cluster (1-1, 1-0, 2-1, ...) more than independent
double-Poisson predicts. Blend P=(1-a)*model + a*empirical, tune a on VAL for fantasy pts (and watch exact-
hit), then A/B on held-out TEST (all+natl) and the played WC games. No retrain — reuses eval_harness cache.
Usage: python D:/Programming/claude/FM/experiments/e1_empirical_blend.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import train_goals as tg


def slate_pts(lh, la, emp, alpha, hg, ag):
    tot = ex = 0
    for i in range(lh.shape[1]):
        g = H.ens_grid(lh[:, i], la[:, i], 0.0, 1.0, emp, alpha)
        pk = tg.ev_pick(g); pts, lab = tg.grade(pk, int(hg[i]), int(ag[i])); tot += pts; ex += lab == "exact"
    return tot, ex


def report(tag, lh, la, y, hg, ag, emp, alpha):
    idx = range(lh.shape[1])
    grids = [H.ens_grid(lh[:, i], la[:, i], 0.0, 1.0, emp, alpha) for i in idx]
    s = H.score_grids(grids, y, hg, ag)
    print(f"  {tag:14s} a={alpha:.2f} rps={s['rps']:.4f} pg={s['pg']:.4f} exact={s['exact']}/{s['n']}", flush=True)


def main():
    c = H.load_cache(); emp = H.empirical_grid()
    print("empirical top cells:", [(f"{i}-{j}", round(float(emp[i, j]), 3))
          for i, j in sorted(np.ndindex(emp.shape), key=lambda t: -emp[t])[:6]], flush=True)
    # tune alpha on VAL by pts (tie-break exact)
    vlh, vla = c["val_lh"], c["val_la"]
    best = max(np.round(np.linspace(0, 0.6, 13), 3),
               key=lambda a: slate_pts(vlh, vla, emp, a, c["va_hg"], c["va_ag"]))
    bp, bex = slate_pts(vlh, vla, emp, best, c["va_hg"], c["va_ag"])
    b0, e0 = slate_pts(vlh, vla, emp, 0.0, c["va_hg"], c["va_ag"])
    print(f"VAL: alpha*={best}  pts {b0}->{bp}  exact {e0}->{bex}", flush=True)
    print("=== held-out TEST + WC (alpha=0 baseline vs alpha*) ===", flush=True)
    for a in (0.0, best):
        te = c["te_natl"].astype(bool)
        report("TEST all", c["te_lh"], c["te_la"], c["te_y"], c["te_hg"], c["te_ag"], emp, a)
        report("TEST natl", c["te_lh"][:, te], c["te_la"][:, te], c["te_y"][te], c["te_hg"][te], c["te_ag"][te], emp, a)
        w = c["wc"]; tot, ex = slate_pts(w["lh"].T, w["la"].T, emp, a, w["hs"], w["as_"])
        print(f"  WC played    a={a:.2f} pts={tot}/{len(w['keys'])} exact={ex}", flush=True)


if __name__ == "__main__":
    main()
