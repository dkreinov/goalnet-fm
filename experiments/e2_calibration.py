"""E2 — exact-score grid calibration. Jointly tune Dixon-Coles rho and a sharpening exponent gamma
(P**gamma renormalised) on VAL to maximise fantasy points (exact=3, outcome=1) — NOT RPS. Sharpening
concentrates mass on the modal scoreline (helps the 3-pt exacts); rho corrects low-score/draw cells.
A/B vs baseline (rho=0, gamma=1) on held-out TEST (all+natl) and the played WC games. No retrain.
Usage: python D:/Programming/claude/FM/experiments/e2_calibration.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import train_goals as tg


def slate_pts(lh, la, rho, gamma, hg, ag):
    tot = ex = 0
    for i in range(lh.shape[1]):
        g = H.ens_grid(lh[:, i], la[:, i], rho, gamma)
        pk = tg.ev_pick(g); pts, lab = tg.grade(pk, int(hg[i]), int(ag[i])); tot += pts; ex += lab == "exact"
    return tot, ex


def report(tag, lh, la, y, hg, ag, rho, gamma):
    grids = [H.ens_grid(lh[:, i], la[:, i], rho, gamma) for i in range(lh.shape[1])]
    s = H.score_grids(grids, y, hg, ag)
    print(f"  {tag:14s} rho={rho:+.2f} g={gamma:.2f} rps={s['rps']:.4f} pg={s['pg']:.4f} exact={s['exact']}/{s['n']}", flush=True)


def main():
    c = H.load_cache()
    vlh, vla = c["val_lh"], c["val_la"]
    grid = [(r, g) for r in [-0.1, -0.05, 0.0, 0.05, 0.1] for g in [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]]
    best = max(grid, key=lambda rg: slate_pts(vlh, vla, rg[0], rg[1], c["va_hg"], c["va_ag"]))
    bp, bex = slate_pts(vlh, vla, best[0], best[1], c["va_hg"], c["va_ag"])
    b0, e0 = slate_pts(vlh, vla, 0.0, 1.0, c["va_hg"], c["va_ag"])
    print(f"VAL: (rho,gamma)*={best}  pts {b0}->{bp}  exact {e0}->{bex}", flush=True)
    print("=== held-out TEST + WC (baseline vs calibrated) ===", flush=True)
    for rho, gamma in [(0.0, 1.0), best]:
        te = c["te_natl"].astype(bool)
        report("TEST all", c["te_lh"], c["te_la"], c["te_y"], c["te_hg"], c["te_ag"], rho, gamma)
        report("TEST natl", c["te_lh"][:, te], c["te_la"][:, te], c["te_y"][te], c["te_hg"][te], c["te_ag"][te], rho, gamma)
        w = c["wc"]; tot, ex = slate_pts(w["lh"].T, w["la"].T, rho, gamma, w["hs"], w["as_"])
        print(f"  WC played    rho={rho:+.2f} g={gamma:.2f} pts={tot}/{len(w['keys'])} exact={ex}", flush=True)


if __name__ == "__main__":
    main()
