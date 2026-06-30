"""Study 2 (NATIONAL/WC lane): does a market-value / FIFA / rank / Elo strength prior improve the FM-GoalNet
predictions on the actual played WC games? We can't retrain (these signals exist only for the 48 WC teams,
not the 69k training set), so this is a prediction-time blend on the wc_cache grids + data/wc_team_strength.csv.
For each signal we build a double-Poisson strength prior, calibrate its coefficient + GoalNet-blend weight by
LEAVE-ONE-OUT (no in-sample overfit on the small slate), and report exact / points / RPS vs GoalNet-only.
Honest caveat: n~48, low statistical power — read direction, not decimals.
Usage: python experiments/value_national.py
"""
import sys, csv, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import train_goals as tg

SIGNALS = [("squad_value_eur", True, "log"), ("top11_value_eur", True, "log"), ("avg_fc", True, "lin"),
           ("fifa_rank", False, "lin"), ("elo", True, "lin")]   # (col, higher_is_better, transform)


def ev_pick(P):
    G = P.shape[0]; ho = tg.hda_from_P(P); best, bs = -1, (1, 0)
    for i in range(G):
        for j in range(G):
            oc = 0 if i > j else (1 if i == j else 2); ev = 3 * P[i, j] + (ho[oc] - P[i, j])
            if ev > best: best, bs = ev, (i, j)
    return bs


def prior_grid(sup, c, b=math.log(1.30)):
    """Double-Poisson grid from a standardized strength supremacy: home/away log-rates = b ± c*sup."""
    return tg.score_matrix(math.exp(b + c * sup), math.exp(b - c * sup), 0.0)


def score_set(grids, hs, as_):
    P3 = np.array([tg.hda_from_P(g) for g in grids]); y = np.array([0 if h > a else (1 if h == a else 2) for h, a in zip(hs, as_)])
    r = tg.rps(y, P3); tot = ex = 0
    for g, h, a in zip(grids, hs, as_):
        pk = ev_pick(g); pts, lab = tg.grade(pk, int(h), int(a)); tot += pts; ex += lab == "exact"
    return r, tot, ex


def main():
    c = H.load_cache(); w = c["wc"]
    keys = [str(k) for k in w["keys"]]; hs = w["hs"]; as_ = w["as_"]
    gn = [H.ens_grid(w["lh"][i], w["la"][i], 0.0) for i in range(len(keys))]   # GoalNet grids
    strength = {r["code"]: r for r in csv.DictReader(open(Path(__file__).resolve().parent.parent / "data" / "wc_team_strength.csv", encoding="utf-8"))}
    n = len(keys)
    base = score_set(gn, hs, as_)
    print(f"=== Study 2: strength-prior blends on {n} played WC games (leave-one-out calibrated) ===", flush=True)
    print(f"  {'GoalNet (FM) baseline':28s} rps={base[0]:.4f} pts={base[1]} exact={base[2]}", flush=True)

    def supremacy(col, higher, tform):
        raw = {}
        for code, r in strength.items():
            v = r.get(col)
            if v in (None, ""): continue
            v = float(v); raw[code] = math.log1p(v) if tform == "log" else v
        vals = np.array(list(raw.values())); mu, sd = vals.mean(), vals.std() + 1e-9
        z = {k: (v - mu) / sd for k, v in raw.items()}
        sup = []
        for key in keys:
            a, b = key.split("-"); za, zb = z.get(a), z.get(b)
            s = (za - zb) if (za is not None and zb is not None) else 0.0
            sup.append(s if higher else -s)
        return np.array(sup)

    CS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]; WS = [0.0, 0.15, 0.3, 0.45, 0.6]
    for col, higher, tform in SIGNALS:
        sup = supremacy(col, higher, tform)
        # leave-one-out: pick (c) for the standalone prior and (w) for the blend on the other 47, eval on i
        prior_grids, blend_grids = [], []
        for i in range(n):
            idx = [j for j in range(n) if j != i]
            bestc = min(CS, key=lambda cc: score_set([prior_grid(sup[j], cc) for j in idx], hs[idx], as_[idx])[0])
            pg_i = prior_grid(sup[i], bestc); prior_grids.append(pg_i)
            bestw = min(WS, key=lambda ww: score_set([(1-ww)*gn[j] + ww*prior_grid(sup[j], bestc) for j in idx], hs[idx], as_[idx])[0])
            bg = (1 - bestw) * gn[i] + bestw * pg_i; blend_grids.append(bg / bg.sum())
        pr = score_set(prior_grids, hs, as_); bl = score_set(blend_grids, hs, as_)
        print(f"  {col:16s} prior-only rps={pr[0]:.4f} pts={pr[1]} ex={pr[2]}  |  GoalNet+{col[:8]} rps={bl[0]:.4f} pts={bl[1]} ex={bl[2]}", flush=True)


if __name__ == "__main__":
    main()
