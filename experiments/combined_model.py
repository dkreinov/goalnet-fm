"""Mix FM (GoalNet) + FIFA (avg_fc) + market value into ONE combined predictor for the WC games and compare
to GoalNet-alone and the single blends. Each strength signal -> a double-Poisson prior grid (coefficient fit
by grid search); the three grids (GoalNet, FIFA-prior, value-prior) are mixed by weights (wg,wf,wv) on a
simplex, calibrated by LEAVE-ONE-OUT (no in-sample overfit on the mix weights). Reports rps / pts /
multiplier-weighted pts / exact on all played WC games.
Usage: python experiments/combined_model.py
"""
import sys, csv, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import train_goals as tg
ROOT = Path(__file__).resolve().parent.parent


def ev_pick(P):
    G = P.shape[0]; ho = tg.hda_from_P(P); best, bs = -1, (1, 0)
    for i in range(G):
        for j in range(G):
            oc = 0 if i > j else (1 if i == j else 2); ev = 3 * P[i, j] + (ho[oc] - P[i, j])
            if ev > best: best, bs = ev, (i, j)
    return bs


def prior_grid(sup, c, b=math.log(1.30)):
    return tg.score_matrix(math.exp(b + c * sup), math.exp(b - c * sup), 0.0)


def score(grids, hs, as_, wts):
    P3 = np.array([tg.hda_from_P(g) for g in grids]); y = np.array([0 if h > a else (1 if h == a else 2) for h, a in zip(hs, as_)])
    r = tg.rps(y, P3); tot = wtot = ex = 0.0
    for i, (g, h, a) in enumerate(zip(grids, hs, as_)):
        pk = ev_pick(g); pts, lab = tg.grade(pk, int(h), int(a)); tot += pts; wtot += pts * wts[i]; ex += lab == "exact"
    return r, int(tot), int(ex), int(wtot)


def zsup(strength, keys, col, higher, tform):
    raw = {}
    for code, r in strength.items():
        v = r.get(col)
        if v in (None, ""): continue
        v = float(v); raw[code] = math.log1p(v) if tform == "log" else v
    vals = np.array(list(raw.values())); mu, sd = vals.mean(), vals.std() + 1e-9
    z = {k: (v - mu) / sd for k, v in raw.items()}
    out = []
    for key in keys:
        a, b = key.split("-"); za, zb = z.get(a), z.get(b)
        s = (za - zb) if (za is not None and zb is not None) else 0.0
        out.append(s if higher else -s)
    return np.array(out)


def main():
    c = H.load_cache(); w = c["wc"]
    keys = [str(k) for k in w["keys"]]; hs, as_ = w["hs"], w["as_"]; n = len(keys)
    gn = [H.ens_grid(w["lh"][i], w["la"][i], 0.0) for i in range(n)]
    strength = {r["code"]: r for r in csv.DictReader(open(ROOT / "data" / "wc_team_strength.csv", encoding="utf-8"))}
    MULT = {"group": 1, "r32": 2, "r16": 4, "qf": 8, "sf": 16, "final": 32}
    try:
        sys.path.insert(0, str(ROOT)); import auto_bet as ab
        bearer, _ = ab.get_access()
        rnd = {frozenset({f["home_team"], f["away_team"]}): f["round"] for f in ab.api(ab.BASE + "/rest/v1/fixtures?select=home_team,away_team,round", bearer=bearer)}
        wts = np.array([MULT.get(rnd.get(frozenset(k.split("-")), "group"), 1) for k in keys])
    except Exception:
        wts = np.ones(n)
    sup_fc = zsup(strength, keys, "avg_fc", True, "lin")
    sup_val = zsup(strength, keys, "squad_value_eur", True, "log")
    # fit each prior's coefficient (grid search on all games; 1 param each, low overfit)
    CS = [0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
    cf = min(CS, key=lambda cc: score([prior_grid(sup_fc[i], cc) for i in range(n)], hs, as_, wts)[0])
    cv = min(CS, key=lambda cc: score([prior_grid(sup_val[i], cc) for i in range(n)], hs, as_, wts)[0])
    FC = [prior_grid(sup_fc[i], cf) for i in range(n)]
    VAL = [prior_grid(sup_val[i], cv) for i in range(n)]

    # weight simplex over (GoalNet, FIFA, value)
    W = [(g/4, f/4, v/4) for g in range(5) for f in range(5) for v in range(5) if g+f+v == 4]
    def mixgrid(i, wt3):
        wg, wf, wv = wt3; P = wg*gn[i] + wf*FC[i] + wv*VAL[i]; return P/P.sum()

    def loo(cands):
        """leave-one-out over candidate weight-tuples, minimising RPS on the other n-1, eval on i."""
        grids = []
        for i in range(n):
            idx = [j for j in range(n) if j != i]
            best = min(cands, key=lambda t: score([mixgrid(j, t) for j in idx], hs[idx], as_[idx], wts[idx])[0])
            grids.append(mixgrid(i, best))
        return grids

    print(f"=== combined FM+FIFA+value on {n} played WC games (LOO-calibrated mix; wtd=multiplier-weighted) ===", flush=True)
    print(f"  fitted prior coeffs: FIFA c={cf}, value c={cv}", flush=True)
    rows = [
        ("GoalNet only (FM)", [gn[i] for i in range(n)]),
        ("GoalNet+FIFA", loo([(g/4, f/4, 0) for g in range(5) for f in range(5) if g+f == 4])),
        ("GoalNet+value", loo([(g/4, 0, v/4) for g in range(5) for v in range(5) if g+v == 4])),
        ("GoalNet+FIFA+value (mix)", loo(W)),
    ]
    print(f"  {'model':26s} {'rps':>7} {'pts':>4} {'wtd':>4} {'exact':>5}", flush=True)
    for name, grids in rows:
        r, tot, ex, wt = score(grids, hs, as_, wts)
        print(f"  {name:26s} {r:7.4f} {tot:4d} {wt:4d} {ex:5d}", flush=True)
    # also the in-sample best mix (ceiling) for reference
    best = min(W, key=lambda t: score([mixgrid(i, t) for i in range(n)], hs, as_, wts)[0])
    r, tot, ex, wt = score([mixgrid(i, best) for i in range(n)], hs, as_, wts)
    print(f"  {'(in-sample best mix)':26s} {r:7.4f} {tot:4d} {wt:4d} {ex:5d}   weights g/f/v={best}", flush=True)


if __name__ == "__main__":
    main()
