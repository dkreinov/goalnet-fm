"""Frozen metric suite for the ablation harness (contract in DESIGN.md — do not rename metrics).

All functions take per-match score grids P (list/array of (M,M), rows=home goals, cols=away goals,
each summing to 1) plus truth arrays. Grid size M = MAXG+1 follows train_goals (MAXG=9).
Self-test: python experiments/ablation/metrics.py --selftest
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import train_goals as tg  # noqa: E402  (ev_pick, grade, hda_from_P, rps, MAXG)

EPS = 1e-12


def outcome_probs(grids):
    """(n,3) H/D/A probabilities from score grids."""
    return np.array([tg.hda_from_P(g) for g in grids])


def empirical_prior(hg, ag, M=tg.MAXG + 1):
    """Empirical score grid from a set of matches (USE TRAIN MASK ONLY — the null model)."""
    E = np.zeros((M, M))
    for h, a in zip(hg, ag):
        E[min(int(h), M - 1), min(int(a), M - 1)] += 1
    return E / E.sum()


def ece_outcome(P3, y, bins=10):
    """Expected calibration error, max-prob bin convention: bin matches by max outcome prob,
    compare mean confidence vs accuracy of the argmax pick within each bin."""
    conf = P3.max(1); pred = P3.argmax(1); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def suite(grids, y, hg, ag, prior):
    """The frozen per-lane metric dict. `prior` = train-split empirical grid (empirical_prior)."""
    n = len(grids)
    if n == 0:
        return {"n": 0}
    grids = np.asarray(grids); M = grids.shape[1]
    hgc = np.minimum(np.asarray(hg, int), M - 1); agc = np.minimum(np.asarray(ag, int), M - 1)
    y = np.asarray(y, int)
    P3 = outcome_probs(grids)
    idx = np.arange(n)
    cell_p = grids[idx, hgc, agc]
    prior_p = prior[hgc, agc]
    grid_nll = float(-np.mean(np.log(np.maximum(cell_p, EPS))))
    grid_nll_prior = float(-np.mean(np.log(np.maximum(prior_p, EPS))))
    # EV-pick based exact metrics (production 3/1 pick — reference behaviour)
    tot = ex = 0
    for g, H, Aa in zip(grids, hgc, agc):
        pk = tg.ev_pick(g); pts, lab = tg.grade(pk, int(H), int(Aa))
        tot += pts; ex += lab == "exact"
    modal = np.unravel_index(prior.argmax(), prior.shape)
    modal_rate = float(np.mean((hgc == modal[0]) & (agc == modal[1])))
    exact_rate = ex / n
    ent = -np.sum(grids * np.log(np.maximum(grids, EPS)), axis=(1, 2))
    return {
        "n": int(n),
        "acc": float((P3.argmax(1) == y).mean()),
        "rps": float(tg.rps(y, P3)),
        "outcome_nll": float(-np.mean(np.log(np.maximum(P3[idx, y], EPS)))),
        "grid_nll": grid_nll,
        "grid_nll_prior": grid_nll_prior,
        "grid_info": grid_nll_prior - grid_nll,
        "ece_outcome": ece_outcome(P3, y),
        "sharpness": float(ent.mean()),
        "exact_rate": float(exact_rate),
        "exact_lift": float(exact_rate / modal_rate) if modal_rate > 0 else float("nan"),
        "pts_g_31": tot / n,
        "exact_n": int(ex),
    }


def lift_table(grids, hg, ag, prior, top=12):
    """Per-scoreline diagnostics. Returns (pred_rows, true_rows).
    pred_rows: for each EV-picked scoreline — count, hit-rate (precision), prior cell prob.
    true_rows: for each common true scoreline — how often the model put it in its top-3 cells."""
    grids = np.asarray(grids); M = grids.shape[1]
    hgc = np.minimum(np.asarray(hg, int), M - 1); agc = np.minimum(np.asarray(ag, int), M - 1)
    picks = [tg.ev_pick(g) for g in grids]
    pred = {}
    for pk, H, Aa in zip(picks, hgc, agc):
        d = pred.setdefault(pk, [0, 0]); d[0] += 1; d[1] += (pk == (H, Aa))
    pred_rows = [{"score": f"{k[0]}-{k[1]}", "picked": c, "hit": h, "precision": h / c,
                  "prior_p": float(prior[k[0], k[1]])}
                 for k, (c, h) in sorted(pred.items(), key=lambda t: -t[1][0])]
    true_counts = {}
    for i, (H, Aa) in enumerate(zip(hgc, agc)):
        d = true_counts.setdefault((int(H), int(Aa)), [0, 0])
        d[0] += 1
        flat = grids[i].flatten(); top3 = np.argpartition(flat, -3)[-3:]
        d[1] += int(H * M + Aa in top3)
    true_rows = [{"score": f"{k[0]}-{k[1]}", "n": c, "top3_recall": r / c}
                 for k, (c, r) in sorted(true_counts.items(), key=lambda t: -t[1][0])[:top]]
    return pred_rows, true_rows


def reliability(grids, y, hg, ag, bins=8):
    """Outcome + exact-score reliability rows: (bin_lo, bin_hi, mean_pred, observed, n)."""
    grids = np.asarray(grids); M = grids.shape[1]
    hgc = np.minimum(np.asarray(hg, int), M - 1); agc = np.minimum(np.asarray(ag, int), M - 1)
    P3 = outcome_probs(grids); idx = np.arange(len(grids))
    out = {"outcome": [], "exact": []}
    # outcome: every (match, outcome) pair
    p = P3.flatten(); hit = np.eye(3)[np.asarray(y, int)].flatten()
    # exact: probability the model put on ITS argmax cell vs whether that cell happened
    am = [np.unravel_index(g.argmax(), g.shape) for g in grids]
    pe = np.array([g.max() for g in grids])
    he = np.array([(a == (H, Aa)) for a, H, Aa in zip(am, hgc, agc)], float)
    for tag, pp, hh in [("outcome", p, hit), ("exact", pe, he)]:
        edges = np.quantile(pp, np.linspace(0, 1, bins + 1))
        for i in range(bins):
            m = (pp >= edges[i]) & (pp <= edges[i + 1] if i == bins - 1 else pp < edges[i + 1])
            if m.sum():
                out[tag].append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                                 "pred": float(pp[m].mean()), "obs": float(hh[m].mean()), "n": int(m.sum())})
    return out


def _selftest():
    M = tg.MAXG + 1
    rng = np.random.default_rng(0)
    n = 400
    hg = rng.integers(0, 4, n); ag = rng.integers(0, 4, n)
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    prior = empirical_prior(hg, ag, M)

    # 1) uniform grids: max entropy, grid_nll = log(M^2), zero grid_info vs itself-as-prior impossible,
    #    but vs empirical prior grid_info must be NEGATIVE (uniform is worse than the prior).
    uni = np.full((n, M, M), 1.0 / (M * M))
    s = suite(uni, y, hg, ag, prior)
    assert abs(s["sharpness"] - np.log(M * M)) < 1e-9, s["sharpness"]
    assert abs(s["grid_nll"] - np.log(M * M)) < 1e-9
    assert s["grid_info"] < 0

    # 2) delta-on-truth grids: nll -> 0, exact_rate = 1, acc = 1, ece ~ 0, sharpness ~ 0
    delta = np.full((n, M, M), EPS)
    for i in range(n):
        delta[i, hg[i], ag[i]] = 1.0
    delta /= delta.sum((1, 2), keepdims=True)
    s = suite(delta, y, hg, ag, prior)
    assert s["grid_nll"] < 1e-6 and s["exact_rate"] == 1.0 and s["acc"] == 1.0
    assert s["ece_outcome"] < 1e-6 and s["sharpness"] < 1e-3
    assert s["exact_lift"] > 1.0

    # 3) prior-as-grid: grid_nll == grid_nll_prior exactly (grid_info == 0);
    #    EV-pick is constant => exact_rate == modal-ish rate => lift ≈ 1 when pick == modal cell.
    pri = np.tile(prior[None], (n, 1, 1))
    s = suite(pri, y, hg, ag, prior)
    assert abs(s["grid_info"]) < 1e-12
    pk = tg.ev_pick(prior)
    modal = np.unravel_index(prior.argmax(), prior.shape)
    if pk == modal:  # EV-pick of a broad grid usually lands on the modal cell
        assert abs(s["exact_lift"] - 1.0) < 1e-9

    # 4) lift_table / reliability run and are shape-sane
    pr, trr = lift_table(pri, hg, ag, prior)
    assert pr and trr and abs(sum(r["picked"] for r in pr) - n) < 1e-9
    rel = reliability(delta, y, hg, ag)
    assert rel["outcome"] and rel["exact"]
    print("SELFTEST PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
