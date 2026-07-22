"""Arm B1 — inference-time market blend (Phase 4). No training.

Loads the cached combo-beta0-w1 per-seed rates, seed-averages the score grids, and blends their
OUTCOME masses toward the de-vigged market H/D/A on the odds-covered matches:
    target_masses = (1-λ) * model_masses + λ * market_probs
    grid' = grid rescaled per outcome region (win/draw/loss triangles) to hit target_masses
(scoreline SHAPE stays the model's; outcome OPINION shifts toward the market). λ tuned on the
odds-covered EARLYSTOP subset by grid-NLL, evaluated on the odds-covered EVAL subsets —
all comparisons model-vs-blend on the SAME covered subset (apples-to-apples).

Appends a registry row 'market-blend-b1' (seeds=0, flags.blend) + regenerates the report.
Usage: python experiments/ablation/blend_market.py
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ablation"))
import metrics  # noqa: E402
import splits  # noqa: E402
import run_ablation as RA  # noqa: E402

BASE = "combo-beta0-w1"
LAMBDAS = [round(0.1 * i, 1) for i in range(10)]


def outcome_masses(grids):
    H = np.tril(grids, -1).sum((1, 2)); D = np.trace(grids, axis1=1, axis2=2)
    return np.stack([H, D, 1 - H - D], 1)


def rescale(grids, target):
    """Rescale each grid's win/draw/loss regions to the target outcome masses."""
    out = grids.copy()
    n, M, _ = grids.shape
    tl = np.tril(np.ones((M, M)), -1); dg = np.eye(M); tu = np.triu(np.ones((M, M)), 1)
    cur = outcome_masses(grids)
    for i in range(n):
        for reg, j in ((tl, 0), (dg, 1), (tu, 2)):
            if cur[i, j] > 1e-12:
                out[i] += (target[i, j] / cur[i, j] - 1.0) * (grids[i] * reg)
    return out / out.sum((1, 2), keepdims=True)


def main():
    t0 = time.time()
    z = np.load(RA.RATES / f"{BASE}.npz", allow_pickle=True)
    rho = float(z["rho"])
    d = splits.load_dataset("players_imp.npz")
    m = splits.get_masks(d["dates"], "pooled")
    oz = np.load(ROOT / "data" / "ctx_odds.npz")
    _of = oz["feats"]; _om = oz["mids"]                     # materialize once (NpzFile is lazy)
    omap = {int(mid): _of[i, :3] for i, mid in enumerate(_om)}
    mids = np.array(d["mids"])

    def lane(mask, lh2, la2):
        """(grids, y, hg, ag, market, covered_mask) for a boolean dataset mask, using cached rates."""
        g = RA.grids_from_arrays(lh2, la2, rho)
        lm = [int(x) for x in mids[mask]]
        cov = np.array([x in omap for x in lm])
        mkt = np.array([omap.get(x, np.zeros(3)) for x in lm], np.float32)
        return g, d["y"][mask], d["hg"][mask], d["ag"][mask], mkt, cov

    # earlystop rates live in the per-seed checkpoints (es_lh/es_la) — tune λ there (plan-honest:
    # tuning never touches the eval lane).
    es_lh, es_la = [], []
    for s in range(5):
        sc = np.load(RA.RATES / f"{BASE}.s{s}.npz")
        es_lh.append(sc["es_lh"]); es_la.append(sc["es_la"])
    es = m["earlystop"]
    g_es = RA.grids_from([(es_lh[i], es_la[i]) for i in range(5)], rho)
    es_mids = [int(x) for x in mids[es]]
    es_cov = np.array([x in omap for x in es_mids])
    es_mkt = np.array([omap.get(x, np.zeros(3)) for x in es_mids], np.float32)

    ev = m["eval"]
    g, y, hg, ag, mkt, cov = lane(ev, z["ev_lh"], z["ev_la"])
    natl = d["natl"][ev]
    prior = metrics.empirical_prior(d["hg"][m["train"]], d["ag"][m["train"]])

    club_cov = cov & ~natl
    natl_cov = cov & natl
    print(f"earlystop covered: {int(es_cov.sum())}/{len(es_mids)}; "
          f"eval covered: club {int(club_cov.sum())}, natl {int(natl_cov.sum())}", flush=True)

    def gridnll(grids, hgs, ags):
        M_ = grids.shape[1]
        hc = np.minimum(hgs.astype(int), M_ - 1); ac = np.minimum(ags.astype(int), M_ - 1)
        p = grids[np.arange(len(grids)), hc, ac]
        return float(-np.mean(np.log(np.maximum(p, 1e-12))))

    # tune λ on the odds-covered EARLYSTOP subset (never the eval lane)
    best_lam, best_nll = 0.0, 1e9
    gt = g_es[es_cov]; hgt = d["hg"][es][es_cov]; agt = d["ag"][es][es_cov]; mt = es_mkt[es_cov]
    mm = outcome_masses(gt)
    for lam in LAMBDAS:
        blended = rescale(gt, (1 - lam) * mm + lam * mt)
        nll = gridnll(blended, hgt, agt)
        print(f"  λ={lam:.1f}: earlystop-covered grid_nll={nll:.4f}", flush=True)
        if nll < best_nll:
            best_nll, best_lam = nll, lam
    print(f"tuned λ*={best_lam} (earlystop-covered, n={int(es_cov.sum())})", flush=True)

    # evaluate on covered subsets: model vs blend, same matches
    rows_out = {}
    for tag, msk in (("eval_club_cov", club_cov), ("eval_natl_cov", natl_cov)):
        gm, ym, hgm, agm, mM = g[msk], y[msk], hg[msk], ag[msk], mkt[msk]
        blended = rescale(gm, (1 - best_lam) * outcome_masses(gm) + best_lam * mM)
        s_model = metrics.suite(gm, ym, hgm, agm, prior)
        s_blend = metrics.suite(blended, ym, hgm, agm, prior)
        rows_out[tag] = {"model": s_model, "blend": s_blend}
        print(f"\n  {tag} (n={int(msk.sum())}):", flush=True)
        for k in ("grid_nll", "grid_info", "rps", "acc", "ece_outcome"):
            print(f"    {k:12s} model {s_model[k]:+.4f}  blend {s_blend[k]:+.4f}  Δ {s_blend[k]-s_model[k]:+.4f}", flush=True)

    commit, dirty = RA.git_info()
    row = {"name": "market-blend-b1", "ts": datetime.now(timezone.utc).isoformat(),
           "git_commit": commit, "dirty": dirty,
           "config": {"npz": "players_imp.npz", "split": "pooled", "beta": 0.0, "W": 1.0,
                      "seeds": 0, "epochs": 0, "rho_policy": f"from:{BASE}", "ctx_extra": [],
                      "decay_halflife": None,
                      "flags": {"blend": "outcome-mass", "lambda": best_lam, "base": BASE,
                                "tuned_on": "earlystop odds-covered subset"},
                      "notes": "Arm B1: inference blend of combo-beta0-w1 grids toward de-vigged market"},
           "data": {"npz_mtime": datetime.fromtimestamp((ROOT / "data" / "players_imp.npz").stat().st_mtime,
                                                        timezone.utc).isoformat(),
                    "n": int(len(d["y"])), "ctx_dim": 10},
           "metrics": {f"{tag}__{who}": s for tag, pair in rows_out.items() for who, s in pair.items()},
           "wall_min": round((time.time() - t0) / 60.0, 2)}
    with open(RA.REG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\nappended registry row 'market-blend-b1' (λ*={best_lam})", flush=True)
    RA.regen_report()


if __name__ == "__main__":
    main()
