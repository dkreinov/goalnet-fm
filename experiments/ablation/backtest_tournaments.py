"""Multi-tournament LEAKAGE-FREE backtest (memory-lean): train with a pre-2024-06 cutoff, evaluate on
held-out Euro 2024 + Copa 2024 + Nations League 24-25. Compares FM+odds vs odds-only (no FM grades) on
~214 games across 3 competitions. Loads only a ~21k subsample (club sample + all nationals + eval),
one array at a time, so the full 4GB feature tensor never sits in RAM (machine has ~3GB free)."""
import sys, gc, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import torch
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments" / "ablation"))
import train_goals as tg, db, metrics, run_ablation as RA

CUTOFF = np.datetime64("2024-06-01"); NATc = (9, 10, 11, 12, 13, 14, 15)
z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
mids = np.array([int(m) for m in z["mids"]]); dates = z["dates"]; y = z["y"].astype(np.int64)
A = len(z["attrs"]); n = len(mids)
con = db.connect()
meta = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
cmp_arr = np.array([meta.get(int(m), (0, 0, 0))[0] for m in mids])
natl = np.isin(cmp_arr, NATc)
hg = np.array([min(meta.get(int(m), (0, 0, 0))[1] or 0, tg.MAXG) for m in mids], np.float32)
ag = np.array([min(meta.get(int(m), (0, 0, 0))[2] or 0, tg.MAXG) for m in mids], np.float32)

def emask(cid, lo, hi): return (cmp_arr == cid) & (dates >= np.datetime64(lo)) & (dates < np.datetime64(hi))
EVAL = {"Euro2024": emask(10, "2024-06-01", "2024-08-01"), "Copa2024": emask(12, "2024-06-01", "2024-08-01"),
        "NL24-25": emask(11, "2024-08-01", "2025-07-01")}
pooled = np.zeros(n, bool)
for m in EVAL.values(): pooled |= m

rng = np.random.default_rng(0)
pre = dates < CUTOFF
natl_pre = np.where(pre & natl)[0]; club_pre = np.where(pre & ~natl)[0]
keep_club = rng.choice(club_pre, min(len(club_pre), 18000), replace=False)
keep = np.zeros(n, bool); keep[natl_pre] = True; keep[keep_club] = True; keep |= pooled
kidx = np.where(keep)[0]   # sorted
print(f"subset rows={len(kidx):,} (train nationals={len(natl_pre)}, train club~20k, +eval) | "
      f"eval: " + ", ".join(f"{k}={int(v.sum())}" for k, v in EVAL.items()) + f", pooled={int(pooled.sum())}", flush=True)

def load_sub(name):
    arr = z[name]; sub = np.array(arr[kidx]); del arr; gc.collect(); return sub
Xh = load_sub("Xh"); Xa = load_sub("Xa"); Rh = load_sub("Rh").astype(np.int64); Ra = load_sub("Ra").astype(np.int64)
cz = np.load(ROOT / "data" / "context.npz"); _cctx, _cmids = cz["ctx"], cz["mids"]   # materialize ONCE (never index lazy npz in a loop)
cmap = {int(m): _cctx[i] for i, m in enumerate(_cmids)}; cdim = _cctx.shape[1]
emap, edim = RA._load_extra("ctx_odds.npz")
CTX = np.stack([np.concatenate([cmap.get(int(mids[i]), np.zeros(cdim, np.float32)),
                                emap.get(int(mids[i]), np.zeros(edim, np.float32))]) for i in kidx]).astype(np.float32)
# subset-space arrays + masks (bool full-arrays indexed by kidx give subset order directly)
dS, yS, natlS, hgS, agS, midS = dates[kidx], y[kidx], natl[kidx], hg[kidx], ag[kidx], mids[kidx]
trS = dS < CUTOFF; esS = trS & (rng.random(len(kidx)) < 0.08); trS = trS & ~esS
evS = pooled[kidx]
mu = Xh[trS].reshape(-1, A).mean(0); sd = Xh[trS].reshape(-1, A).std(0) + 1e-6
cmu = CTX[trS].mean(0); csd = CTX[trS].std(0) + 1e-6
Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32); CTXn = ((CTX - cmu) / csd).astype(np.float32)
del Xh, Xa, CTX; gc.collect()
prior = metrics.empirical_prior(hgS[trS], agS[trS])

def build_D():
    return {"A": A, "nctx": CTXn.shape[1], "Xhn": Xhn, "Xan": Xan, "CTXn": CTXn, "Rh": Rh, "Ra": Ra,
            "hg": hgS, "ag": agS, "y": yS, "natl": natlS, "dates": dS, "tr": trS, "es": esS, "ev": evS,
            "decay": np.ones(len(kidx), np.float32), "npz": "players_imp.npz", "mids": midS,
            "mu": mu, "sd": sd, "cmu": cmu, "csd": csd}

def run(seeds=3, epochs=150):
    D = build_D(); TR, ES = RA.make_split_tensors(D, 1.0)
    es_r, ev_r = [], []
    for s in range(seeds):
        net, brps, ep = RA.train_one(s, D, TR, ES, beta=0.0, epochs=epochs, arch="goalnet")
        es_r.append(RA.infer(net, Xhn[esS], Rh[esS], Xan[esS], Ra[esS], CTXn[esS]))
        ev_r.append(RA.infer(net, Xhn[evS], Rh[evS], Xan[evS], Ra[evS], CTXn[evS]))
        print(f"    seed {s}: val rps={brps:.4f} (e={ep})", flush=True)
    rho = max(RA.RHOS, key=lambda r: RA.points_of(RA.grids_from(es_r, r), hgS[esS], agS[esS]))
    return ev_r, rho

def score(ev_r, rho, tag):
    idx = np.where(evS)[0]; posn = {o: i for i, o in enumerate(idx)}
    for name, mask in list(EVAL.items()) + [("POOLED", pooled)]:
        subsel = mask[kidx] & evS
        sel = [posn[o] for o in np.where(subsel)[0]]
        if not sel: continue
        gr = RA.grids_from([(lh[sel], la[sel]) for lh, la in ev_r], rho)
        s = metrics.suite(gr, yS[subsel], hgS[subsel], agS[subsel], prior)
        print(f"  {tag:9s} {name:9s} n={s['n']:3d}  grid_info={s['grid_info']:+.4f}  rps={s['rps']:.4f}  "
              f"acc={s['acc']:.3f}  exact_rate={s['exact_rate']:.3f}", flush=True)

print("\n=== FM + odds ===", flush=True); e1, r1 = run(); score(e1, r1, "FM+odds")
del e1; gc.collect(); Xhn[:] = 0.0; Xan[:] = 0.0; gc.collect()
print("\n=== odds ONLY (no FM grades) ===", flush=True); e2, r2 = run(); score(e2, r2, "odds-only")
print("\n(leakage-free: trained pre-2024-06; eval tournaments post-cutoff. Train = ~20k club sample + all nationals.)", flush=True)
