"""Multi-tournament LEAKAGE-FREE backtest: train with a pre-2024-06 cutoff, evaluate on the held-out
Euro 2024 + Copa America 2024 + Nations League 24-25 games (all with odds+lineups). Compares FM+odds
vs odds-only (no FM grades) on a ~214-game, 3-competition sample to see if the WC2026 findings hold.
"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import torch
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments" / "ablation"))
import train_goals as tg, db, metrics, splits, run_ablation as RA

CUTOFF = np.datetime64("2024-06-01")
d = splits.load_dataset("players_imp.npz")
mids = [int(m) for m in d["mids"]]; dates = d["dates"]
A = d["Xh"].shape[-1]
# append de-vigged odds (5 dims) to the 10-dim context, masked (zeros) where absent
emap, edim = RA._load_extra("ctx_odds.npz")
ODD = np.stack([emap.get(m, np.zeros(edim, np.float32)) for m in mids]).astype(np.float32)
CTX = np.concatenate([d["CTX"], ODD], 1).astype(np.float32)

con = db.connect()
comp = {r[0]: r[1] for r in con.execute("SELECT match_id,competition_id FROM match")}
cmp_arr = np.array([comp.get(m, 0) for m in mids])
def emask(cid, lo, hi):
    return (cmp_arr == cid) & (dates >= np.datetime64(lo)) & (dates < np.datetime64(hi))
EVAL = {"Euro2024": emask(10, "2024-06-01", "2024-08-01"),
        "Copa2024": emask(12, "2024-06-01", "2024-08-01"),
        "NL24-25": emask(11, "2024-08-01", "2025-07-01")}
pooled = np.zeros(len(mids), bool)
for m in EVAL.values(): pooled |= m
tr_all = dates < CUTOFF
rng = np.random.default_rng(0); es = tr_all & (rng.random(len(mids)) < 0.08)   # early-stop val slice
tr = tr_all & ~es
print(f"train={tr.sum():,} val={es.sum():,} | eval: " + ", ".join(f"{k}={int(v.sum())}" for k, v in EVAL.items())
      + f", pooled={int(pooled.sum())}", flush=True)

hg, ag, y, natl = d["hg"], d["ag"], d["y"], d["natl"]
mu = d["Xh"][tr].reshape(-1, A).mean(0); sd = d["Xh"][tr].reshape(-1, A).std(0) + 1e-6
cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6
Xhn = ((d["Xh"] - mu) / sd).astype(np.float32); Xan = ((d["Xa"] - mu) / sd).astype(np.float32)
CTXn = ((CTX - cmu) / csd).astype(np.float32)
prior = metrics.empirical_prior(hg[tr], ag[tr])

def build_D(no_players):
    Xh2, Xa2 = Xhn.copy(), Xan.copy()
    if no_players: Xh2[:] = 0; Xa2[:] = 0
    return {"A": A, "nctx": CTX.shape[1], "Xhn": Xh2, "Xan": Xa2, "CTXn": CTXn, "Rh": d["Rh"], "Ra": d["Ra"],
            "hg": hg, "ag": ag, "y": y, "natl": natl, "dates": dates, "tr": tr, "es": es, "ev": pooled,
            "decay": np.ones(len(mids), np.float32), "npz": "players_imp.npz", "mids": d["mids"],
            "mu": mu, "sd": sd, "cmu": cmu, "csd": csd}

def run(no_players, seeds=3, epochs=150):
    D = build_D(no_players); TR, ES = RA.make_split_tensors(D, 1.0)
    es_rates, ev_rates = [], []
    for s in range(seeds):
        net, brps, ep = RA.train_one(s, D, TR, ES, beta=0.0, epochs=epochs, arch="goalnet")
        es_rates.append(RA.infer(net, D["Xhn"][es], D["Rh"][es], D["Xan"][es], D["Ra"][es], D["CTXn"][es]))
        ev_rates.append(RA.infer(net, D["Xhn"][pooled], D["Rh"][pooled], D["Xan"][pooled], D["Ra"][pooled], D["CTXn"][pooled]))
        print(f"    seed {s}: val rps={brps:.4f} (e={ep})", flush=True)
    rho = max(RA.RHOS, key=lambda r: RA.points_of(RA.grids_from(es_rates, r), hg[es], ag[es]))
    return ev_rates, rho

def score(ev_rates, rho, tag):
    idx = np.where(pooled)[0]; pos = {m: i for i, m in enumerate(idx)}
    for name, mask in list(EVAL.items()) + [("POOLED", pooled)]:
        sel = [pos[i] for i in np.where(mask)[0]]
        gr = RA.grids_from([(lh[sel], la[sel]) for lh, la in ev_rates], rho)
        s = metrics.suite(gr, y[mask], hg[mask], ag[mask], prior)
        print(f"  {tag:14s} {name:9s} n={s['n']:3d}  grid_info={s['grid_info']:+.4f}  rps={s['rps']:.4f}  "
              f"acc={s['acc']:.3f}  exact_rate={s['exact_rate']:.3f}", flush=True)

print("\n=== FM + odds ===", flush=True); ev1, rho1 = run(False); score(ev1, rho1, "FM+odds")
print("\n=== odds ONLY (no FM grades) ===", flush=True); ev2, rho2 = run(True); score(ev2, rho2, "odds-only")
print("\n(leakage-free: trained on data before 2024-06-01; eval tournaments are all post-cutoff)", flush=True)
