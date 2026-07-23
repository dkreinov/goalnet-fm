"""Phase-6 Step-5 tripwire: score the 104-game WC slate with a goalnet checkpoint and compare v2 vs v1.

Mirrors predict_game's inference on the frozen wc_inputs.npz slate (base 10-dim ctx; +wc_odds.npz 5 dims
when the checkpoint was trained --odds), seed-ensembles the score grids at the checkpoint's DC rho, and
scores with the frozen metric suite against the players_imp train-empirical prior (the replay's null).
Verifies the retrained production model reproduces the replay winner and beats the archived v1 before cutover.

Usage: python experiments/ablation/tripwire_v2.py data/goalnet.pt [models/archive/goalnet_v1_20260723.pt]
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ablation"))
import train_goals as tg  # noqa: E402
import metrics  # noqa: E402

_TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}


def T(a):
    a = np.ascontiguousarray(a)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)


def score_ckpt(path, prior):
    c = torch.load(path, weights_only=False)
    A, nctx, rho = c["A"], c["nctx"], float(c["rho"])
    mu, sd, cmu, csd = c["mu"], c["sd"], c["cmu"], c["csd"]
    states = c.get("states") or [c["state"]]
    nets = []
    for st in states:
        n = tg.GoalNet(A, nctx); n.load_state_dict(st); n.eval(); nets.append(n)
    w = np.load(ROOT / "experiments" / "ablation" / "wc_inputs.npz", allow_pickle=True)
    keys = [str(k) for k in w["keys"]]
    Xh, Xa = w["Xh"].astype(np.float32), w["Xa"].astype(np.float32)
    Rh, Ra = w["Rh"].astype(np.int64), w["Ra"].astype(np.int64)
    ctx = w["ctx"].astype(np.float32)
    hg, ag = w["hs"].astype(np.float32), w["as_"].astype(np.float32)
    if c.get("odds"):
        wz = np.load(ROOT / "data" / "wc_odds.npz", allow_pickle=True)
        kmap = {str(k): wz["feats"][i].astype(np.float32) for i, k in enumerate(wz["keys"])}
        EX = np.stack([kmap.get(k, np.zeros(5, np.float32)) for k in keys]).astype(np.float32)
        ctx = np.concatenate([ctx, EX], 1)
    assert ctx.shape[1] == nctx, f"ctx dim {ctx.shape[1]} != checkpoint nctx {nctx}"
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    ctxn = ((ctx - cmu) / csd).astype(np.float32)
    grids_acc = None
    with torch.no_grad():
        for n in nets:
            lh, la = n(T(Xhn), T(Rh), T(Xan), T(Ra), T(ctxn))
            lh = np.exp(np.array(lh.tolist(), np.float32)); la = np.exp(np.array(la.tolist(), np.float32))
            gs = np.stack([tg.score_matrix(a, b, rho) for a, b in zip(lh, la)])
            grids_acc = gs if grids_acc is None else grids_acc + gs
    grids = grids_acc / len(nets)
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    s = metrics.suite(grids, y, hg, ag, prior)
    s["_flags"] = {"odds": bool(c.get("odds")), "beta": c.get("beta"), "W": c.get("W"),
                   "nseed": len(states), "nctx": nctx}
    return s


def main():
    paths = [a for a in sys.argv[1:]]
    if not paths:
        paths = [str(ROOT / "data" / "goalnet.pt")]
    # replay null: players_imp train-empirical grid (all matches — the prior the replay used)
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    y = z["y"]
    import db
    con = db.connect()
    meta = {r[0]: (r[1], r[2]) for r in con.execute("SELECT match_id,home_goals,away_goals FROM match")}
    mids = [int(m) for m in z["mids"]]
    hg = np.array([min(meta.get(m, (0, 0))[0] or 0, tg.MAXG) for m in mids])
    ag = np.array([min(meta.get(m, (0, 0))[1] or 0, tg.MAXG) for m in mids])
    prior = metrics.empirical_prior(hg, ag)
    print(f"{'checkpoint':52s} {'grid_info':>10s} {'rps':>7s} {'acc':>6s} {'ece':>6s} {'pts/g':>6s}  flags")
    for p in paths:
        s = score_ckpt(p, prior)
        f = s["_flags"]
        print(f"{Path(p).name:52s} {s['grid_info']:>+10.4f} {s['rps']:>7.4f} {s['acc']:>6.3f} "
              f"{s['ece_outcome']:>6.3f} {s['pts_g_31']:>6.3f}  odds={f['odds']} β={f['beta']} W={f['W']} "
              f"seeds={f['nseed']}")


if __name__ == "__main__":
    main()
