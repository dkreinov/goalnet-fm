"""Evaluate our ONE combined 11v11 model (xfmr + 10-feat context) specifically on NATIONAL-team matches,
and test whether upweighting nationals in training improves them without hurting the club set.

Reports, for val and test, split into ALL / CLUB / NATIONAL:
  - outcome accuracy, RPS
  - "fantasy points" under the WC league GROUP scoring (exact score = 3, correct outcome only = 1, else 0),
    using a modal-scoreline pick for our predicted outcome (H->1-0, D->1-1, A->0-1) — outcome is what the
    backtest says actually drives points; exact is ~luck. Naive (favourite by Elo-diff, 1-0) shown for ref.
Runs for several national sample-weights so you can see the trade-off.
Usage: python D:/Programming/claude/FM/src/eval_natl.py [--weights 1,3,6] [--epochs 120]
"""
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import train_pos2 as tp        # reuse PosNet/Encoder

NAT_COMPS = {9, 10, 11, 12, 13, 14, 15}
CLASSES = ["H", "D", "A"]
MODAL = {0: (1, 0), 1: (1, 1), 2: (0, 1)}     # predicted outcome -> scoreline pick


def rps(y, p):
    cp = np.cumsum(p, 1); co = np.cumsum(np.eye(3)[y], 1)
    return float(np.mean(np.sum((cp - co) ** 2, 1) / 2))


def fantasy_points(pred_out, hg, ag, exact=3, correct=1):
    """Group-stage scoring on a modal-scoreline pick. Returns (total, exact_n, outcome_n, n)."""
    tot = ex = oc = 0
    for o, h, a in zip(pred_out, hg, ag):
        ph, pa = MODAL[o]
        if ph == h and pa == a:
            tot += exact; ex += 1
        elif (ph - pa > 0) == (h - a > 0) and (ph == pa) == (h == a):
            tot += correct; oc += 1
    return tot, ex, oc, len(pred_out)


def main():
    def arg(k, d):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    weights = [float(x) for x in arg("--weights", "1,3,6").split(",")]
    ep = int(arg("--epochs", "120"))

    z = np.load(ROOT / "data" / "players.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]
    Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    A = Xh.shape[2]

    cz = np.load(ROOT / "data" / "context.npz")
    cctx, cmids = cz["ctx"], cz["mids"]        # materialize once (NpzFile indexing is lazy)
    cmap = {int(m): cctx[i] for i, m in enumerate(cmids)}
    nctx = cctx.shape[1]
    CTX = np.stack([cmap.get(m, np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)

    con = sqlite3.connect(str(ROOT / "data" / "fm.db"))
    meta = {r[0]: (r[1], r[2], r[3]) for r in
            con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([meta.get(m, (0, 0, 0))[0] in NAT_COMPS for m in mids])
    hg = np.array([meta.get(m, (0, 0, 0))[1] or 0 for m in mids])
    ag = np.array([meta.get(m, (0, 0, 0))[2] or 0 for m in mids])

    tr = dates < np.datetime64("2024-08-01")
    va = (dates >= np.datetime64("2024-08-01")) & (dates < np.datetime64("2025-08-01"))
    te = dates >= np.datetime64("2025-08-01")
    print(f"national matches: train {(tr&natl).sum()} val {(va&natl).sum()} test {(te&natl).sum()}", flush=True)

    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xh = ((Xh - mu) / sd).astype(np.float32); Xa = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6
    CTX = ((CTX - cmu) / csd).astype(np.float32)

    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a)
        return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t):
        return np.array(t.detach().tolist(), dtype=np.float32)
    pk = lambda m: (T(Xh[m]), T(Rh[m]), T(Xa[m]), T(Ra[m]), T(CTX[m]), T(y[m]))
    Xhtr, Rhtr, Xatr, Ratr, Ctr, ytr = pk(tr)
    Vh, Vrh, Va_, Vra, Cv, _ = pk(va)
    Eh, Erh, Ea, Era, Ce, _ = pk(te)

    def proba(net, A_, B_, C_, D_, E_):
        net.eval()
        with torch.no_grad():
            return tonp(torch.softmax(net(A_, B_, C_, D_, E_), 1))

    def block(tag, msk, p):
        yy = y[msk]; pr = p
        acc = float((pr.argmax(1) == yy).mean())
        r = rps(yy, pr)
        pts, ex, oc, n = fantasy_points(pr.argmax(1), hg[msk], ag[msk])
        # naive: favourite by ctx elo_diff (col 2 pre-standardization sign preserved post-standardization)
        fav = np.where(CTX[msk][:, 2] >= 0, 0, 2)
        npts, nex, noc, _ = fantasy_points(fav, hg[msk], ag[msk])
        print(f"    {tag:14s} n={n:4d} acc={acc:.3f} rps={r:.4f} | pts={pts:3d} "
              f"({pts/n:.3f}/g exact={ex} outcome={oc}) | naive_pts={npts} ({npts/n:.3f}/g)", flush=True)

    for W in weights:
        print(f"\n=== national sample-weight W={W} ===", flush=True)
        torch.manual_seed(7); np.random.seed(7)
        net = tp.PosNet(A, "xfmr", nctx=nctx)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        lossf = nn.CrossEntropyLoss(reduction="none")
        w_all = T(np.where(natl[tr], W, 1.0).astype(np.float32))
        bs, n = 512, Xhtr.size(0)
        best, best_state, bad = 9, None, 0
        for e in range(ep):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                out = net(Xhtr[b], Rhtr[b], Xatr[b], Ratr[b], Ctr[b])
                l = (lossf(out, ytr[b]) * w_all[b]).mean()
                l.backward(); opt.step()
            sched.step()
            pv = proba(net, Vh, Vrh, Va_, Vra, Cv)
            r = rps(y[va], pv)           # early stop on overall val rps (keep the combined model honest)
            if r < best - 1e-4:
                best, best_state, bad = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
            if bad >= 20:
                break
        net.load_state_dict(best_state)
        pv = proba(net, Vh, Vrh, Va_, Vra, Cv); pe = proba(net, Eh, Erh, Ea, Era, Ce)
        print("  VAL:", flush=True)
        block("all", va, pv); block("club", va & ~natl, pv[~natl[va]]); block("national", va & natl, pv[natl[va]])
        print("  TEST:", flush=True)
        block("all", te, pe); block("club", te & ~natl, pe[~natl[te]]); block("national", te & natl, pe[natl[te]])


if __name__ == "__main__":
    main()
