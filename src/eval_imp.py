"""Does adding the 10+1-imputed games to TRAINING help? Train the GoalNet on (a) the strict 48k
fully-11v11 set and (b) the expanded 68k set (<=1 imputed starter/side), and evaluate BOTH on the SAME
clean fully-graded held-out test games, so any difference is purely from the extra training data.
Usage: python D:/Programming/claude/FM/src/eval_imp.py [--w 5] [--epochs 45]
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db
import train_goals as tg

NATc = {9, 10, 11, 12, 13, 14, 15}
TEST_CUT = np.datetime64("2025-08-01")
VAL_LO, VAL_HI = np.datetime64("2024-08-01"), np.datetime64("2025-08-01")


def load(npz, meta):
    z = np.load(ROOT / "data" / npz, allow_pickle=True)
    mids = [int(m) for m in z["mids"]]
    hg = np.array([min(meta.get(m, (0, 0, 0))[1] or 0, tg.MAXG) for m in mids], np.float32)
    ag = np.array([min(meta.get(m, (0, 0, 0))[2] or 0, tg.MAXG) for m in mids], np.float32)
    natl = np.array([meta.get(m, (0, 0, 0))[0] in NATc for m in mids])
    return z, np.array(mids), hg, ag, natl


def main():
    def arg(k, d):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    W = float(arg("--w", "5")); ep = int(arg("--epochs", "45"))
    con = db.connect()
    meta = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    cz = np.load(ROOT / "data" / "context.npz"); cctx, cmids = cz["ctx"], cz["mids"]
    cmap = {int(m): cctx[i] for i, m in enumerate(cmids)}; nctx = cctx.shape[1]

    zs, mids_s, hgs, ags, nats = load("players.npz", meta)
    zi, mids_i, hgi, agi, nati = load("players_imp.npz", meta)
    A = zs["Xh"].shape[2]
    ATTRS = [str(a) for a in zs["attrs"]]
    # clean test = strict-set games in the test period (the gold yardstick for both models)
    ds = zs["dates"]; test_s = ds >= TEST_CUT
    test_mids = set(int(m) for m in mids_s[test_s])
    print(f"strict {len(mids_s):,} | imp {len(mids_i):,} | clean test games {test_s.sum():,}", flush=True)

    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t):
        return np.array(t.detach().tolist(), dtype=np.float32)

    def ctxof(mid):
        return cmap.get(int(mid), np.zeros(nctx, np.float32))

    def prep(z, mids, train_mask):
        Xh = z["Xh"].astype(np.float32); Xa = z["Xa"].astype(np.float32)
        Rh = z["Rh"].astype(np.int64); Ra = z["Ra"].astype(np.int64)
        CTX = np.stack([ctxof(m) for m in mids]).astype(np.float32)
        mu = Xh[train_mask].reshape(-1, A).mean(0); sd = Xh[train_mask].reshape(-1, A).std(0) + 1e-6
        cmu = CTX[train_mask].mean(0); csd = CTX[train_mask].std(0) + 1e-6
        return ((Xh - mu) / sd).astype(np.float32), ((Xa - mu) / sd).astype(np.float32), Rh, Ra, \
               ((CTX - cmu) / csd).astype(np.float32), mu, sd, cmu, csd

    def train_eval(name, z, mids, hg, ag, natl):
        dates = z["dates"]
        # train = everything before test period and NOT a clean-test mid (avoid leak of the exact test games)
        train_mask = (dates < TEST_CUT) & np.array([int(m) not in test_mids for m in mids])
        Xhn, Xan, Rh, Ra, CTXn, mu, sd, cmu, csd = prep(z, mids, train_mask)
        g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
        Xt, Rt, Xat, Rat, Ct, hgt, agt = g(train_mask)
        wt = T(np.where(natl[train_mask], W, 1.0).astype(np.float32))
        torch.manual_seed(7); np.random.seed(7)
        net = tg.GoalNet(A, nctx)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
        bs, n = 512, Xt.size(0)
        for e in range(ep):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                lh, la = net(Xt[b], Rt[b], Xat[b], Rat[b], Ct[b])
                ((pois(lh, hgt[b]) + pois(la, agt[b])) * wt[b]).mean().backward(); opt.step()
            sched.step()
        net.eval()
        return net, (mu, sd, cmu, csd)

    # build the SHARED clean-test tensors from the strict set
    def eval_on_clean(net, scal, z, mids):
        mu, sd, cmu, csd = scal
        sel = np.array([int(m) in test_mids for m in mids])
        Xh = ((z["Xh"][sel].astype(np.float32) - mu) / sd).astype(np.float32)
        Xa = ((z["Xa"][sel].astype(np.float32) - mu) / sd).astype(np.float32)
        Rh = z["Rh"][sel].astype(np.int64); Ra = z["Ra"][sel].astype(np.int64)
        CTX = ((np.stack([ctxof(m) for m in mids[sel]]).astype(np.float32) - cmu) / csd).astype(np.float32)
        yy = z["y"][sel].astype(np.int64); natsel = nats[sel] if z is zs else None
        hgt = hgs[sel] if z is zs else None
        with torch.no_grad():
            lh, la = net(T(Xh), T(Rh), T(Xa), T(Ra), T(CTX))
        return np.exp(tonp(lh)), np.exp(tonp(la)), yy

    def metrics(lh, la, yy, sel=None):
        if sel is None:
            sel = np.ones(len(yy), bool)
        agc = ags[test_s][sel]; hgc = hgs[test_s][sel]
        P = np.array([tg.hda_from_P(tg.score_matrix(a, b)) for a, b in zip(lh[sel], la[sel])])
        acc = float((P.argmax(1) == yy[sel]).mean()); r = tg.rps(yy[sel], P)
        tot = ex = 0
        for a, b, H, Aa in zip(lh[sel], la[sel], hgc, agc):
            pk = tg.ev_pick(tg.score_matrix(a, b)); pts, lab = tg.grade(pk, int(H), int(Aa))
            tot += pts; ex += lab == "exact"
        return acc, r, tot / sel.sum(), ex / sel.sum() * 100

    # both models evaluated on the SAME clean strict-test games
    netS, scS = train_eval("strict", zs, mids_s, hgs, ags, nats)
    netI, scI = train_eval("imp", zi, mids_i, hgi, agi, nati)
    lhS, laS, yy = eval_on_clean(netS, scS, zs, mids_s)
    lhI, laI, _ = eval_on_clean(netI, scI, zs, mids_s)
    natsel = nats[test_s]
    print("\n  model    set       acc    rps     pts/g  exact%", flush=True)
    for name, lh, la in [("STRICT(48k)", lhS, laS), ("IMP(68k)", lhI, laI)]:
        a1 = metrics(lh, la, yy); a2 = metrics(lh, la, yy, natsel)
        print(f"  {name:11s} ALL   {a1[0]:.3f}  {a1[1]:.4f}  {a1[2]:.3f}  {a1[3]:.1f}", flush=True)
        print(f"  {'':11s} NATL  {a2[0]:.3f}  {a2[1]:.4f}  {a2[2]:.3f}  {a2[3]:.1f}", flush=True)


if __name__ == "__main__":
    main()
