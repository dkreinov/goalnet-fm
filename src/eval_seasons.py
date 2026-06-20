"""Robustness eval: instead of one contiguous time-split (val 24-25 / test 25-26, which are adjacent and
can share trends), hold out SEASONS and measure variance. Reports leave-one-season-out (each season as
the test set, train on the rest) plus a non-adjacent pair as a single test set. Uses the GoalNet
scoreline model with national upweighting. Fixed epoch budget (no peeking at the held-out season).
Usage: python D:/Programming/claude/FM/src/eval_seasons.py [--w 5] [--epochs 45]
"""
import math
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


def season_of(d):
    """Football season label from a datetime64 date (Aug-Jul)."""
    y = d.astype("datetime64[Y]").astype(int) + 1970
    m = d.astype("datetime64[M]").astype(int) % 12 + 1
    s = y if m >= 8 else y - 1
    return f"{s}-{str(s + 1)[2:]}"


def main():
    def arg(k, d):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    W = float(arg("--w", "5")); ep = int(arg("--epochs", "45"))

    z = np.load(ROOT / "data" / "players.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    ATTRS = [str(a) for a in z["attrs"]]; A = len(ATTRS)
    cz = np.load(ROOT / "data" / "context.npz"); cctx, cmids = cz["ctx"], cz["mids"]
    cmap = {int(m): cctx[i] for i, m in enumerate(cmids)}; nctx = cctx.shape[1]
    CTX = np.stack([cmap.get(m, np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)

    con = db.connect()
    meta = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([meta.get(m, (0, 0, 0))[0] in NATc for m in mids])
    hg = np.array([min(meta.get(m, (0, 0, 0))[1] or 0, tg.MAXG) for m in mids], np.float32)
    ag = np.array([min(meta.get(m, (0, 0, 0))[2] or 0, tg.MAXG) for m in mids], np.float32)
    season = np.array([season_of(d) for d in dates])
    seasons = sorted(set(season))
    print("seasons:", {s: int((season == s).sum()) for s in seasons}, flush=True)

    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t):
        return np.array(t.detach().tolist(), dtype=np.float32)

    def run(test_mask, label):
        trm = ~test_mask
        mu = Xh[trm].reshape(-1, A).mean(0); sd = Xh[trm].reshape(-1, A).std(0) + 1e-6
        Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
        cmu = CTX[trm].mean(0); csd = CTX[trm].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
        g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
        Xt, Rt, Xat, Rat, Ct, hgt, agt = g(trm)
        wt = T(np.where(natl[trm], W, 1.0).astype(np.float32))
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
        with torch.no_grad():
            lh, la = net(T(Xhn[test_mask]), T(Rh[test_mask]), T(Xan[test_mask]), T(Ra[test_mask]), T(CTXn[test_mask]))
        lh, la = np.exp(tonp(lh)), np.exp(tonp(la))
        yy = y[test_mask]; nmask = natl[test_mask]; hgt2, agt2 = hg[test_mask], ag[test_mask]
        P = np.array([tg.hda_from_P(tg.score_matrix(a, b)) for a, b in zip(lh, la)])
        def metr(sel):
            if sel.sum() == 0:
                return "n=0"
            acc = float((P[sel].argmax(1) == yy[sel]).mean()); r = tg.rps(yy[sel], P[sel])
            tot = ex = 0
            for a, b, H, Aa in zip(lh[sel], la[sel], hgt2[sel], agt2[sel]):
                pk = tg.ev_pick(tg.score_matrix(a, b)); pts, lab = tg.grade(pk, int(H), int(Aa))
                tot += pts; ex += lab == "exact"
            return f"n={int(sel.sum()):4d} acc={acc:.3f} rps={r:.4f} pts/g={tot/sel.sum():.3f} exact%={ex/sel.sum()*100:.1f}"
        allm = np.ones(len(yy), bool)
        print(f"  {label:24s} ALL  {metr(allm)}", flush=True)
        print(f"  {'':24s} NATL {metr(nmask)}", flush=True)

    print(f"\n=== leave-one-season-out (variance across seasons) ===", flush=True)
    for s in seasons:
        run(season == s, f"test={s}")
    # non-adjacent pair (skip a season between them)
    if len(seasons) >= 4:
        s1, s3 = seasons[1], seasons[3]
        print(f"\n=== non-adjacent pair test={s1}+{s3} (train on the rest incl. between/after) ===", flush=True)
        run((season == s1) | (season == s3), f"test={s1}+{s3}")
    print("\n(compare the spread across LOO rows: large variance => the single adjacent split was not "
          "representative.)", flush=True)


if __name__ == "__main__":
    main()
