"""Diagnostic: does the GoalNet EV-pick ever choose DRAWS, on the eval data? Trains the split model
(train<2024-08), predicts val+test, and reports actual draw rate vs EV-pick draw rate vs how many actual
draws it catches — at the val-tuned rho AND at a draw-inflating rho, to show the calibration lever.
Usage: python D:/Programming/claude/FM/src/draw_diag.py [--epochs 60]
"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db, train_goals as tg
NATc = {9, 10, 11, 12, 13, 14, 15}


def main():
    ep = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 60
    npz = sys.argv[sys.argv.index("--npz") + 1] if "--npz" in sys.argv else "players_imp.npz"
    z = np.load(ROOT / "data" / npz, allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    A = Xh.shape[2]
    cz = np.load(ROOT / "data" / "context.npz"); cctx, cmids = cz["ctx"], cz["mids"]
    cmap = {int(m): cctx[i] for i, m in enumerate(cmids)}; nctx = cctx.shape[1]
    CTX = np.stack([cmap.get(m, np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)
    con = db.connect()
    meta = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    hg = np.array([min(meta.get(m, (0, 0, 0))[1] or 0, tg.MAXG) for m in mids], np.float32)
    ag = np.array([min(meta.get(m, (0, 0, 0))[2] or 0, tg.MAXG) for m in mids], np.float32)
    natl = np.array([meta.get(m, (0, 0, 0))[0] in NATc for m in mids])
    tr = dates < np.datetime64("2024-08-01")
    ev = dates >= np.datetime64("2024-08-01")     # val+test = the held-out eval
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)
    g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
    Xt, Rt, Xat, Rat, Ct, hgt, agt = g(tr)
    wt = T(np.where(natl[tr], 5.0, 1.0).astype(np.float32))
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
        lh, la = net(T(Xhn[ev]), T(Rh[ev]), T(Xan[ev]), T(Ra[ev]), T(CTXn[ev]))
    lh, la = np.exp(tonp(lh)), np.exp(tonp(la))
    ye, hge, age_ = y[ev], hg[ev], ag[ev]
    n_ev = len(ye); actual_draws = int((ye == 1).sum())
    print(f"eval games={n_ev}  actual draws={actual_draws} ({actual_draws/n_ev*100:.1f}%)", flush=True)
    print(f"\n  rho     EVpick_draws   mean_P(draw)   draws_caught(correct)   total_pts", flush=True)
    for rho in [0.05, 0.0, -0.05, -0.10, -0.15]:
        picks_draw = caught = tot = 0; pdraw_sum = 0.0
        for a, b, yy, H, Aa in zip(lh, la, ye, hge, age_):
            P = tg.score_matrix(a, b, rho)
            pdraw_sum += tg.hda_from_P(P)[1]
            pk = tg.ev_pick(P)
            if pk[0] == pk[1]:
                picks_draw += 1
                if yy == 1:
                    caught += 1
            pts, _ = tg.grade(pk, int(H), int(Aa)); tot += pts
        print(f"  {rho:+.2f}   {picks_draw:5d} ({picks_draw/n_ev*100:4.1f}%)   "
              f"{pdraw_sum/n_ev*100:5.1f}%        {caught:4d}/{actual_draws}              {tot}", flush=True)
    print("\n  (val-tuned rho was +0.05. mean P(draw) is what the model BELIEVES; EVpick_draws is what it"
          " ACTS on. Gap between them + the actual-draw rate = the calibration headroom.)", flush=True)


if __name__ == "__main__":
    main()
