"""Decision-focused experiment: train the scoreline model to directly maximize EXPECTED FANTASY POINTS
(exact=3, outcome=1) instead of only Poisson goal-NLL. The EV-pick (argmax) is non-differentiable, so we
use a soft-pick surrogate: build the score grid P(h,a) from the model, compute EV(i,j) for every scoreline,
take pi = softmax(EV/tau), and maximize sum_ij pi(i,j) * points((i,j), real_score). Anchored by Poisson
NLL (total = Poisson - beta * expected_points). A/B vs beta=0 (current model) on a held-out season.
Usage: python D:/Programming/claude/FM/src/train_comp.py [--test 2024-25] [--betas 0,1,3] [--epochs 50]
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
G = 7                     # scoreline grid 0..G (8x8)
TAU = 0.08               # soft-pick temperature (-> hard EV-pick as tau->0)


def season_of(d):
    y = d.astype("datetime64[Y]").astype(int) + 1970
    m = d.astype("datetime64[M]").astype(int) % 12 + 1
    return f"{(y if m>=8 else y-1)}-{str((y if m>=8 else y-1)+1)[2:]}"


def main():
    def arg(k, d):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    test_season = arg("--test", "2024-25"); ep = int(arg("--epochs", "50"))
    betas = [float(x) for x in arg("--betas", "0,1,3").split(",")]

    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    A = Xh.shape[2]
    cz = np.load(ROOT / "data" / "context.npz"); _cc, _cm = cz["ctx"], cz["mids"]
    cmap = {int(m): _cc[i] for i, m in enumerate(_cm)}; nctx = _cc.shape[1]
    CTX = np.stack([cmap.get(m, np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)
    con = db.connect()
    md = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([md.get(m, (0, 0, 0))[0] in NATc for m in mids])
    hg = np.array([min(md.get(m, (0, 0, 0))[1] or 0, G) for m in mids], np.int64)
    ag = np.array([min(md.get(m, (0, 0, 0))[2] or 0, G) for m in mids], np.int64)
    season = np.array([season_of(d) for d in dates])
    te = season == test_season; tr = ~te
    print(f"test={test_season} n={te.sum()}  train n={tr.sum()}  grid={G+1}x{G+1} tau={TAU}", flush=True)

    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)

    # precompute grid helpers (torch)
    ii = torch.arange(G + 1)
    I = ii.view(G + 1, 1).expand(G + 1, G + 1)            # home goals per cell
    J = ii.view(1, G + 1).expand(G + 1, G + 1)            # away goals per cell
    O = torch.where(I > J, 0, torch.where(I == J, 1, 2))  # outcome id per cell (0 H,1 D,2 A)
    logfac = torch.lgamma(ii.float() + 1)

    def grid(lh, la):                                     # P(h,a) double-Poisson, (B,G+1,G+1)
        ph = torch.exp(ii.float().view(1, -1) * torch.log(lh.view(-1, 1).clamp(min=1e-6)) - lh.view(-1, 1) - logfac.view(1, -1))
        pa = torch.exp(ii.float().view(1, -1) * torch.log(la.view(-1, 1).clamp(min=1e-6)) - la.view(-1, 1) - logfac.view(1, -1))
        P = ph.unsqueeze(2) * pa.unsqueeze(1)
        return P / P.sum(dim=[1, 2], keepdim=True).clamp(min=1e-9)

    def exp_points(lh, la, th, ta):                      # differentiable expected points of the soft EV-pick
        P = grid(lh, la)
        oprob = torch.stack([P[:, I > J].sum(1) if False else torch.tril(P, -1).sum([1, 2]),
                             torch.diagonal(P, dim1=1, dim2=2).sum(1),
                             torch.triu(P, 1).sum([1, 2])], dim=1)            # (B,3)
        EV = 2 * P + oprob[:, O]                                            # (B,G+1,G+1)
        pi = torch.softmax(EV.reshape(EV.size(0), -1) / TAU, dim=1).reshape_as(EV)
        exact = (I.unsqueeze(0) == th.view(-1, 1, 1)) & (J.unsqueeze(0) == ta.view(-1, 1, 1))
        Otru = torch.where(th > ta, 0, torch.where(th == ta, 1, 2))         # (B,)
        omatch = (O.unsqueeze(0) == Otru.view(-1, 1, 1))
        R = 3.0 * exact.float() + 1.0 * (omatch & ~exact).float()
        return (pi * R).sum([1, 2])                                          # (B,)

    g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
    Xt, Rt, Xat, Rat, Ct, hgt, agt = g(tr)
    wt = T(np.where(natl[tr], 5.0, 1.0).astype(np.float32))
    Eh, Erh, Ea, Era, Ce, _, _ = g(te)

    def run(beta):
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
                pl = ((pois(lh, hgt[b].float()) + pois(la, agt[b].float())) * wt[b]).mean()
                loss = pl
                if beta > 0:
                    ep_ = exp_points(torch.exp(lh), torch.exp(la), hgt[b], agt[b])
                    loss = pl - beta * (ep_ * wt[b]).mean()
                loss.backward(); opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            lh, la = net(Eh, Erh, Ea, Era, Ce)
        lh, la = np.exp(tonp(lh)), np.exp(tonp(la))
        ye, hge, age_ = y[te], hg[te], ag[te]
        P = np.array([tg.hda_from_P(tg.score_matrix(a, b)) for a, b in zip(lh, la)])
        acc = float((P.argmax(1) == ye).mean()); r = tg.rps(ye, P)
        tot = ex = 0
        for a, b, H, Aa in zip(lh, la, hge, age_):
            pk = tg.ev_pick(tg.score_matrix(a, b)); pts, lab = tg.grade(pk, int(H), int(Aa)); tot += pts; ex += lab == "exact"
        nm = natl[te]; accn = float((P[nm].argmax(1) == ye[nm]).mean()) if nm.sum() else 0
        print(f"  beta={beta:<4} acc={acc:.3f} rps={r:.4f} pts/g={tot/te.sum():.4f} exact%={ex/te.sum()*100:.1f} natl_acc={accn:.3f}", flush=True)

    print("\ndecision-focused vs poisson-only (held-out season):", flush=True)
    for b in betas:
        run(b)


if __name__ == "__main__":
    main()
