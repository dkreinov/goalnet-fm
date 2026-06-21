"""Two ablation sweeps on the decision-focused scoreline model, held-out season:
  (1) NATIONAL WEIGHT: train with national upweight W in {1,3,5,8,15,30} -> ALL vs NATIONAL performance.
  (2) DATA FRACTION: train on a random {100,90,80,70,60,50}% of the train set -> data-scaling curve.
Reports acc / rps / fantasy-pts (overall and national). Usage: python src/train_ablate.py [--epochs 50]
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
GG, TAU, BETA = 7, 0.08, 3.0


def season_of(d):
    y = d.astype("datetime64[Y]").astype(int) + 1970; m = d.astype("datetime64[M]").astype(int) % 12 + 1
    return f"{(y if m>=8 else y-1)}-{str((y if m>=8 else y-1)+1)[2:]}"


def main():
    ep = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 50
    test_season = sys.argv[sys.argv.index("--test") + 1] if "--test" in sys.argv else "2024-25"
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
    hg = np.array([min(md.get(m, [0]*4)[1] or 0, GG) for m in mids], np.int64)
    ag = np.array([min(md.get(m, [0]*4)[2] or 0, GG) for m in mids], np.int64)
    season = np.array([season_of(d) for d in dates]); te = season == test_season; tr = ~te
    print(f"test={test_season} n={te.sum()} (natl {int((te&natl).sum())})  train n={tr.sum()}", flush=True)

    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)
    ii = torch.arange(GG + 1); I = ii.view(GG+1,1).expand(GG+1,GG+1); J = ii.view(1,GG+1).expand(GG+1,GG+1)
    O = torch.where(I > J, 0, torch.where(I == J, 1, 2)); lf = torch.lgamma(ii.float()+1)
    def grid(lh, la):
        ph = torch.exp(ii.float().view(1,-1)*torch.log(lh.view(-1,1).clamp(min=1e-6))-lh.view(-1,1)-lf.view(1,-1))
        pa = torch.exp(ii.float().view(1,-1)*torch.log(la.view(-1,1).clamp(min=1e-6))-la.view(-1,1)-lf.view(1,-1))
        P = ph.unsqueeze(2)*pa.unsqueeze(1); return P/P.sum([1,2],keepdim=True).clamp(min=1e-9)
    def exp_points(lh, la, th, ta):
        P = grid(lh, la); op = torch.stack([torch.tril(P,-1).sum([1,2]), torch.diagonal(P,dim1=1,dim2=2).sum(1), torch.triu(P,1).sum([1,2])],1)
        EV = 2*P + op[:,O]; pi = torch.softmax(EV.reshape(EV.size(0),-1)/TAU,1).reshape_as(EV)
        ex = (I.unsqueeze(0)==th.view(-1,1,1)) & (J.unsqueeze(0)==ta.view(-1,1,1))
        Ot = torch.where(th>ta,0,torch.where(th==ta,1,2)); om = (O.unsqueeze(0)==Ot.view(-1,1,1))
        return (pi*(3.0*ex.float()+(om&~ex).float())).sum([1,2])
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
    tr_idx = np.where(tr)[0]
    Eh, Erh, Ea, Era, Ce = T(Xhn[te]), T(Rh[te]), T(Xan[te]), T(Ra[te]), T(CTXn[te])

    def train_eval(W, frac, tag):
        rng = np.random.RandomState(7)
        sel = tr_idx if frac >= 1.0 else rng.choice(tr_idx, int(len(tr_idx) * frac), replace=False)
        m = np.zeros(len(y), bool); m[sel] = True
        Xt, Rt, Xat, Rat, Ct = T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m])
        hgt, agt = T(hg[m]), T(ag[m]); wt = T(np.where(natl[m], W, 1.0).astype(np.float32))
        torch.manual_seed(7); np.random.seed(7); net = tg.GoalNet(A, nctx)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        bs, n = 512, Xt.size(0)
        for e in range(ep):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i+bs]; opt.zero_grad()
                lh, la = net(Xt[b], Rt[b], Xat[b], Rat[b], Ct[b])
                loss = ((pois(lh,hgt[b])+pois(la,agt[b]))*wt[b]).mean() - BETA*(exp_points(torch.exp(lh),torch.exp(la),hgt[b],agt[b])*wt[b]).mean()
                loss.backward(); opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            lh, la = net(Eh, Erh, Ea, Era, Ce)
        lh, la = np.exp(tonp(lh)), np.exp(tonp(la))
        def metric(sel2):
            ye, hge, age_ = y[te][sel2], hg[te][sel2], ag[te][sel2]
            P = np.array([tg.hda_from_P(tg.score_matrix(a, b)) for a, b in zip(lh[sel2], la[sel2])])
            acc = float((P.argmax(1) == ye).mean()); r = tg.rps(ye, P); tot = 0
            for a, b, H, Aa in zip(lh[sel2], la[sel2], hge, age_):
                pk = tg.ev_pick(tg.score_matrix(a, b)); pts, _ = tg.grade(pk, int(H), int(Aa)); tot += pts
            return acc, r, tot / sel2.sum()
        allm = np.ones(te.sum(), bool); nm = natl[te]
        aa = metric(allm); na = metric(nm)
        print(f"  {tag:18s} ALL acc={aa[0]:.3f} rps={aa[1]:.4f} pts/g={aa[2]:.4f} | NATL acc={na[0]:.3f} rps={na[1]:.4f} pts/g={na[2]:.4f}", flush=True)

    print("\n=== (1) NATIONAL-WEIGHT sweep (full data) ===", flush=True)
    for W in [1, 3, 5, 8, 15, 30]:
        train_eval(W, 1.0, f"W={W}")
    print("\n=== (2) DATA-FRACTION sweep (W=5, random removal) ===", flush=True)
    for fr in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        train_eval(5, fr, f"data={int(fr*100)}%")


if __name__ == "__main__":
    main()
