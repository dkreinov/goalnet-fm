"""E13 — national-specialised model (tests tonight's synthesis: club and national lanes want different models;
the WC target is purely national). Three configs, single-seed train-split, A/B on held-out test:
  A baseline  : W=15 decision-focused on ALL train matches (production recipe).
  B finetune  : A, then fine-tuned on NATIONAL-only train matches (low LR, few epochs) — transfer to the lane.
  C natl-only : trained from scratch on national-only train matches.
Report ALL and NATL test (RPS / pts/g / exact). The question: does specialising past W=15 lift the NATL lane
(the WC target), and at what cost to the broad set. Usage: python experiments/e13_national_specialize.py
"""
import sys, warnings, math
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db, train_goals as tg
NATc = {9, 10, 11, 12, 13, 14, 15}
GG, TAU, BETA, W = 7, 0.08, 3.0, 15.0


def ev_pick(P):
    G = P.shape[0]; ho = tg.hda_from_P(P); best, bs = -1, (1, 0)
    for i in range(G):
        for j in range(G):
            oc = 0 if i > j else (1 if i == j else 2); ev = 3 * P[i, j] + (ho[oc] - P[i, j])
            if ev > best: best, bs = ev, (i, j)
    return bs


def main():
    def arg(k, d): return int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    ep = arg("--epochs", 60); fep = arg("--ftepochs", 40)
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
    tr = dates < np.datetime64("2024-08-01"); te = dates >= np.datetime64("2025-08-01")
    print(f"train {tr.sum()} (natl {int((tr&natl).sum())}) test {te.sum()} (natl {int((te&natl).sum())})", flush=True)
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)
    ii = torch.arange(GG + 1); I = ii.view(GG+1,1).expand(GG+1,GG+1); J = ii.view(1,GG+1).expand(GG+1,GG+1)
    O = torch.where(I > J, 0, torch.where(I == J, 1, 2)); lf = torch.lgamma(ii.float()+1)
    def grid_t(lh, la):
        ph = torch.exp(ii.float().view(1,-1)*torch.log(lh.view(-1,1).clamp(min=1e-6))-lh.view(-1,1)-lf.view(1,-1))
        pa = torch.exp(ii.float().view(1,-1)*torch.log(la.view(-1,1).clamp(min=1e-6))-la.view(-1,1)-lf.view(1,-1))
        P = ph.unsqueeze(2)*pa.unsqueeze(1); return P/P.sum([1,2],keepdim=True).clamp(min=1e-9)
    def exp_points(lh, la, th, ta):
        P = grid_t(lh, la); op = torch.stack([torch.tril(P,-1).sum([1,2]), torch.diagonal(P,dim1=1,dim2=2).sum(1), torch.triu(P,1).sum([1,2])],1)
        EV = 2*P + op[:,O]; pi = torch.softmax(EV.reshape(EV.size(0),-1)/TAU,1).reshape_as(EV)
        ex = (I.unsqueeze(0)==th.view(-1,1,1)) & (J.unsqueeze(0)==ta.view(-1,1,1))
        Ot = torch.where(th>ta,0,torch.where(th==ta,1,2)); om = (O.unsqueeze(0)==Ot.view(-1,1,1))
        return (pi*(3.0*ex.float()+(om&~ex).float())).sum([1,2])
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
    _lg = np.array([math.lgamma(k + 1) for k in range(GG + 1)])
    def npgrid(lh, la):
        k = np.arange(GG+1); ph = np.exp(k*np.log(max(lh,1e-6))-lh-_lg); pa = np.exp(k*np.log(max(la,1e-6))-la-_lg)
        P = np.outer(ph, pa); return P/P.sum()
    g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
    Eh, Erh, Ea, Era, Ce, _, _ = g(te)

    def train(mask, weights, epochs, init=None, lr=2e-3):
        Tr = g(mask); wt = T(weights.astype(np.float32))
        torch.manual_seed(0); np.random.seed(0); net = tg.GoalNet(A, nctx)
        if init is not None: net.load_state_dict(init)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        bs, n = 512, Tr[0].size(0)
        for e in range(epochs):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i+bs]; opt.zero_grad()
                lh, la = net(Tr[0][b], Tr[1][b], Tr[2][b], Tr[3][b], Tr[4][b])
                loss = ((pois(lh,Tr[5][b])+pois(la,Tr[6][b]))*wt[b]).mean() - BETA*(exp_points(torch.exp(lh),torch.exp(la),Tr[5][b],Tr[6][b])*wt[b]).mean()
                loss.backward(); opt.step()
            sched.step()
        net.eval(); return net

    def evalnet(net, tag):
        with torch.no_grad():
            el = net(Eh, Erh, Ea, Era, Ce)
        elh, ela = np.exp(tonp(el[0])), np.exp(tonp(el[1])); grids = np.array([npgrid(a, b) for a, b in zip(elh, ela)])
        def sc(msk):
            P3 = np.array([tg.hda_from_P(gr) for gr in grids[msk]]); r=tg.rps(y[te][msk],P3); tot=ex=0
            for gm,Hh,Aa in zip(grids[msk], hg[te][msk], ag[te][msk]):
                pk=ev_pick(gm); pts,lab=tg.grade(pk,int(Hh),int(Aa)); tot+=pts; ex+=lab=="exact"
            return r,tot/msk.sum(),ex
        allm=np.ones(te.sum(),bool); nm=natl[te]
        ra=sc(allm); rn=sc(nm)
        print(f"  {tag:24s} ALL rps={ra[0]:.4f} pg={ra[1]:.4f} ex={ra[2]} | NATL rps={rn[0]:.4f} pg={rn[1]:.4f} ex={rn[2]}", flush=True)
        return net

    print("=== E13 national specialisation (held-out test) ===", flush=True)
    wA = np.where(natl[tr], W, 1.0)
    netA = train(tr, wA, ep); evalnet(netA, "A baseline W=15")
    trn = tr & natl
    netB = train(trn, np.ones(int(trn.sum())), fep, init={k: v.clone() for k, v in netA.state_dict().items()}, lr=5e-4)
    evalnet(netB, "B +natl finetune")
    netC = train(trn, np.ones(int(trn.sum())), ep); evalnet(netC, "C natl-only scratch")


if __name__ == "__main__":
    main()
