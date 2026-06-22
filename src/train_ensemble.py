"""Cheap wins on the production model: SEED ENSEMBLE + score-grid CALIBRATION.
Train N seeds of the W=15 decision-focused GoalNet, ensemble by averaging the per-match score grids, then
tune a sharpness temperature `a` on val (P -> P**a renormalised) to maximise fantasy points. A/B single vs
ensemble vs ensemble+calibration on the held-out test, overall + national.
Usage: python D:/Programming/claude/FM/src/train_ensemble.py [--seeds 5] [--epochs 60]
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


def ev_pick(P):   # grid-size-agnostic (tg.ev_pick hard-codes MAXG=9)
    G = P.shape[0]; ho = tg.hda_from_P(P); best, bs = -1, (1, 0)
    for i in range(G):
        for j in range(G):
            oc = 0 if i > j else (1 if i == j else 2); ev = 3 * P[i, j] + (ho[oc] - P[i, j])
            if ev > best: best, bs = ev, (i, j)
    return bs


def main():
    def arg(k, d):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    nseed = int(arg("--seeds", "5")); ep = int(arg("--epochs", "60"))
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
    tr = dates < np.datetime64("2024-08-01")
    va = (dates >= np.datetime64("2024-08-01")) & (dates < np.datetime64("2025-08-01"))
    te = dates >= np.datetime64("2025-08-01")
    print(f"train {tr.sum()} val {va.sum()} (natl {int((va&natl).sum())}) test {te.sum()} (natl {int((te&natl).sum())})", flush=True)

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
    g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
    Tr = g(tr); wt = T(np.where(natl[tr], W, 1.0).astype(np.float32))
    Vh, Vrh, Va_, Vra, Cv, _, _ = g(va); Eh, Erh, Ea, Era, Ce, _, _ = g(te)
    _lg = np.array([math.lgamma(k + 1) for k in range(GG + 1)])
    def npgrid(lh, la):
        k = np.arange(GG+1); ph = np.exp(k*np.log(max(lh,1e-6))-lh-_lg); pa = np.exp(k*np.log(max(la,1e-6))-la-_lg)
        P = np.outer(ph, pa); return P/P.sum()

    val_grids, test_grids = [], []
    for s in range(nseed):
        torch.manual_seed(s); np.random.seed(s); net = tg.GoalNet(A, nctx)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        bs, n = 512, Tr[0].size(0)
        for e in range(ep):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i+bs]; opt.zero_grad()
                lh, la = net(Tr[0][b], Tr[1][b], Tr[2][b], Tr[3][b], Tr[4][b])
                loss = ((pois(lh,Tr[5][b])+pois(la,Tr[6][b]))*wt[b]).mean() - BETA*(exp_points(torch.exp(lh),torch.exp(la),Tr[5][b],Tr[6][b])*wt[b]).mean()
                loss.backward(); opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            vl = net(Vh, Vrh, Va_, Vra, Cv); el = net(Eh, Erh, Ea, Era, Ce)
        vlh, vla = np.exp(tonp(vl[0])), np.exp(tonp(vl[1])); elh, ela = np.exp(tonp(el[0])), np.exp(tonp(el[1]))
        val_grids.append(np.array([npgrid(a, b) for a, b in zip(vlh, vla)]))
        test_grids.append(np.array([npgrid(a, b) for a, b in zip(elh, ela)]))
        print(f"  seed {s} done", flush=True)

    def score(grids, msk_te, a_pow=1.0):
        ye, hge, age_ = y[te][msk_te], hg[te][msk_te], ag[te][msk_te]
        G = grids[msk_te] ** a_pow; G = G / G.sum((1, 2), keepdims=True)
        P3 = np.array([tg.hda_from_P(g) for g in G]); acc = float((P3.argmax(1) == ye).mean()); r = tg.rps(ye, P3); tot = ex = 0
        for gm, H, Aa in zip(G, hge, age_):
            pk = ev_pick(gm); pts, lab = tg.grade(pk, int(H), int(Aa)); tot += pts; ex += lab == "exact"
        return acc, r, tot / msk_te.sum(), ex
    def valpts(a_pow):
        ens = np.mean(val_grids, 0); ye, hgv, agv = y[va], hg[va], ag[va]; G = ens ** a_pow; G = G / G.sum((1, 2), keepdims=True); tot = 0
        for gm, H, Aa in zip(G, hgv, agv):
            pk = ev_pick(gm); pts, _ = tg.grade(pk, int(H), int(Aa)); tot += pts
        return tot
    best_a = max(np.linspace(0.7, 1.5, 9), key=valpts)
    single = test_grids[0]; ens = np.mean(test_grids, 0)
    allm = np.ones(te.sum(), bool); nm = natl[te]
    print(f"\n=== ensemble({nseed}) + calibration (best a={best_a:.2f} on val) — held-out test ===", flush=True)
    for tag, G, ap in [("single seed", single, 1.0), ("ensemble", ens, 1.0), ("ensemble+calib", ens, best_a)]:
        a1 = score(G, allm, ap); a2 = score(G, nm, ap)
        print(f"  {tag:16s} ALL acc={a1[0]:.3f} rps={a1[1]:.4f} pts/g={a1[2]:.4f} ex={a1[3]} | NATL acc={a2[0]:.3f} rps={a2[1]:.4f} pts/g={a2[2]:.4f} ex={a2[3]}", flush=True)


if __name__ == "__main__":
    main()
