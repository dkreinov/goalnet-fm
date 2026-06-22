"""E6 — squad experience / cohesion feature. New context feature class (the one lever the data says is still
open): for each match, the mean number of prior starts (career appearances-to-date, leakage-free) of the home
XI and away XI, plus lineup CONTINUITY (fraction of the XI that also started the team's previous match). These
capture squad experience + stability, expected to matter most for the NATIONAL lane. A/B: retrain the W=15
decision-focused GoalNet with base 10-feat context vs context+experience, on the held-out test (all + natl).
Single seed, train split (no leakage). Usage: python experiments/e6_cohesion.py [--epochs 60]
"""
import sys, warnings, math
from pathlib import Path
from collections import defaultdict
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


def build_experience2(mids):
    """Per-match [home_xi_mean_prior_starts, away_xi_mean_prior_starts, home_continuity, away_continuity].
    Leakage-free: counts/prev-XI use only matches strictly before the current one (date order)."""
    con = db.connect()
    mdate = {r[0]: r[1] for r in con.execute("SELECT match_id,match_date FROM match WHERE home_goals IS NOT NULL")}
    hac = {r[0]: (r[1], r[2]) for r in con.execute("SELECT match_id,home_club_id,away_club_id FROM match")}
    starters = defaultdict(list)
    for mid, pid, club in con.execute("SELECT match_id,player_id,club_id FROM match_player WHERE started=1"):
        starters[mid].append((pid, club))
    con.close()
    order = sorted(mdate, key=lambda m: (str(mdate[m]), m))
    cnt = defaultdict(int); prevxi = {}
    feat = {}
    for m in order:
        home, away = hac.get(m, (None, None))
        hs, as_, hx, ax = [], [], set(), set()
        for pid, club in starters.get(m, []):
            (hs if club == home else as_).append(cnt[pid])
            (hx if club == home else ax).add(pid)
        def cont(xi, club):
            pv = prevxi.get(club)
            return (len(xi & pv) / len(xi)) if pv and xi else 0.0
        feat[m] = [np.log1p(np.mean(hs) if hs else 0), np.log1p(np.mean(as_) if as_ else 0),
                   cont(hx, home), cont(ax, away)]
        for pid, club in starters.get(m, []): cnt[pid] += 1
        if hx: prevxi[home] = hx
        if ax: prevxi[away] = ax
    return np.array([feat.get(m, [0, 0, 0, 0]) for m in mids], np.float32)


def main():
    def arg(k, d): return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    ep = int(arg("--epochs", "60"))
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    A = Xh.shape[2]
    cz = np.load(ROOT / "data" / "context.npz"); _cc, _cm = cz["ctx"], cz["mids"]
    cmap = {int(m): _cc[i] for i, m in enumerate(_cm)}; nctx0 = _cc.shape[1]
    CTX0 = np.stack([cmap.get(m, np.zeros(nctx0, np.float32)) for m in mids]).astype(np.float32)
    print("building experience features...", flush=True)
    EXP = build_experience2(mids)
    print(f"exp feature means (all): {np.round(EXP.mean(0), 3)}", flush=True)
    con = db.connect()
    md = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([md.get(m, (0, 0, 0))[0] in NATc for m in mids])
    hg = np.array([min(md.get(m, [0]*4)[1] or 0, GG) for m in mids], np.int64)
    ag = np.array([min(md.get(m, [0]*4)[2] or 0, GG) for m in mids], np.int64)
    tr = dates < np.datetime64("2024-08-01")
    va = (dates >= np.datetime64("2024-08-01")) & (dates < np.datetime64("2025-08-01"))
    te = dates >= np.datetime64("2025-08-01")
    print(f"train {tr.sum()} val {va.sum()} test {te.sum()} (natl-te {int((te&natl).sum())})", flush=True)
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
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

    def run(CTX, tag):
        nctx = CTX.shape[1]
        cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
        g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
        Tr = g(tr); wt = T(np.where(natl[tr], W, 1.0).astype(np.float32)); Eh, Erh, Ea, Era, Ce, _, _ = g(te)
        torch.manual_seed(0); np.random.seed(0); net = tg.GoalNet(A, nctx)
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
            el = net(Eh, Erh, Ea, Era, Ce)
        elh, ela = np.exp(tonp(el[0])), np.exp(tonp(el[1]))
        grids = np.array([npgrid(a, b) for a, b in zip(elh, ela)])
        def sc(msk):
            P3 = np.array([tg.hda_from_P(g) for g in grids[msk]]); acc=float((P3.argmax(1)==y[te][msk]).mean()); r=tg.rps(y[te][msk],P3); tot=ex=0
            for gm,H,Aa in zip(grids[msk], hg[te][msk], ag[te][msk]):
                pk=ev_pick(gm); pts,lab=tg.grade(pk,int(H),int(Aa)); tot+=pts; ex+=lab=="exact"
            return acc,r,tot/msk.sum(),ex
        allm=np.ones(te.sum(),bool); nm=natl[te]
        a1,a2=sc(allm),sc(nm)
        print(f"  {tag:16s} ALL acc={a1[0]:.3f} rps={a1[1]:.4f} pg={a1[2]:.4f} ex={a1[3]} | NATL rps={a2[1]:.4f} pg={a2[2]:.4f} ex={a2[3]}", flush=True)

    print("=== E6 squad-experience feature A/B (held-out test) ===", flush=True)
    run(CTX0, "base ctx(10)")
    run(np.concatenate([CTX0, EXP], 1), "ctx+exp(14)")
    run(np.concatenate([CTX0, EXP[:, :2]], 1), "ctx+apps(12)")


if __name__ == "__main__":
    main()
