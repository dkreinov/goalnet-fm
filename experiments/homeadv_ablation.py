"""Steps 2+3: does venue-aware home advantage beat the single global fixture-home boost?
Three variants, single-seed train-split, held-out test (ALL + NATL):
  base    : production GoalNet (global home_adv on whoever is listed home).
  +flag   : GoalNet + [true_home, neutral, venue_known] in the context (soft venue modulation).
  +perteam: GoalNet with a per-team home_adv embedding, GATED by true_home (=per-stadium), hard weight-decayed
            so it must earn deviations from the global. Prints the learned team home-boost ranking (sanity).
Usage: python experiments/homeadv_ablation.py [--epochs 60]
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


class PerTeam(tg.GoalNet):
    def __init__(self, A, nctx, n_teams):
        super().__init__(A, nctx)
        self.team_adv = nn.Embedding(n_teams, 1); nn.init.zeros_(self.team_adv.weight)

    def forward(self, Xh, Rh, Xa, Ra, C, hidx=None, thome=None):
        th = self.enc(Xh, Rh); ta = self.enc(Xa, Ra)
        adh, ada = self.ad(th), self.ad(ta); ch, ca = self.ctx(C).unbind(-1)
        extra = (self.team_adv(hidx).squeeze(-1) * thome) if hidx is not None else 0.0
        logh = self.home_adv + extra + adh[:, 0] - ada[:, 1] + ch
        loga = ada[:, 0] - adh[:, 1] + ca
        return logh, loga


def main():
    def arg(k, d): return int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    ep = arg("--epochs", 60)
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    A = Xh.shape[2]
    cz = np.load(ROOT / "data" / "context.npz"); _cc, _cm = cz["ctx"], cz["mids"]
    cmap = {int(m): _cc[i] for i, m in enumerate(_cm)}; nctx0 = _cc.shape[1]
    CTX0 = np.stack([cmap.get(m, np.zeros(nctx0, np.float32)) for m in mids]).astype(np.float32)
    vz = np.load(ROOT / "data" / "venue.npz"); _vf, _vm, _vi = vz["feats"], vz["mids"], vz["home_idx"]  # materialize (lazy npz)
    n_teams = int(vz["n_teams"])
    vmap = {int(m): (_vf[i], int(_vi[i])) for i, m in enumerate(_vm)}
    VF = np.stack([vmap.get(m, (np.zeros(3, np.float32), 0))[0] for m in mids]).astype(np.float32)
    HID = np.array([vmap.get(m, (None, 0))[1] for m in mids], np.int64)
    con = db.connect()
    md = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([md.get(m, (0, 0, 0))[0] in NATc for m in mids])
    hg = np.array([min(md.get(m, [0]*4)[1] or 0, GG) for m in mids], np.int64)
    ag = np.array([min(md.get(m, [0]*4)[2] or 0, GG) for m in mids], np.int64)
    tr = dates < np.datetime64("2024-08-01"); te = dates >= np.datetime64("2025-08-01")
    print(f"train {tr.sum()} test {te.sum()} (natl-te {int((te&natl).sum())}) | n_teams {n_teams}", flush=True)
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a): a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)
    ii = torch.arange(GG + 1); I = ii.view(GG+1,1).expand(GG+1,GG+1); J = ii.view(1,GG+1).expand(GG+1,GG+1)
    O = torch.where(I>J,0,torch.where(I==J,1,2)); lf = torch.lgamma(ii.float()+1)
    def grid_t(lh, la):
        ph = torch.exp(ii.float().view(1,-1)*torch.log(lh.view(-1,1).clamp(min=1e-6))-lh.view(-1,1)-lf.view(1,-1))
        pa = torch.exp(ii.float().view(1,-1)*torch.log(la.view(-1,1).clamp(min=1e-6))-la.view(-1,1)-lf.view(1,-1))
        P = ph.unsqueeze(2)*pa.unsqueeze(1); return P/P.sum([1,2],keepdim=True).clamp(min=1e-9)
    def exp_points(lh, la, thg, tag):
        P = grid_t(lh, la); op = torch.stack([torch.tril(P,-1).sum([1,2]), torch.diagonal(P,dim1=1,dim2=2).sum(1), torch.triu(P,1).sum([1,2])],1)
        EV = 2*P+op[:,O]; pi = torch.softmax(EV.reshape(EV.size(0),-1)/TAU,1).reshape_as(EV)
        ex = (I.unsqueeze(0)==thg.view(-1,1,1))&(J.unsqueeze(0)==tag.view(-1,1,1))
        Ot = torch.where(thg>tag,0,torch.where(thg==tag,1,2)); om=(O.unsqueeze(0)==Ot.view(-1,1,1))
        return (pi*(3.0*ex.float()+(om&~ex).float())).sum([1,2])
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
    _lg = np.array([math.lgamma(k+1) for k in range(GG+1)])
    def npgrid(lh, la):
        k = np.arange(GG+1); ph = np.exp(k*np.log(max(lh,1e-6))-lh-_lg); pa = np.exp(k*np.log(max(la,1e-6))-la-_lg)
        P = np.outer(ph, pa); return P/P.sum()
    wtr = np.where(natl[tr], W, 1.0).astype(np.float32)
    Rht,Rat,hgt,agt = T(Rh[tr]),T(Ra[tr]),T(hg[tr]),T(ag[tr]); wt=T(wtr)
    Hidt, Tht = T(HID[tr]), T(VF[tr,0].astype(np.float32))
    Rhe,Rae = T(Rh[te]),T(Ra[te]); Hide,The = T(HID[te]),T(VF[te,0].astype(np.float32))

    def run(mode):
        CTX = np.concatenate([CTX0, VF], 1) if mode == "flag" else CTX0
        nctx = CTX.shape[1]
        cmu = CTX[tr].mean(0); csd = CTX[tr].std(0)+1e-6; CTXn = ((CTX-cmu)/csd).astype(np.float32)
        Ct, Ce = T(CTXn[tr]), T(CTXn[te]); Xht,Xat=T(Xhn[tr]),T(Xan[tr]); Xhe,Xae=T(Xhn[te]),T(Xan[te])
        torch.manual_seed(0); np.random.seed(0)
        pt = mode == "perteam"
        net = PerTeam(A, nctx, n_teams) if pt else tg.GoalNet(A, nctx)
        if pt:  # hard decay on team_adv only; normal on the rest
            base_params = [p for n,p in net.named_parameters() if n != "team_adv.weight"]
            opt = torch.optim.AdamW([{"params": base_params, "weight_decay": 1e-4},
                                     {"params": [net.team_adv.weight], "weight_decay": 0.1}], lr=2e-3)
        else:
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep); bs, n = 512, Xht.size(0)
        for e in range(ep):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i+bs]; opt.zero_grad()
                if pt: lh, la = net(Xht[b], Rht[b], Xat[b], Rat[b], Ct[b], Hidt[b], Tht[b])
                else:  lh, la = net(Xht[b], Rht[b], Xat[b], Rat[b], Ct[b])
                loss = ((pois(lh,hgt[b])+pois(la,agt[b]))*wt[b]).mean() - BETA*(exp_points(torch.exp(lh),torch.exp(la),hgt[b],agt[b])*wt[b]).mean()
                loss.backward(); opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            el = net(Xhe, Rhe, Xae, Rae, Ce, Hide, The) if pt else net(Xhe, Rhe, Xae, Rae, Ce)
        elh, ela = np.exp(tonp(el[0])), np.exp(tonp(el[1])); grids = np.array([npgrid(a, b) for a, b in zip(elh, ela)])
        def sc(msk):
            P3 = np.array([tg.hda_from_P(g) for g in grids[msk]]); acc=float((P3.argmax(1)==y[te][msk]).mean()); r=tg.rps(y[te][msk],P3); tot=ex=0
            for gm,Hh,Aa in zip(grids[msk], hg[te][msk], ag[te][msk]):
                pk=ev_pick(gm); pts,lab=tg.grade(pk,int(Hh),int(Aa)); tot+=pts; ex+=lab=="exact"
            return acc,r,tot/msk.sum(),ex
        a1,a2 = sc(np.ones(te.sum(),bool)), sc(natl[te])
        print(f"  {mode:9s} ALL acc={a1[0]:.3f} rps={a1[1]:.4f} pg={a1[2]:.4f} ex={a1[3]} | NATL rps={a2[1]:.4f} pg={a2[2]:.4f} ex={a2[3]}", flush=True)
        if pt:
            tv = tonp(net.team_adv.weight).ravel()
            i2c = {v: int(k) for k, v in __import__("json").load(open(ROOT/"data"/"venue_map.json",encoding="utf-8"))["club2idx"].items()}
            cn = {r[0]: r[1] for r in con.execute("SELECT club_id,name FROM club")}
            # only teams with enough home games matter; show extremes among teams seen in training
            order = np.argsort(tv)
            def nm(idx): c=i2c.get(idx); return cn.get(c, str(c))
            print("  learned home-boost (top/bottom, exp(x)=goal mult):", flush=True)
            for idx in list(order[::-1][:6]) + list(order[:6]):
                print(f"      {nm(idx):24s} {tv[idx]:+.3f} ({math.exp(tv[idx]):.2f}x)", flush=True)

    print("=== home-advantage A/B (held-out test) ===", flush=True)
    run("base"); run("flag"); run("perteam")


if __name__ == "__main__":
    main()
