"""A/B: does baking CLUB market value into the NATIONAL-fine-tuned GoalNet help? Runs the production national
recipe (train-split all-data W=15 decision-focused -> fine-tune on national matches) TWICE: base context vs
base+value context, and compares on the held-out test (ALL + NATIONAL). This tests whether value adds ON TOP
of the national fine-tune (unlike playerval_ablation which had no fine-tune). Single-seed for speed.
Usage: python experiments/valnatl_ab.py [--epochs 60] [--ftepochs 40]
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


def ev_pick(P):
    G = P.shape[0]; ho = tg.hda_from_P(P); best, bs = -1, (1, 0)
    for i in range(G):
        for j in range(G):
            oc = 0 if i > j else (1 if i == j else 2); ev = 3 * P[i, j] + (ho[oc] - P[i, j])
            if ev > best: best, bs = ev, (i, j)
    return bs


def build_value(mids):
    """Per-match [mean home-XI club value, mean away-XI club value, diff, home cov, away cov] — the proven
    club-value proxy (club match: match_player.club_id; national: player's real club via player_snapshot)."""
    con = db.connect()
    seasonlab = {r[0]: r[1] for r in con.execute("SELECT season_id,label FROM season")}
    minfo = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,season_id,home_club_id,away_club_id FROM match")}
    cval, cval_any = {}, {}
    for cid, season, sv in con.execute("SELECT club_id,season,squad_value_eur FROM club_season_tm ORDER BY season"):
        if sv: cval[(cid, season)] = sv; cval_any[cid] = sv
    pclub = {}
    for pid, cid in con.execute("SELECT player_id,club_id FROM player_snapshot WHERE club_id IS NOT NULL ORDER BY snapshot_date"):
        pclub[pid] = cid
    starters = defaultdict(list)
    for mid, pid, cid in con.execute("SELECT match_id,player_id,club_id FROM match_player WHERE started=1"):
        starters[mid].append((pid, cid))
    con.close()
    def pval(pid, teamcid, lab):
        v = cval.get((teamcid, lab)) or cval_any.get(teamcid)
        if v: return v
        rc = pclub.get(pid)
        return (cval.get((rc, lab)) or cval_any.get(rc)) if rc is not None else None
    def lv(x): return math.log1p(x) if x else 0.0
    F = []
    for m in mids:
        si, hc, ac = minfo.get(m, (None, None, None)); lab = seasonlab.get(si)
        hv, av = [], []
        for pid, cid in starters.get(m, []):
            (hv if cid == hc else av).append(pval(pid, cid, lab))
        hvv = [x for x in hv if x]; avv = [x for x in av if x]
        F.append([lv(np.mean(hvv)) if hvv else 0.0, lv(np.mean(avv)) if avv else 0.0,
                  (lv(np.mean(hvv)) - lv(np.mean(avv))) if (hvv and avv) else 0.0,
                  len(hvv)/max(len(hv),1), len(avv)/max(len(av),1)])
    return np.array(F, np.float32)


def main():
    def arg(k, d): return int(sys.argv[sys.argv.index(k)+1]) if k in sys.argv else d
    ep, fep = arg("--epochs", 60), arg("--ftepochs", 40)
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    A = Xh.shape[2]
    cz = np.load(ROOT / "data" / "context.npz"); _cc, _cm = cz["ctx"], cz["mids"]
    cmap = {int(m): _cc[i] for i, m in enumerate(_cm)}; nctx0 = _cc.shape[1]
    CTX0 = np.stack([cmap.get(m, np.zeros(nctx0, np.float32)) for m in mids]).astype(np.float32)
    print("building club-value feature...", flush=True); VALUE = build_value(mids)
    con = db.connect()
    md = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([md.get(m, (0, 0, 0))[0] in NATc for m in mids])
    hg = np.array([min(md.get(m, [0]*4)[1] or 0, GG) for m in mids], np.int64)
    ag = np.array([min(md.get(m, [0]*4)[2] or 0, GG) for m in mids], np.int64)
    tr = dates < np.datetime64("2024-08-01"); te = dates >= np.datetime64("2025-08-01"); trn = tr & natl
    print(f"train {tr.sum()} (natl {int(trn.sum())}) test {te.sum()} (natl {int((te&natl).sum())})", flush=True)
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu)/sd).astype(np.float32); Xan = ((Xa - mu)/sd).astype(np.float32)
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a): a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)
    ii = torch.arange(GG+1); I = ii.view(GG+1,1).expand(GG+1,GG+1); J = ii.view(1,GG+1).expand(GG+1,GG+1)
    O = torch.where(I>J,0,torch.where(I==J,1,2)); lf = torch.lgamma(ii.float()+1)
    def grid_t(lh, la):
        ph = torch.exp(ii.float().view(1,-1)*torch.log(lh.view(-1,1).clamp(min=1e-6))-lh.view(-1,1)-lf.view(1,-1))
        pa = torch.exp(ii.float().view(1,-1)*torch.log(la.view(-1,1).clamp(min=1e-6))-la.view(-1,1)-lf.view(1,-1))
        P = ph.unsqueeze(2)*pa.unsqueeze(1); return P/P.sum([1,2],keepdim=True).clamp(min=1e-9)
    def exp_points(lh, la, th, ta):
        P = grid_t(lh, la); op = torch.stack([torch.tril(P,-1).sum([1,2]), torch.diagonal(P,dim1=1,dim2=2).sum(1), torch.triu(P,1).sum([1,2])],1)
        EV = 2*P+op[:,O]; pi = torch.softmax(EV.reshape(EV.size(0),-1)/TAU,1).reshape_as(EV)
        ex = (I.unsqueeze(0)==th.view(-1,1,1))&(J.unsqueeze(0)==ta.view(-1,1,1))
        Ot = torch.where(th>ta,0,torch.where(th==ta,1,2)); om = (O.unsqueeze(0)==Ot.view(-1,1,1))
        return (pi*(3.0*ex.float()+(om&~ex).float())).sum([1,2])
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
    _lg = np.array([math.lgamma(k+1) for k in range(GG+1)])
    def npgrid(lh, la):
        k = np.arange(GG+1); ph = np.exp(k*np.log(max(lh,1e-6))-lh-_lg); pa = np.exp(k*np.log(max(la,1e-6))-la-_lg)
        P = np.outer(ph, pa); return P/P.sum()

    def run(CTX, tag):
        nctx = CTX.shape[1]
        cmu = CTX[tr].mean(0); csd = CTX[tr].std(0)+1e-6; CTXn = ((CTX-cmu)/csd).astype(np.float32)
        g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
        Eh, Erh, Ea, Era, Ce, _, _ = g(te)
        def train(mask, weights, epochs, init, lr):
            Tr = g(mask); wt = T(weights.astype(np.float32))
            torch.manual_seed(0); np.random.seed(0); net = tg.GoalNet(A, nctx)
            if init is not None: net.load_state_dict(init)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            bs, n = 512, Tr[0].size(0)
            for e in range(epochs):
                net.train(); perm = torch.randperm(n)
                for i in range(0, n, bs):
                    b = perm[i:i+bs]; opt.zero_grad()
                    lh, la = net(Tr[0][b], Tr[1][b], Tr[2][b], Tr[3][b], Tr[4][b])
                    loss = ((pois(lh,Tr[5][b])+pois(la,Tr[6][b]))*wt[b]).mean() - BETA*(exp_points(torch.exp(lh),torch.exp(la),Tr[5][b],Tr[6][b])*wt[b]).mean()
                    loss.backward(); opt.step()
                sch.step()
            net.eval(); return net
        base = train(tr, np.where(natl[tr], W, 1.0), ep, None, 2e-3)
        ft = train(trn, np.ones(int(trn.sum())), fep, {k: v.clone() for k, v in base.state_dict().items()}, 5e-4)
        with torch.no_grad():
            el = ft(Eh, Erh, Ea, Era, Ce)
        elh, ela = np.exp(tonp(el[0])), np.exp(tonp(el[1])); grids = np.array([npgrid(a, b) for a, b in zip(elh, ela)])
        def sc(msk):
            P3 = np.array([tg.hda_from_P(gr) for gr in grids[msk]]); r=tg.rps(y[te][msk],P3); tot=ex=0
            for gm,Hh,Aa in zip(grids[msk], hg[te][msk], ag[te][msk]):
                pk=ev_pick(gm); pts,lab=tg.grade(pk,int(Hh),int(Aa)); tot+=pts; ex+=lab=="exact"
            return r,tot/msk.sum(),ex
        a1, a2 = sc(np.ones(te.sum(),bool)), sc(natl[te])
        print(f"  {tag:22s} ALL rps={a1[0]:.4f} pg={a1[1]:.4f} ex={a1[2]} | NATL rps={a2[0]:.4f} pg={a2[1]:.4f} ex={a2[2]}", flush=True)

    print("=== bake club value into national-finetuned GoalNet (held-out test) ===", flush=True)
    run(CTX0, "natl-ft base")
    run(np.concatenate([CTX0, VALUE], 1), "natl-ft +value")


if __name__ == "__main__":
    main()
