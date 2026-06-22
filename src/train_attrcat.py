"""FM-attribute category ablation: does REMOVING a category of FM grades help or hurt?
Leave-one-category-out over the 62 attrs grouped into {technical, mental, physical, goalkeeping, hidden}.
For each config, neutralise that category (standardise -> set its columns to 0 = impute to dataset mean,
removing its signal), train the W=15 decision-focused GoalNet (same recipe as production), and A/B on the
held-out test (overall + national) by RPS and fantasy pts/g. 'full' = no removal (baseline).
Usage: python D:/Programming/claude/FM/src/train_attrcat.py [--epochs 60] [--seeds 1]
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

# 62 FM attributes grouped by category
CAT = {
    "technical": {"corners", "crossing", "dribbling", "finishing", "first-touch", "free-kick-taking",
                  "free-kicks", "heading", "long-shots", "long-throws", "marking", "passing",
                  "penalties", "penalty-taking", "tackling", "technique"},
    "mental": {"aggression", "anticipation", "bravery", "composure", "concentration", "decisions",
               "determination", "flair", "leadership", "off-the-ball", "positioning", "teamwork",
               "vision", "work-rate"},
    "physical": {"acceleration", "agility", "balance", "jumping-reach", "natural-fitness", "pace",
                 "stamina", "strength"},
    "goalkeeping": {"aerial-reach", "command-of-area", "communication", "eccentricity", "handling",
                    "kicking", "one-on-ones", "punching-tendency", "reflexes", "rushing-out-tendency",
                    "throwing"},
    "hidden": {"adaptability", "ambition", "consistency", "controversy", "dirtiness", "important-matches",
               "injury-proneness", "loyalty", "pressure", "professionalism", "sportsmanship",
               "temperament", "versatility"},
}


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
    ep = int(arg("--epochs", "60")); nseed = int(arg("--seeds", "1"))
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    ATTRS = [str(a) for a in z["attrs"]]; A = Xh.shape[2]; aidx = {n: i for i, n in enumerate(ATTRS)}
    # sanity: every attr lands in exactly one category
    allcat = set().union(*CAT.values())
    miss = set(ATTRS) - allcat; extra = allcat - set(ATTRS)
    if miss or extra: print(f"WARN uncategorised={sorted(miss)} unknown={sorted(extra)}", flush=True)
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
    print(f"train {tr.sum()} val {va.sum()} test {te.sum()} (natl-te {int((te&natl).sum())})", flush=True)

    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn0 = ((Xh - mu) / sd).astype(np.float32); Xan0 = ((Xa - mu) / sd).astype(np.float32)
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
    wt = T(np.where(natl[tr], W, 1.0).astype(np.float32))
    Rht, Rat, Ct = T(Rh[tr]), T(Ra[tr]), T(CTXn[tr]); hgt, agt = T(hg[tr]), T(ag[tr])
    Crh, Cra, Cc = T(Rh[te]), T(Ra[te]), T(CTXn[te])
    ye, hge, age_ = y[te], hg[te], ag[te]; nm = natl[te]

    def run(drop):
        cols = [aidx[a] for a in CAT[drop]] if drop else []
        Xhn = Xhn0.copy(); Xan = Xan0.copy()
        if cols:
            Xhn[:, :, cols] = 0.0; Xan[:, :, cols] = 0.0     # impute category to dataset mean
        Xht, Xat = T(Xhn[tr]), T(Xan[tr]); Xhe, Xae = T(Xhn[te]), T(Xan[te])
        grids = []
        for s in range(nseed):
            torch.manual_seed(s); np.random.seed(s); net = tg.GoalNet(A, nctx)
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
            bs, n = 512, Xht.size(0)
            for e in range(ep):
                net.train(); perm = torch.randperm(n)
                for i in range(0, n, bs):
                    b = perm[i:i+bs]; opt.zero_grad()
                    lh, la = net(Xht[b], Rht[b], Xat[b], Rat[b], Ct[b])
                    loss = ((pois(lh,hgt[b])+pois(la,agt[b]))*wt[b]).mean() - BETA*(exp_points(torch.exp(lh),torch.exp(la),hgt[b],agt[b])*wt[b]).mean()
                    loss.backward(); opt.step()
                sched.step()
            net.eval()
            with torch.no_grad():
                el = net(Xhe, Crh, Xae, Cra, Cc)
            elh, ela = np.exp(tonp(el[0])), np.exp(tonp(el[1]))
            grids.append(np.array([npgrid(a, b) for a, b in zip(elh, ela)]))
        G = np.mean(grids, 0)
        def sc(msk):
            P3 = np.array([tg.hda_from_P(g) for g in G[msk]]); acc = float((P3.argmax(1)==ye[msk]).mean()); r = tg.rps(ye[msk], P3); tot=ex=0
            for gm, H, Aa in zip(G[msk], hge[msk], age_[msk]):
                pk = ev_pick(gm); pts, lab = tg.grade(pk, int(H), int(Aa)); tot += pts; ex += lab=="exact"
            return acc, r, tot/msk.sum(), ex
        allm = np.ones(te.sum(), bool)
        a1, a2 = sc(allm), sc(nm)
        print(f"  {('full' if not drop else 'drop '+drop):16s} ALL acc={a1[0]:.3f} rps={a1[1]:.4f} pts/g={a1[2]:.4f} ex={a1[3]} | "
              f"NATL rps={a2[1]:.4f} pts/g={a2[2]:.4f} ex={a2[3]}", flush=True)

    print(f"\n=== FM-attribute category ablation (seeds={nseed}, ep={ep}) — held-out test ===", flush=True)
    run(None)
    for cat in ["technical", "mental", "physical", "goalkeeping", "hidden"]:
        run(cat)


if __name__ == "__main__":
    main()
