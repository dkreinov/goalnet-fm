"""Benchmark our model against the BOOKMAKER on the held-out eval, on both W/D/L and exact result.

We have Bet365 1X2 odds (b365h/d/a) on ~46% of matches -> the gold-standard W/D/L benchmark (de-vig the
odds to implied probabilities). For EXACT result the bookmaker only gives 1X2 (no correct-score market we
archived), so we build the standard MARKET-DC proxy: fit per-match Poisson rates (lambda_h, lambda_a) whose
double-Poisson outcome probs match the bookmaker's implied 1X2, then EV-pick under 3/1 — the same pipeline
as our model. Trains the decision-focused model on train-only (no leak), evaluates on the test season's
odds-bearing matches. Usage: python src/bookmaker_bench.py [--test 2024-25]
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
GG, TAU, BETA = 7, 0.08, 3.0


def season_of(d):
    y = d.astype("datetime64[Y]").astype(int) + 1970; m = d.astype("datetime64[M]").astype(int) % 12 + 1
    return f"{(y if m>=8 else y-1)}-{str((y if m>=8 else y-1)+1)[2:]}"


def rps(yv, p):
    cp = np.cumsum(p, 1); co = np.cumsum(np.eye(3)[yv], 1); return float(np.mean(np.sum((cp - co) ** 2, 1) / 2))


def main():
    def arg(k, d):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    test_season = arg("--test", "2024-25"); ep = int(arg("--epochs", "50"))
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    A = Xh.shape[2]
    cz = np.load(ROOT / "data" / "context.npz"); _cc, _cm = cz["ctx"], cz["mids"]
    cmap = {int(m): _cc[i] for i, m in enumerate(_cm)}; nctx = _cc.shape[1]
    CTX = np.stack([cmap.get(m, np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)
    con = db.connect()
    rm = {r[0]: r for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals,b365h,b365d,b365a FROM match")}
    natl = np.array([rm.get(m, (0, 0))[1] in NATc for m in mids])
    hg = np.array([min(rm.get(m, [0]*4)[2] or 0, GG) for m in mids], np.int64)
    ag = np.array([min(rm.get(m, [0]*4)[3] or 0, GG) for m in mids], np.int64)
    def implied(m):
        r = rm.get(m)
        if not r or r[4] is None or r[5] is None or r[6] is None or min(r[4], r[5], r[6]) <= 1.0:
            return None
        inv = np.array([1/r[4], 1/r[5], 1/r[6]]); return inv / inv.sum()   # de-vig
    MK = np.array([implied(m) if implied(m) is not None else [np.nan]*3 for m in mids], np.float32)
    has = ~np.isnan(MK).any(1)
    season = np.array([season_of(d) for d in dates]); te = (season == test_season); tr = ~te
    teo = te & has        # test matches WITH odds = the benchmark set
    print(f"test={test_season} n={te.sum()} | with bookmaker odds: {teo.sum()}", flush=True)

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

    # train decision-focused model on train-only
    g = lambda m: [T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m])]
    Tr = g(tr); wt = T(np.where(natl[tr],5.0,1.0).astype(np.float32)); pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
    torch.manual_seed(7); np.random.seed(7); net = tg.GoalNet(A, nctx)
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
        lh, la = net(T(Xhn[teo]), T(Rh[teo]), T(Xan[teo]), T(Ra[teo]), T(CTXn[teo]))
    mlh, mla = np.exp(tonp(lh)), np.exp(tonp(la))

    # MARKET-DC: fit (lh,la) on a grid so double-Poisson outcome probs match the bookmaker implied 1X2
    gridL = np.linspace(0.2, 3.6, 35)
    def out_probs(a, b2):
        k = np.arange(GG+1); _lg = np.array([math.lgamma(x+1) for x in k])
        pa = np.exp(k*np.log(a)-a-_lg); pb = np.exp(k*np.log(b2)-b2-_lg); P = np.outer(pa, pb)
        return np.array([np.tril(P,-1).sum(), np.trace(P), np.triu(P,1).sum()])
    LUT = {(a, b): out_probs(a, b) for a in gridL for b in gridL}
    keys = list(LUT); KP = np.array([LUT[k] for k in keys]); KL = np.array(keys)
    mk = MK[teo]
    book_l = np.array([KL[np.argmin(((KP - mk[i]) ** 2).sum(1))] for i in range(len(mk))])

    ye, hge, age_ = y[teo], hg[teo], ag[teo]
    def hda(lh_, la_):
        return np.array([tg.hda_from_P(tg.score_matrix(a, b)) for a, b in zip(lh_, la_)])
    def fpts(lh_, la_):
        tot = ex = 0
        for a, b, H, Aa in zip(lh_, la_, hge, age_):
            pk = tg.ev_pick(tg.score_matrix(a, b)); pts, lab = tg.grade(pk, int(H), int(Aa)); tot += pts; ex += lab == "exact"
        return tot/len(ye), ex/len(ye)*100
    Pm = hda(mlh, mla); Pb = mk    # bookmaker W/D/L = its own implied probs
    print(f"\n=== vs BOOKMAKER on {teo.sum()} held-out matches with odds ===", flush=True)
    print(f"  {'':10s}  outcome-acc   RPS      pts/g   exact%", flush=True)
    for nm, P, (lh2, la2) in [("OUR MODEL", Pm, (mlh, mla)), ("BOOKMAKER", Pb, (book_l[:,0], book_l[:,1]))]:
        acc = float((P.argmax(1) == ye).mean()); r = rps(ye, P); pg, ex = fpts(lh2, la2)
        print(f"  {nm:10s}   {acc:.3f}       {r:.4f}   {pg:.4f}  {ex:.1f}", flush=True)
    print("  (bookmaker W/D/L = de-vigged 1X2 implied probs; bookmaker exact = MARKET-DC fit of 1X2 -> EV-pick)", flush=True)


if __name__ == "__main__":
    main()
