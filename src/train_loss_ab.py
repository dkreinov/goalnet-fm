"""Three-way loss A/B for the scoreline model + tie behaviour + a 'tie-floor' pick rule.
Losses (all on the same GoalNet, held-out season):
  poisson   : Poisson NLL on goals (current production)
  weighted  : 3*logP(exact actual score) + 1*logP(actual outcome)  -- the simple 3/1-weighted likelihood
  decision  : Poisson - beta*expected_points (soft EV-pick surrogate)
For each: acc, rps, pts/g, exact%, and TIE behaviour (ties picked, ties caught). Then for the best model,
a tie-floor inference rule: pick the modal draw when its EV is within eps of the max EV (exploits that
exact-ties concentrate on 0-0/1-1/2-2 and pay 3). Usage: python src/train_loss_ab.py [--test 2024-25]
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
G, TAU, BETA = 7, 0.08, 3.0


def season_of(d):
    y = d.astype("datetime64[Y]").astype(int) + 1970
    m = d.astype("datetime64[M]").astype(int) % 12 + 1
    return f"{(y if m>=8 else y-1)}-{str((y if m>=8 else y-1)+1)[2:]}"


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
    md = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([md.get(m, (0, 0, 0))[0] in NATc for m in mids])
    hg = np.array([min(md.get(m, (0, 0, 0))[1] or 0, G) for m in mids], np.int64)
    ag = np.array([min(md.get(m, (0, 0, 0))[2] or 0, G) for m in mids], np.int64)
    season = np.array([season_of(d) for d in dates]); te = season == test_season; tr = ~te
    print(f"test={test_season} n={te.sum()}  train n={tr.sum()}", flush=True)

    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)
    ii = torch.arange(G + 1); I = ii.view(G + 1, 1).expand(G + 1, G + 1); J = ii.view(1, G + 1).expand(G + 1, G + 1)
    O = torch.where(I > J, 0, torch.where(I == J, 1, 2)); lf = torch.lgamma(ii.float() + 1)
    def grid(lh, la):
        ph = torch.exp(ii.float().view(1, -1) * torch.log(lh.view(-1, 1).clamp(min=1e-6)) - lh.view(-1, 1) - lf.view(1, -1))
        pa = torch.exp(ii.float().view(1, -1) * torch.log(la.view(-1, 1).clamp(min=1e-6)) - la.view(-1, 1) - lf.view(1, -1))
        P = ph.unsqueeze(2) * pa.unsqueeze(1); return P / P.sum(dim=[1, 2], keepdim=True).clamp(min=1e-9)
    def oprob_of(P):
        return torch.stack([torch.tril(P, -1).sum([1, 2]), torch.diagonal(P, dim1=1, dim2=2).sum(1), torch.triu(P, 1).sum([1, 2])], 1)
    def exp_points(lh, la, th, ta):
        P = grid(lh, la); EV = 2 * P + oprob_of(P)[:, O]
        pi = torch.softmax(EV.reshape(EV.size(0), -1) / TAU, 1).reshape_as(EV)
        ex = (I.unsqueeze(0) == th.view(-1, 1, 1)) & (J.unsqueeze(0) == ta.view(-1, 1, 1))
        Ot = torch.where(th > ta, 0, torch.where(th == ta, 1, 2)); om = (O.unsqueeze(0) == Ot.view(-1, 1, 1))
        return (pi * (3.0 * ex.float() + (om & ~ex).float())).sum([1, 2])
    def weighted_nll(lh, la, th, ta):
        P = grid(lh, la); B = P.size(0)
        pe = P[torch.arange(B), th, ta].clamp(min=1e-9)
        Ot = torch.where(th > ta, 0, torch.where(th == ta, 1, 2)); po = oprob_of(P)[torch.arange(B), Ot].clamp(min=1e-9)
        return -(3.0 * torch.log(pe) + 1.0 * torch.log(po))

    g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
    Tr = g(tr); Te = g(te); hgt, agt = Tr[5].long(), Tr[6].long(); wt = T(np.where(natl[tr], 5.0, 1.0).astype(np.float32))
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")

    def train(kind):
        torch.manual_seed(7); np.random.seed(7)
        net = tg.GoalNet(A, nctx); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep); bs, n = 512, Tr[0].size(0)
        for e in range(ep):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]; opt.zero_grad()
                lh, la = net(Tr[0][b], Tr[1][b], Tr[2][b], Tr[3][b], Tr[4][b])
                if kind == "weighted":
                    loss = (weighted_nll(torch.exp(lh), torch.exp(la), hgt[b], agt[b]) * wt[b]).mean()
                else:
                    loss = ((pois(lh, Tr[5][b]) + pois(la, Tr[6][b])) * wt[b]).mean()
                    if kind == "decision":
                        loss = loss - BETA * (exp_points(torch.exp(lh), torch.exp(la), hgt[b], agt[b]) * wt[b]).mean()
                loss.backward(); opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            lh, la = net(Te[0], Te[1], Te[2], Te[3], Te[4])
        return np.exp(tonp(lh)), np.exp(tonp(la))

    def evaluate(lh, la, tiefloor=0.0):
        ye, hge, age_ = y[te], hg[te], ag[te]
        P = np.array([tg.hda_from_P(tg.score_matrix(a, b)) for a, b in zip(lh, la)])
        acc = float((P.argmax(1) == ye).mean()); r = tg.rps(ye, P)
        tot = ex = tie_pick = tie_caught = 0
        for a, b, H, Aa in zip(lh, la, hge, age_):
            M = tg.score_matrix(a, b)
            # EV for every cell
            ho = tg.hda_from_P(M)
            best, bs = -1, (1, 0); bestdraw, ds = -1, None
            for i in range(tg.MAXG + 1):
                for j in range(tg.MAXG + 1):
                    o = 0 if i > j else (1 if i == j else 2); ev = 3 * M[i, j] + (ho[o] - M[i, j])
                    if ev > best: best, bs = ev, (i, j)
                    if i == j and ev > bestdraw: bestdraw, ds = ev, (i, j)
            pk = ds if (tiefloor and bestdraw >= best - tiefloor) else bs
            pts, lab = tg.grade(pk, int(H), int(Aa)); tot += pts; ex += lab == "exact"
            if pk[0] == pk[1]:
                tie_pick += 1; tie_caught += (H == Aa)
        n = te.sum()
        return f"acc={acc:.3f} rps={r:.4f} pts/g={tot/n:.4f} exact%={ex/n*100:.1f} tie_picks={tie_pick} tie_caught={tie_caught}"

    print("\n=== loss comparison (held-out season) ===", flush=True)
    res = {}
    for kind in ["poisson", "weighted", "decision"]:
        lh, la = train(kind); res[kind] = (lh, la)
        print(f"  {kind:10s} {evaluate(lh, la)}", flush=True)
    actual_draws = int((y[te] == 1).sum())
    print(f"\n  (actual draws in test: {actual_draws}/{te.sum()})", flush=True)
    print("\n=== FIFTH WAY: tie-floor on the decision model (eps = EV slack to grab a draw) ===", flush=True)
    lh, la = res["decision"]
    for eps in [0.0, 0.03, 0.06, 0.10, 0.15]:
        print(f"  eps={eps:<5} {evaluate(lh, la, tiefloor=eps)}", flush=True)


if __name__ == "__main__":
    main()
