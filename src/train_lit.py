"""Literature ablations for the scoreline model — test techniques used in football / sports / structured
prediction, each toggled against the current decision-focused GoalNet, on a held-out season.

Variants (vs baseline = Poisson + decision-focused loss):
  --aux-stats   multi-task: also predict shots/SOT/corners (regulariser; data on ~43% of matches)
  --aux-market  knowledge-distillation: also predict the bookmaker's implied H/D/A (train-time only)
  --negbin      Negative-Binomial goal head (overdispersion) instead of Poisson
  --cross       cross-team attention (home players attend to away) before pooling
Reports held-out acc / rps / pts-per-game (the decision metric). Usage: python src/train_lit.py [--test 2024-25]
"""
import sys, warnings, math
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db, train_goals as tg, train_pos2 as tp
NATc = {9, 10, 11, 12, 13, 14, 15}
GG, TAU, BETA = 7, 0.08, 3.0


def season_of(d):
    y = d.astype("datetime64[Y]").astype(int) + 1970; m = d.astype("datetime64[M]").astype(int) % 12 + 1
    return f"{(y if m>=8 else y-1)}-{str((y if m>=8 else y-1)+1)[2:]}"


class LitNet(nn.Module):
    def __init__(self, A, nctx, cfg, d=64, h=128, p=0.3):
        super().__init__()
        self.cfg = cfg
        self.enc = tp.Encoder(A, "xfmr", d, h, p) if not cfg.get("cross") else None
        if cfg.get("cross"):
            self.player = nn.Sequential(nn.Linear(A, 128), nn.ReLU(), nn.LayerNorm(128), nn.Dropout(p), nn.Linear(128, d))
            self.role = nn.Embedding(4, d)
            self.self_a = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=p, batch_first=True), 1)
            self.cross_a = nn.MultiheadAttention(d, 4, dropout=p, batch_first=True)
            self.team = nn.Sequential(nn.Linear(4 * d, h), nn.ReLU(), nn.LayerNorm(h), nn.Dropout(p), nn.Linear(h, h))
            self.register_buffer("d2r", torch.arange(4))
        self.ad = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 2))
        self.ctx = nn.Sequential(nn.Linear(nctx, 32), nn.ReLU(), nn.Linear(32, 2))
        self.home_adv = nn.Parameter(torch.tensor(0.25))
        if cfg.get("negbin"):
            self.logr = nn.Parameter(torch.zeros(2))      # NB dispersion (home, away)
        if cfg.get("aux_stats"):
            self.h_stats = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Linear(64, 6))
        if cfg.get("aux_market"):
            self.h_mkt = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Linear(64, 3))

    def _team(self, X, R):
        if self.enc is not None:
            return self.enc(X, R)
        pe = self.self_a(self.player(X) + self.role(R)); return pe   # return per-player for cross-attn

    def forward(self, Xh, Rh, Xa, Ra, C):
        if self.cfg.get("cross"):
            peh = self._team(Xh, Rh); pea = self._team(Xa, Ra)
            ph, _ = self.cross_a(peh, pea, pea); pa, _ = self.cross_a(pea, peh, peh)
            def pool(pe, R):
                pools = [(pe * (R == r).unsqueeze(-1).float()).sum(1) / (R == r).unsqueeze(-1).float().sum(1).clamp(min=1) for r in range(4)]
                return torch.cat(pools, -1)
            th = self.team(pool(peh + ph, Rh)); ta = self.team(pool(pea + pa, Ra))
        else:
            th = self.enc(Xh, Rh); ta = self.enc(Xa, Ra)
        adh, ada = self.ad(th), self.ad(ta); ch, ca = self.ctx(C).unbind(-1)
        logh = self.home_adv + adh[:, 0] - ada[:, 1] + ch
        loga = ada[:, 0] - adh[:, 1] + ca
        extra = (torch.cat([th, ta], -1),)
        return logh, loga, extra[0]


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
    rowm = {r[0]: r for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals,hs,as_,hst,ast,hc,ac,b365h,b365d,b365a FROM match")}
    natl = np.array([rowm.get(m, (0, 0))[1] in NATc for m in mids])
    hg = np.array([min(rowm.get(m, [0, 0, 0, 0])[2] or 0, GG) for m in mids], np.int64)
    ag = np.array([min(rowm.get(m, [0, 0, 0, 0])[3] or 0, GG) for m in mids], np.int64)
    # aux stats targets (hs,as,hst,ast,hc,ac) + mask
    ST = np.array([[(rowm.get(m, [None]*10)[i] if rowm.get(m) and rowm[m][i] is not None else np.nan) for i in range(4, 10)] for m in mids], np.float32)
    smask = (~np.isnan(ST).any(1)).astype(np.float32)
    # market implied H/D/A from b365 (normalize 1/odds) + mask
    def imp(m):
        r = rowm.get(m);
        if not r or r[10] is None or r[11] is None or r[12] is None or min(r[10], r[11], r[12]) <= 0:
            return [np.nan, np.nan, np.nan]
        inv = np.array([1 / r[10], 1 / r[11], 1 / r[12]]); return list(inv / inv.sum())
    MK = np.array([imp(m) for m in mids], np.float32); mmask = (~np.isnan(MK).any(1)).astype(np.float32)
    season = np.array([season_of(d) for d in dates]); te = season == test_season; tr = ~te
    print(f"test={test_season} n={te.sum()} train={tr.sum()} | stats {int(smask[tr].sum())} market {int(mmask[tr].sum())}", flush=True)

    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    sm = np.nanmean(ST[tr], 0); ss = np.nanstd(ST[tr], 0) + 1e-6
    STz = np.nan_to_num((ST - sm) / ss).astype(np.float32)
    MKf = np.nan_to_num(MK).astype(np.float32)
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)
    ii = torch.arange(GG + 1); I = ii.view(GG + 1, 1).expand(GG + 1, GG + 1); J = ii.view(1, GG + 1).expand(GG + 1, GG + 1)
    O = torch.where(I > J, 0, torch.where(I == J, 1, 2)); lf = torch.lgamma(ii.float() + 1)
    def grid(lh, la):
        ph = torch.exp(ii.float().view(1, -1) * torch.log(lh.view(-1, 1).clamp(min=1e-6)) - lh.view(-1, 1) - lf.view(1, -1))
        pa = torch.exp(ii.float().view(1, -1) * torch.log(la.view(-1, 1).clamp(min=1e-6)) - la.view(-1, 1) - lf.view(1, -1))
        P = ph.unsqueeze(2) * pa.unsqueeze(1); return P / P.sum([1, 2], keepdim=True).clamp(min=1e-9)
    def exp_points(lh, la, th, ta):
        P = grid(lh, la); op = torch.stack([torch.tril(P, -1).sum([1, 2]), torch.diagonal(P, dim1=1, dim2=2).sum(1), torch.triu(P, 1).sum([1, 2])], 1)
        EV = 2 * P + op[:, O]; pi = torch.softmax(EV.reshape(EV.size(0), -1) / TAU, 1).reshape_as(EV)
        ex = (I.unsqueeze(0) == th.view(-1, 1, 1)) & (J.unsqueeze(0) == ta.view(-1, 1, 1))
        Ot = torch.where(th > ta, 0, torch.where(th == ta, 1, 2)); om = (O.unsqueeze(0) == Ot.view(-1, 1, 1))
        return (pi * (3.0 * ex.float() + (om & ~ex).float())).sum([1, 2])

    g = lambda m: [T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]), T(STz[m]), T(smask[m]), T(MKf[m]), T(mmask[m])]
    Tr = g(tr); Te = g(te); wt = T(np.where(natl[tr], 5.0, 1.0).astype(np.float32))
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")

    def run(name, cfg):
        torch.manual_seed(7); np.random.seed(7)
        net = LitNet(A, nctx, cfg); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep); bs, n = 512, Tr[0].size(0)
        for e in range(ep):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]; opt.zero_grad()
                lh, la, rep = net(Tr[0][b], Tr[1][b], Tr[2][b], Tr[3][b], Tr[4][b])
                eh, ea = torch.exp(lh), torch.exp(la)
                if cfg.get("negbin"):
                    r = torch.exp(net.logr).clamp(1, 50)
                    def nbnll(mu, k, rr):
                        return -(torch.lgamma(k + rr) - torch.lgamma(rr) - torch.lgamma(k + 1)
                                 + rr * torch.log(rr / (rr + mu)) + k * torch.log(mu / (rr + mu) + 1e-9))
                    gl = (nbnll(eh, Tr[5][b].float(), r[0]) + nbnll(ea, Tr[6][b].float(), r[1]))
                else:
                    gl = pois(lh, Tr[5][b]) + pois(la, Tr[6][b])
                loss = (gl * wt[b]).mean() - BETA * (exp_points(eh, ea, Tr[5][b], Tr[6][b]) * wt[b]).mean()
                if cfg.get("aux_stats"):
                    pred = net.h_stats(rep); loss = loss + 0.1 * (((pred - Tr[7][b]) ** 2).mean(1) * Tr[8][b]).mean()
                if cfg.get("aux_market"):
                    pm = torch.log_softmax(net.h_mkt(rep), 1)
                    loss = loss + 0.3 * (-(Tr[9][b] * pm).sum(1) * Tr[10][b]).mean()
                loss.backward(); opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            lh, la, _ = net(Te[0], Te[1], Te[2], Te[3], Te[4])
        lh, la = np.exp(tonp(lh)), np.exp(tonp(la)); ye, hge, age_ = y[te], hg[te], ag[te]
        P = np.array([tg.hda_from_P(tg.score_matrix(a, b)) for a, b in zip(lh, la)])
        acc = float((P.argmax(1) == ye).mean()); r = tg.rps(ye, P); tot = ex = 0
        for a, b, H, Aa in zip(lh, la, hge, age_):
            pk = tg.ev_pick(tg.score_matrix(a, b)); pts, lab = tg.grade(pk, int(H), int(Aa)); tot += pts; ex += lab == "exact"
        print(f"  {name:16s} acc={acc:.3f} rps={r:.4f} pts/g={tot/te.sum():.4f} exact%={ex/te.sum()*100:.1f}", flush=True)

    print("\nliterature ablations (held-out season, all on decision-focused base):", flush=True)
    run("baseline", {})
    run("+aux_stats", {"aux_stats": True})
    run("+aux_market", {"aux_market": True})
    run("+negbin", {"negbin": True})
    run("+cross_attn", {"cross": True})
    run("+stats+market", {"aux_stats": True, "aux_market": True})


if __name__ == "__main__":
    main()
