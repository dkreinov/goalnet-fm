"""Version B — autoregressive score-effects model. The encoder gives base full-match rates (lh,la); a
learned state-multiplier table modulates the per-15-min scoring rate by the current goal-difference
(leading/level/trailing). Trained on per-segment Poisson NLL (teacher-forced on actual states). At
inference, a DP rolls the score state forward over 6 segments to a FINAL-score distribution, which we
compare to the static double-Poisson (same base rates, state_mult=1) and score on exact/RPS/points.
Only matches with VALID segments are used. Usage: python src/train_autoreg.py [--epochs 60]
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
GG, W = 7, 15.0
NB = 5  # diff buckets: <=-2, -1, 0, +1, >=+2 (from a team's perspective)
def bucket(diff):
    return 0 if diff <= -2 else 1 if diff == -1 else 2 if diff == 0 else 3 if diff == 1 else 4


def ev_pick(P):   # grid-size-agnostic (tg.ev_pick hard-codes MAXG=9)
    G = P.shape[0]; ho = tg.hda_from_P(P); best, bs = -1, (1, 0)
    for i in range(G):
        for j in range(G):
            oc = 0 if i > j else (1 if i == j else 2); ev = 3 * P[i, j] + (ho[oc] - P[i, j])
            if ev > best: best, bs = ev, (i, j)
    return bs


class AutoReg(nn.Module):
    def __init__(self, A, nctx, d=64, h=128, p=0.3):
        super().__init__()
        self.gn = tg.GoalNet(A, nctx)
        self.smult = nn.Parameter(torch.zeros(NB))   # log-multiplier per diff bucket (0 -> x1)

    def base(self, Xh, Rh, Xa, Ra, C):
        return self.gn(Xh, Rh, Xa, Ra, C)            # loglh, logla (full match)


def main():
    ep = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 60
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    A = Xh.shape[2]
    cz = np.load(ROOT / "data" / "context.npz"); _cc, _cm = cz["ctx"], cz["mids"]
    cmap = {int(m): _cc[i] for i, m in enumerate(_cm)}; nctx = _cc.shape[1]
    CTX = np.stack([cmap.get(m, np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)
    sz = np.load(ROOT / "data" / "segments.npz")
    _sm, _ss, _sv = sz["mids"], sz["seg"], sz["valid"]      # materialize (NpzFile indexing is lazy)
    seg_of = {int(_sm[i]): _ss[i] for i in range(len(_sm)) if _sv[i]}
    con = db.connect()
    md = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([md.get(m, (0, 0, 0))[0] in NATc for m in mids])
    hg = np.array([min(md.get(m, [0]*4)[1] or 0, GG) for m in mids], np.int64)
    ag = np.array([min(md.get(m, [0]*4)[2] or 0, GG) for m in mids], np.int64)
    has = np.array([m in seg_of for m in mids])
    # per-match segment goals (6,2) capped, and the diff-bucket of each team at each segment start
    SEG = np.zeros((len(mids), 6, 2), np.int64); BKT = np.zeros((len(mids), 6, 2), np.int64)
    for i, m in enumerate(mids):
        s = seg_of.get(m)
        if s is None: continue
        ch = ca = 0
        for t in range(6):
            BKT[i, t, 0] = bucket(ch - ca); BKT[i, t, 1] = bucket(ca - ch)
            SEG[i, t, 0] = min(int(s[t][0]), 3); SEG[i, t, 1] = min(int(s[t][1]), 3)
            ch += int(s[t][0]); ca += int(s[t][1])
    tr = (dates < np.datetime64("2024-08-01")) & has
    te = (dates >= np.datetime64("2025-08-01")) & has
    print(f"valid-segment matches: train {tr.sum()} test {te.sum()} (natl {int((te&natl).sum())})", flush=True)
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)
    g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]))
    Xt, Rt, Xat, Rat, Ct = g(tr); SEGt = T(SEG[tr]); BKTt = T(BKT[tr]); wt = T(np.where(natl[tr], W, 1.0).astype(np.float32))
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")

    torch.manual_seed(7); np.random.seed(7); net = AutoReg(A, nctx)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    bs, n = 512, Xt.size(0)
    for e in range(ep):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; opt.zero_grad()
            llh, lla = net.base(Xt[b], Rt[b], Xat[b], Rat[b], Ct[b])      # (B,)
            sm = net.smult                                                # (NB,)
            # per-segment log-rate = log(lambda/6) + smult[bucket]
            base_h = llh.unsqueeze(1) - math.log(6); base_a = lla.unsqueeze(1) - math.log(6)  # (B,1)
            mh = sm[BKTt[b][:, :, 0]]; ma = sm[BKTt[b][:, :, 1]]          # (B,6)
            logh = base_h + mh; loga = base_a + ma                        # (B,6)
            loss_h = pois(logh, SEGt[b][:, :, 0].float()).mean(1)
            loss_a = pois(loga, SEGt[b][:, :, 1].float()).mean(1)
            loss = ((loss_h + loss_a) * wt[b]).mean()
            loss.backward(); opt.step()
        sched.step()
    net.eval()
    sm = np.exp(tonp(net.smult))
    print(f"  learned state multipliers by diff [<=-2,-1,0,+1,>=+2]: {np.round(sm,3)}", flush=True)
    with torch.no_grad():
        llh, lla = net.base(T(Xhn[te]), T(Rh[te]), T(Xan[te]), T(Ra[te]), T(CTXn[te]))
    lh, la = np.exp(tonp(llh)), np.exp(tonp(lla))

    _lg = np.array([math.lgamma(k+1) for k in range(5)])
    def segpmf(lam):   # Poisson 0..3 (cap), normalized
        k = np.arange(4); p = np.exp(k*np.log(max(lam,1e-9))-lam-_lg[:4]); return p/p.sum()
    def static_grid(a, b):
        return tg.score_matrix(a, b, 0.0)
    def autoreg_grid(a, b):
        P = np.zeros((GG+1, GG+1)); P[0, 0] = 1.0
        for t in range(6):
            nP = np.zeros_like(P)
            for h in range(GG+1):
                for aa in range(GG+1):
                    if P[h, aa] <= 1e-12: continue
                    ph = segpmf(a/6*sm[bucket(h-aa)]); pa = segpmf(b/6*sm[bucket(aa-h)])
                    for gh in range(4):
                        for ga in range(4):
                            nP[min(h+gh, GG), min(aa+ga, GG)] += P[h, aa]*ph[gh]*pa[ga]
            P = nP
        return P/P.sum()
    ye, hge, age_ = y[te], hg[te], ag[te]
    def report(name, gridfn, msk):
        sel = np.where(msk)[0]
        P3 = np.array([tg.hda_from_P(gridfn(lh[i], la[i])) for i in sel])
        acc = float((P3.argmax(1) == ye[sel]).mean()); r = tg.rps(ye[sel], P3); tot = ex = 0
        for i in sel:
            pk = ev_pick(gridfn(lh[i], la[i])); pts, lab = tg.grade(pk, int(hge[i]), int(age_[i])); tot += pts; ex += lab == "exact"
        print(f"  {name:22s} acc={acc:.3f} rps={r:.4f} pts/g={tot/len(sel):.4f} exact={ex}/{len(sel)}", flush=True)
    allm = np.ones(len(ye), bool); nm = natl[te]
    print("\n=== static double-Poisson vs autoregressive (same base rates) — held-out ===", flush=True)
    report("static (ALL)", static_grid, allm); report("autoreg (ALL)", autoreg_grid, allm)
    if nm.sum() > 5:
        report("static (NATL)", static_grid, nm); report("autoreg (NATL)", autoreg_grid, nm)


if __name__ == "__main__":
    main()
