"""Goal-output 11v11 model: predict the actual SCORELINE (home goals, away goals), not just who wins.

Same grade encoder as the result model (xfmr over the 11 + role + national context), but the head emits
two expected-goals rates via an attack/defence structure:
    logλ_home = home_adv + att_home - def_away + ctx_home
    logλ_away =            att_away - def_home + ctx_away
trained with Poisson NLL on the real goals (who-won is implied, used only as an auxiliary CE regulariser).
At inference we form the full scoreline distribution P(h,a) (double-Poisson + a tunable Dixon-Coles
low-score/draw correction ρ) and pick the EV-optimal scoreline under the league scoring (exact=3,
outcome=1). Reports H/D/A acc/RPS + fantasy points on val/test (all & national), and scores the played
WC2026 games. Usage: python D:/Programming/claude/FM/src/train_goals.py [--w 5] [--epochs 150]
"""
import json
import math
import sys
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db
import train_pos2 as tp

WC = Path(r"D:\Programming\claude\worldcup\team_db")
NATc = {9, 10, 11, 12, 13, 14, 15}
FMV, MAXG = 3, 9
ROLE_MEANKEY = (9, 10, 11, 12, 13, 14, 15)
NAME_FIX = {"IR Iran": "Iran", "Korea Republic": "South Korea", "Côte d'Ivoire": "Ivory Coast",
            "Cabo Verde": "Cape Verde"}
K, HADV, BASE = 20.0, 60.0, 1500.0
_logfac = np.array([math.lgamma(k + 1) for k in range(MAXG + 1)])


def pos_role(p):
    p = (p or "").upper()
    if p.startswith("G"):
        return 0
    if p[:2] in ("CD", "LB", "RB", "WB") or (p.startswith("D") and p != "DM"):
        return 1
    if p.startswith("F") or p.startswith("ST") or p in ("LW", "RW"):
        return 3
    return 2


class GoalNet(nn.Module):
    """Encoder -> (attack, defence) per team; Poisson rates with home advantage + context."""
    def __init__(self, A, nctx, d=64, h=128, p=0.3):
        super().__init__()
        self.enc = tp.Encoder(A, "xfmr", d, h, p)
        self.ad = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 2))  # att,def
        self.ctx = nn.Sequential(nn.Linear(nctx, 32), nn.ReLU(), nn.Linear(32, 2))           # ctx_h,ctx_a
        self.home_adv = nn.Parameter(torch.tensor(0.25))

    def forward(self, Xh, Rh, Xa, Ra, C):
        th = self.enc(Xh, Rh); ta = self.enc(Xa, Ra)
        adh, ada = self.ad(th), self.ad(ta)            # (B,2): [attack, defence]
        ch, ca = self.ctx(C).unbind(-1)
        logh = self.home_adv + adh[:, 0] - ada[:, 1] + ch
        loga = ada[:, 0] - adh[:, 1] + ca
        return logh, loga                              # log expected goals (home, away)


def score_matrix(lh, la, rho=0.0):
    """P(h,a) on 0..MAXG from double-Poisson + Dixon-Coles low-score correction rho (vectorised np)."""
    k = np.arange(MAXG + 1)
    ph = np.exp(k * np.log(max(lh, 1e-6)) - lh - _logfac)
    pa = np.exp(k * np.log(max(la, 1e-6)) - la - _logfac)
    P = np.outer(ph, pa)
    if rho:
        P[0, 0] *= 1 - lh * la * rho; P[1, 1] *= 1 - rho
        P[0, 1] *= 1 + lh * rho;      P[1, 0] *= 1 + la * rho
    return P / P.sum()


def hda_from_P(P):
    h = np.tril(P, -1).sum(); d = np.trace(P); a = np.triu(P, 1).sum()
    return np.array([h, d, a])


def ev_pick(P):
    """Scoreline maximising expected league points (exact=3, correct outcome=1)."""
    ho = hda_from_P(P)
    best, bs = -1, (1, 0)
    for i in range(MAXG + 1):
        for j in range(MAXG + 1):
            oc = 0 if i > j else (1 if i == j else 2)
            ev = 3 * P[i, j] + 1 * (ho[oc] - P[i, j])
            if ev > best:
                best, bs = ev, (i, j)
    return bs


def grade(pred, hs, as_):
    ph, pa = pred
    if ph == hs and pa == as_:
        return 3, "exact"
    ro = lambda x, y: (x > y) - (x < y)
    return (1, "correct") if ro(ph, pa) == ro(hs, as_) else (0, "wrong")


def rps(y, p):
    cp = np.cumsum(p, 1); co = np.cumsum(np.eye(3)[y], 1)
    return float(np.mean(np.sum((cp - co) ** 2, 1) / 2))


def national_context(con):
    rows = con.execute(
        f"""SELECT home_club_id,away_club_id,home_goals,away_goals FROM match
            WHERE competition_id IN {tuple(NATc)} AND home_goals IS NOT NULL
            ORDER BY match_date,match_id""").fetchall()
    elo = defaultdict(lambda: BASE); form = defaultdict(lambda: deque(maxlen=5)); gd = defaultdict(lambda: deque(maxlen=5))
    for hc, ac, hg, ag in rows:
        eh, ea = elo[hc], elo[ac]
        exp = 1 / (1 + 10 ** (-((eh + HADV) - ea) / 400.0))
        s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo[hc] = eh + K * (s - exp); elo[ac] = ea - K * (s - exp)
        ph = 3 if hg > ag else (1 if hg == ag else 0)
        form[hc].append(ph); form[ac].append(3 - ph if ph != 1 else 1)
        gd[hc].append(hg - ag); gd[ac].append(ag - hg)
    return {cid: (elo[cid], np.mean(form[cid]) if form[cid] else 1.0,
                  np.mean(gd[cid]) if gd[cid] else 0.0) for cid in elo}


def ctx_vec(h, a):
    eh, fh, gh = h; ea, fa, ga = a
    return np.array([eh / 400, ea / 400, (eh - ea) / 400, fh, fa, fh - fa, gh, ga, 0.5, 0.5], np.float32)


def main():
    def arg(k, d):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    W = float(arg("--w", "5")); ep = int(arg("--epochs", "150"))

    z = np.load(ROOT / "data" / "players.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    ATTRS = [str(a) for a in z["attrs"]]; A = len(ATTRS); aidx = {n: i for i, n in enumerate(ATTRS)}
    cz = np.load(ROOT / "data" / "context.npz"); cctx, cmids = cz["ctx"], cz["mids"]
    cmap = {int(m): cctx[i] for i, m in enumerate(cmids)}; nctx = cctx.shape[1]
    CTX = np.stack([cmap.get(m, np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)

    con = db.connect()
    meta = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([meta.get(m, (0, 0, 0))[0] in NATc for m in mids])
    hg = np.array([min(meta.get(m, (0, 0, 0))[1] or 0, MAXG) for m in mids], np.float32)
    ag = np.array([min(meta.get(m, (0, 0, 0))[2] or 0, MAXG) for m in mids], np.float32)

    tr = dates < np.datetime64("2024-08-01")
    va = (dates >= np.datetime64("2024-08-01")) & (dates < np.datetime64("2025-08-01"))
    te = dates >= np.datetime64("2025-08-01")
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    role_mean = {r: Xh[tr][Rh[tr] == r].mean(0) for r in range(4)}

    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t):
        return np.array(t.detach().tolist(), dtype=np.float32)
    g = lambda m: (T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(hg[m]), T(ag[m]))
    Xhtr, Rhtr, Xatr, Ratr, Ctr, hgtr, agtr = g(tr)
    wt = T(np.where(natl[tr], W, 1.0).astype(np.float32))
    Vh, Vrh, Va_, Vra, Cv, _, _ = g(va)

    print(f"training GoalNet (Poisson, national W={W})...", flush=True)
    torch.manual_seed(7); np.random.seed(7)
    net = GoalNet(A, nctx)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
    bs, n = 512, Xhtr.size(0); best, bstate, bad = 9, None, 0

    def val_rps():
        net.eval()
        with torch.no_grad():
            lh, la = net(Vh, Vrh, Va_, Vra, Cv)
        lh, la = tonp(lh), tonp(la)
        P = np.array([hda_from_P(score_matrix(math.exp(a), math.exp(b))) for a, b in zip(lh, la)])
        return rps(y[va], P)

    for e in range(ep):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lh, la = net(Xhtr[b], Rhtr[b], Xatr[b], Ratr[b], Ctr[b])
            loss = ((pois(lh, hgtr[b]) + pois(la, agtr[b])) * wt[b]).mean()
            loss.backward(); opt.step()
        sched.step()
        r = val_rps()
        if r < best - 1e-4:
            best, bstate, bad = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 25:
            break
    net.load_state_dict(bstate); net.eval()

    def rates(msk):
        with torch.no_grad():
            lh, la = net(T(Xhn[msk]), T(Rh[msk]), T(Xan[msk]), T(Ra[msk]), T(CTXn[msk]))
        return np.exp(tonp(lh)), np.exp(tonp(la))

    # tune Dixon-Coles rho on val by fantasy points
    lhv, lav = rates(va)
    def points(lh, la, hga, aga, rho):
        tot = ex = co = 0
        for a, b, H, Aa in zip(lh, la, hga, aga):
            pk = ev_pick(score_matrix(a, b, rho)); pts, lab = grade(pk, int(H), int(Aa))
            tot += pts; ex += lab == "exact"; co += lab == "correct"
        return tot, ex, co
    best_rho = max([-0.15, -0.1, -0.05, 0.0, 0.05], key=lambda r: points(lhv, lav, hg[va], ag[va], r)[0])
    print(f"  val rps={best:.4f}  best DC rho={best_rho}", flush=True)

    def report(tag, msk):
        lh, la = rates(msk)
        P = np.array([hda_from_P(score_matrix(a, b, best_rho)) for a, b in zip(lh, la)])
        acc = float((P.argmax(1) == y[msk]).mean()); r = rps(y[msk], P)
        tot, ex, co = points(lh, la, hg[msk], ag[msk], best_rho)
        nn_ = msk.sum()
        print(f"    {tag:14s} n={nn_:4d} acc={acc:.3f} rps={r:.4f} | pts={tot} (exact={ex} correct={co}) "
              f"= {tot/nn_:.3f}/g", flush=True)

    print("  VAL:", flush=True); report("all", va); report("national", va & natl)
    print("  TEST:", flush=True); report("all", te); report("national", te & natl)

    # ---- retrain on ALL data (train+val+test) for the actual WC2026 prediction (no leakage: every
    #      match is in the past relative to June-2026 games). Fixed epoch budget ~ where val converged. ----
    if "--full" in sys.argv:
        full_ep = max(30, e // 2)
        print(f"\nretraining on ALL {len(mids):,} matches for WC prediction ({full_ep} epochs)...", flush=True)
        allm = np.ones(len(mids), bool)
        Xhf, Rhf, Xaf, Raf, Cf, hgf, agf = g(allm)
        wf = T(np.where(natl, W, 1.0).astype(np.float32))
        torch.manual_seed(7); np.random.seed(7)
        net = GoalNet(A, nctx)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, full_ep)
        nf = Xhf.size(0)
        for e2 in range(full_ep):
            net.train(); perm = torch.randperm(nf)
            for i in range(0, nf, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                lh, la = net(Xhf[b], Rhf[b], Xaf[b], Raf[b], Cf[b])
                loss = ((pois(lh, hgf[b]) + pois(la, agf[b])) * wf[b]).mean()
                loss.backward(); opt.step()
            sched.step()
        net.eval()
        print("  (full-data model trained)", flush=True)

    # ---- score the played WC2026 games ----
    natctx = national_context(con)
    name2cid = {r[1]: r[0] for r in con.execute("SELECT club_id,name FROM club")}
    teams = {}
    for f in (WC / "teams").glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))["team"]
        teams[t["code"]] = name2cid.get(NAME_FIX.get(t["name"], t["name"]))
    snap = {}
    for sid, ca, nm in con.execute("SELECT s.snapshot_id,s.ca,p.norm_name FROM player_snapshot s "
                                   "JOIN player p ON p.player_id=s.player_id WHERE s.fm_version_id=?", (FMV,)):
        if nm not in snap or (ca or 0) > snap[nm][1]:
            snap[nm] = (sid, ca or 0)
    ab = defaultdict(dict)
    for sid, name, val in con.execute("SELECT a.snapshot_id,a.attr_name,a.attr_value FROM player_attribute a "
                                      "JOIN player_snapshot s ON s.snapshot_id=a.snapshot_id WHERE s.fm_version_id=?", (FMV,)):
        ab[sid][name] = val

    def vec_for(full):
        s = snap.get(db.norm(full))
        if not s:
            return None
        v = np.zeros(A, np.float32)
        for nm, vl in ab.get(s[0], {}).items():
            j = aidx.get(nm)
            if j is not None:
                v[j] = vl
        return v

    def side(xi):
        ps = [(pos_role(p.get("pos")), vec_for(p.get("full", ""))) for p in xi]
        imp = sum(v is None for _, v in ps)
        ps = [(r, v if v is not None else role_mean[r]) for r, v in ps]
        ps.sort(key=lambda t: t[0]); ps = ps[:11] + [(2, role_mean[2])] * max(0, 11 - len(ps))
        return np.stack([v for _, v in ps[:11]]), np.array([r for r, _ in ps[:11]], np.int64), imp

    L = json.load(open(WC / "lineups.json", encoding="utf-8"))
    Rz = json.load(open(WC / "results.json", encoding="utf-8"))
    games = [(k, v) for k, v in Rz.items() if v.get("status") == "finished" and k in L]
    tot = ex = co = wr = imps = 0
    print(f"\n  game           res   pred(EV)  pts", flush=True)
    for k, res in games:
        gg = L[k]
        Xh1, Rh1, i1 = side(gg.get("home_xi", [])); Xa1, Ra1, i2 = side(gg.get("away_xi", []))
        imps += i1 + i2
        ctx = ctx_vec(natctx.get(teams.get(res["home"]), (BASE, 1, 0)), natctx.get(teams.get(res["away"]), (BASE, 1, 0)))
        Xh1 = ((Xh1 - mu) / sd).astype(np.float32); Xa1 = ((Xa1 - mu) / sd).astype(np.float32)
        ctxn = ((ctx - cmu) / csd).astype(np.float32)
        with torch.no_grad():
            lh, la = net(T(Xh1[None]), T(Rh1[None]), T(Xa1[None]), T(Ra1[None]), T(ctxn[None]))
        pk = ev_pick(score_matrix(math.exp(tonp(lh)[0]), math.exp(tonp(la)[0]), best_rho))
        pts, lab = grade(pk, res["hs"], res["as"]); tot += pts; ex += lab == "exact"; co += lab == "correct"; wr += lab == "wrong"
        print(f"  {res['home']}-{res['away']:<3} {res['hs']}-{res['as']}   {pk[0]}-{pk[1]}      {pts}", flush=True)
    print(f"\n=== GoalNet on {len(games)} played WC2026 games ===", flush=True)
    print(f"  exact={ex} correct={co} wrong={wr}  ->  TOTAL = {tot}", flush=True)
    print(f"  (you/YOU=19 [0 exact,19 correct]; top=31 [6 exact,13 correct]; {imps}/{len(games)*22} imputed)", flush=True)

    # ---- detailed prediction for one specific game (played or live) ----
    if "--game" in sys.argv:
        key = sys.argv[sys.argv.index("--game") + 1]
        gg = L.get(key)
        if not gg:
            print(f"\n--game {key}: not found in lineups.json", flush=True)
            return
        res = Rz.get(key, {})
        Xh1, Rh1, i1 = side(gg.get("home_xi", [])); Xa1, Ra1, i2 = side(gg.get("away_xi", []))
        hc, ac = key.split("-")
        ctx = ctx_vec(natctx.get(teams.get(hc), (BASE, 1, 0)), natctx.get(teams.get(ac), (BASE, 1, 0)))
        Xh1 = ((Xh1 - mu) / sd).astype(np.float32); Xa1 = ((Xa1 - mu) / sd).astype(np.float32)
        ctxn = ((ctx - cmu) / csd).astype(np.float32)
        with torch.no_grad():
            lh, la = net(T(Xh1[None]), T(Rh1[None]), T(Xa1[None]), T(Ra1[None]), T(ctxn[None]))
        lhh, laa = math.exp(tonp(lh)[0]), math.exp(tonp(la)[0])
        P = score_matrix(lhh, laa, best_rho)
        ho = hda_from_P(P); pk = ev_pick(P)
        flat = sorted(((P[i, j], i, j) for i in range(MAXG + 1) for j in range(MAXG + 1)), reverse=True)
        print(f"\n=== PREDICTION: {hc} (home) vs {ac} (away)  [status={res.get('status','?')}] ===", flush=True)
        print(f"  expected goals: {hc} {lhh:.2f} - {laa:.2f} {ac}  (imputed starters: {i1+i2}/22)", flush=True)
        print(f"  outcome probs:  {hc} win {ho[0]*100:.0f}%  draw {ho[1]*100:.0f}%  {ac} win {ho[2]*100:.0f}%", flush=True)
        print(f"  EV-optimal pick: {hc} {pk[0]}-{pk[1]} {ac}", flush=True)
        print(f"  most likely scorelines:", flush=True)
        for p, i, j in flat[:6]:
            print(f"     {hc} {i}-{j} {ac}   {p*100:.1f}%", flush=True)
        if res.get("status") == "finished":
            print(f"  ACTUAL: {hc} {res['hs']}-{res['as']} {ac}", flush=True)


if __name__ == "__main__":
    main()
