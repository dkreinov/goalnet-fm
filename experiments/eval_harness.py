"""E0 — eval harness foundation for the Phase-1 (no-retrain) pick/betting-layer experiments.
Produces a cache of per-seed Poisson rates (lh, la) so any downstream experiment can rebuild score grids at
any rho / sharpening / empirical-blend instantly and average across the ensemble, then score by RPS / pts/g /
exact-count. Two eval surfaces:
  - HELD-OUT TEST (statistical power): 5 seeds trained on the TRAIN split only (no leakage), rates dumped for
    val (for tuning) and test (for reporting).
  - PLAYED WC2026 GAMES (on-target, n~40): rates from the full-data production goalnet.pt (correct — every
    match precedes the WC), built from the real lineups exactly like predict_game.py.
Writes experiments/grids_cache.npz (test/val rates+labels) and experiments/wc_cache.npz (WC rates+labels).
Usage: python D:/Programming/claude/FM/experiments/eval_harness.py [--seeds 5] [--epochs 60]
Reusable API (import): load_cache() -> dict; ens_grid(lh_seeds, la_seeds, rho, gamma=1, emp=None, alpha=0).
"""
import sys, json, math, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db, train_goals as tg
WC = Path(r"D:\Programming\claude\worldcup\team_db")
NATc = {9, 10, 11, 12, 13, 14, 15}
GG, TAU, BETA, W = 7, 0.08, 3.0, 15.0
CACHE = ROOT / "experiments" / "grids_cache.npz"
WCACHE = ROOT / "experiments" / "wc_cache.npz"


# ---------- shared scoring utilities (grid is MAXG+1 square, consistent with production) ----------
def empirical_grid():
    """Historical final-score distribution over all scored matches, capped to the MAXG grid."""
    con = db.connect(); M = tg.MAXG + 1; E = np.zeros((M, M))
    for h, a in con.execute("SELECT home_goals,away_goals FROM match WHERE home_goals IS NOT NULL"):
        E[min(h, tg.MAXG), min(a, tg.MAXG)] += 1
    con.close(); return E / E.sum()


def make_grid(lh, la, rho=0.0, gamma=1.0, emp=None, alpha=0.0):
    """One score grid from rates, with optional sharpening (P**gamma) and empirical blend."""
    P = tg.score_matrix(lh, la, rho)
    if gamma != 1.0:
        P = P ** gamma; P = P / P.sum()
    if emp is not None and alpha > 0:
        P = (1 - alpha) * P + alpha * emp; P = P / P.sum()
    return P


def ens_grid(lh_seeds, la_seeds, rho=0.0, gamma=1.0, emp=None, alpha=0.0):
    """Ensemble grid for one match: average per-seed grids (production convention)."""
    Ps = [make_grid(lh, la, rho, gamma, emp, alpha) for lh, la in zip(lh_seeds, la_seeds)]
    P = np.mean(Ps, 0); return P / P.sum()


def score_grids(grids, y, hg, ag):
    """RPS / pts/g / exact for a list of score grids vs truth. pts under EV-pick (exact=3, outcome=1)."""
    P3 = np.array([tg.hda_from_P(g) for g in grids]); acc = float((P3.argmax(1) == y).mean())
    r = tg.rps(y, P3); tot = ex = 0
    for g, H, Aa in zip(grids, hg, ag):
        pk = tg.ev_pick(g); pts, lab = tg.grade(pk, int(H), int(Aa)); tot += pts; ex += lab == "exact"
    return dict(acc=acc, rps=r, pts=tot, pg=tot / len(grids), exact=ex, n=len(grids))


def load_cache():
    c = dict(np.load(CACHE)); w = dict(np.load(WCACHE, allow_pickle=True)); c["wc"] = w; return c


# ---------- builders ----------
def _T(a):
    a = np.ascontiguousarray(a); tt = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    return torch.frombuffer(bytearray(a.tobytes()), dtype=tt[a.dtype]).reshape(a.shape)


def _tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)


def build_test_cache(nseed, ep):
    """Train nseed seeds on the TRAIN split (decision-focused W=15) and dump per-seed rates for val+test."""
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
    print(f"train {tr.sum()} val {va.sum()} test {te.sum()} (natl-te {int((te&natl).sum())})", flush=True)
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    g = lambda m: (_T(Xhn[m]), _T(Rh[m]), _T(Xan[m]), _T(Ra[m]), _T(CTXn[m]), _T(hg[m]), _T(ag[m]))
    Tr = g(tr); wt = _T(np.where(natl[tr], W, 1.0).astype(np.float32))
    Vh, Vrh, Va_, Vra, Cv, _, _ = g(va); Eh, Erh, Ea, Era, Ce, _, _ = g(te)
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
    vlh, vla, elh, ela = [], [], [], []
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
        vlh.append(np.exp(_tonp(vl[0]))); vla.append(np.exp(_tonp(vl[1])))
        elh.append(np.exp(_tonp(el[0]))); ela.append(np.exp(_tonp(el[1])))
        print(f"  seed {s} done", flush=True)
    np.savez_compressed(CACHE,
        val_lh=np.array(vlh), val_la=np.array(vla), va_y=y[va], va_hg=hg[va], va_ag=ag[va], va_natl=natl[va],
        te_lh=np.array(elh), te_la=np.array(ela), te_y=y[te], te_hg=hg[te], te_ag=ag[te], te_natl=natl[te])
    print(f"  saved {CACHE}", flush=True)


def build_wc_cache():
    """Per-seed rates for the played WC2026 games from the full-data production goalnet.pt (predict_game logic)."""
    ck = ROOT / "data" / "goalnet.pt"
    c = torch.load(ck, weights_only=False)
    A, nctx = c["A"], c["nctx"]; mu, sd, cmu, csd = c["mu"], c["sd"], c["cmu"], c["csd"]
    ATTRS = c["attrs"]; aidx = {n: i for i, n in enumerate(ATTRS)}; role_mean = c["role_mean"]
    states = c.get("states") or [c["state"]]
    nets = []
    for st in states:
        nt = tg.GoalNet(A, nctx); nt.load_state_dict(st); nt.eval(); nets.append(nt)
    con = db.connect(); natctx = tg.national_context(con)
    name2cid = {r[1]: r[0] for r in con.execute("SELECT club_id,name FROM club")}
    teams = {}
    for f in (WC / "teams").glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))["team"]; teams[t["code"]] = name2cid.get(tg.NAME_FIX.get(t["name"], t["name"]))
    EDRANK = {3: 10, 4: 9, 1: 8, 5: 7, 10: 6, 2: 5, 6: 4, 7: 3, 8: 2, 9: 1}
    snap = {}
    for sid, fmv, ca, nm in con.execute("SELECT s.snapshot_id,s.fm_version_id,s.ca,p.norm_name FROM player_snapshot s JOIN player p ON p.player_id=s.player_id"):
        r = EDRANK.get(fmv, 0); cur = snap.get(nm)
        if cur is None or r > cur[1] or (r == cur[1] and (ca or 0) > cur[2]):
            snap[nm] = (sid, r, ca or 0)
    chosen = set(v[0] for v in snap.values()); ab = defaultdict(dict)
    for sid, name, val in con.execute("SELECT snapshot_id,attr_name,attr_value FROM player_attribute"):
        if sid in chosen: ab[sid][name] = val
    def vec_for(full):
        s = snap.get(db.norm(full))
        if not s: return None
        v = np.zeros(A, np.float32)
        for nm, vl in ab.get(s[0], {}).items():
            j = aidx.get(nm)
            if j is not None: v[j] = vl
        return v
    def side(xi):
        ps = [(tg.pos_role(p.get("pos")), vec_for(p.get("full", ""))) for p in xi]
        ps = [(r, v if v is not None else role_mean[r]) for r, v in ps]
        ps.sort(key=lambda t: t[0]); ps = ps[:11] + [(2, role_mean[2])] * max(0, 11 - len(ps))
        return np.stack([v for _, v in ps[:11]]), np.array([r for r, _ in ps[:11]], np.int64)
    rosters = {}
    for f in (WC / "teams").glob("*.json"):
        d = json.load(open(f, encoding="utf-8")); rosters[d["team"]["code"]] = set(db.norm(p["name"]).split()[-1] for p in d["players"])
    def detect(xi, a, b):
        sn = [db.norm(p.get("full", "")).split()[-1] for p in xi if p.get("full")]
        na = sum(s in rosters.get(a, ()) for s in sn); nb = sum(s in rosters.get(b, ()) for s in sn)
        return a if na >= nb else b
    L = json.load(open(WC / "lineups.json", encoding="utf-8"))
    Rz = json.load(open(WC / "results.json", encoding="utf-8"))
    keys, LH, LA, HS, AS, ST = [], [], [], [], [], []
    for key, gg in L.items():
        res = Rz.get(key, {})
        if res.get("status") != "finished":
            continue
        ca0, cb0 = key.split("-"); hc = detect(gg.get("home_xi", []), ca0, cb0); ac = cb0 if hc == ca0 else ca0
        Xh, Rh = side(gg.get("home_xi", [])); Xa, Ra = side(gg.get("away_xi", []))
        ctx = tg.ctx_vec(natctx.get(teams.get(hc), (tg.BASE, 1, 0)), natctx.get(teams.get(ac), (tg.BASE, 1, 0)))
        Xh = ((Xh - mu) / sd).astype(np.float32); Xa = ((Xa - mu) / sd).astype(np.float32); ctxn = ((ctx - cmu) / csd).astype(np.float32)
        lhs, las = [], []
        with torch.no_grad():
            for nt in nets:
                lh, la = nt(_T(Xh[None]), _T(Rh[None]), _T(Xa[None]), _T(Ra[None]), _T(ctxn[None]))
                lhs.append(math.exp(float(lh[0]))); las.append(math.exp(float(la[0])))
        keys.append(f"{hc}-{ac}"); LH.append(lhs); LA.append(las)
        HS.append(min(int(res["hs"]), GG)); AS.append(min(int(res["as"]), GG)); ST.append(res.get("status", ""))
    np.savez_compressed(WCACHE, keys=np.array(keys), lh=np.array(LH), la=np.array(LA),
                        hs=np.array(HS), as_=np.array(AS))
    print(f"  saved {WCACHE}: {len(keys)} played WC games", flush=True)


def main():
    def arg(k, d): return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    nseed = int(arg("--seeds", "5")); ep = int(arg("--epochs", "60"))
    print("=== building held-out test cache (train-split seeds) ===", flush=True)
    build_test_cache(nseed, ep)
    print("=== building WC played cache (full-data goalnet.pt) ===", flush=True)
    build_wc_cache()
    # sanity baseline
    c = load_cache(); emp = empirical_grid()
    for tag, lh, la, y, hg, ag, msk in [
        ("TEST all", c["te_lh"], c["te_la"], c["te_y"], c["te_hg"], c["te_ag"], np.ones(len(c["te_y"]), bool)),
        ("TEST natl", c["te_lh"], c["te_la"], c["te_y"], c["te_hg"], c["te_ag"], c["te_natl"].astype(bool))]:
        grids = [ens_grid(lh[:, i], la[:, i], 0.0) for i in np.where(msk)[0]]
        s = score_grids(grids, y[msk], hg[msk], ag[msk])
        print(f"  baseline {tag:10s} rps={s['rps']:.4f} pg={s['pg']:.4f} exact={s['exact']}/{s['n']}", flush=True)
    w = c["wc"]; wg = [ens_grid(w["lh"][i], w["la"][i], 0.0) for i in range(len(w["keys"]))]
    tot = ex = 0
    for i in range(len(w["keys"])):
        pk = tg.ev_pick(wg[i]); pts, lab = tg.grade(pk, int(w["hs"][i]), int(w["as_"][i])); tot += pts; ex += lab == "exact"
    print(f"  baseline WC played pts={tot}/{len(wg)} games (exact={ex})", flush=True)


if __name__ == "__main__":
    main()
