"""Head-to-head on the 36 played WC2026 games: OUR MODEL vs the REAL BOOKMAKER (Bet365 1X2 from
betexplorer, data/wc_odds.csv). Outcome: de-vig the 1X2 -> implied W/D/L. Exact: MARKET-DC (fit Poisson
rates to the 1X2 -> EV-pick) since no correct-score market is archived. Our model loads goalnet.pt and
predicts each game's scoreline from the confirmed XIs. Scores both on accuracy, RPS, fantasy pts, exact%.
Usage: python D:/Programming/claude/FM/src/wc_book_compare.py
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db, train_goals as tg
WC = Path(r"D:\Programming\claude\worldcup\team_db")
FMV = 3
# betexplorer slug-name -> FM/worldcup nation name
SLUG2NAME = {"usa": "United States", "d-r-congo": "Congo DR", "ivory-coast": "Côte d'Ivoire",
             "cape-verde": "Cabo Verde", "south-korea": "Korea Republic", "czech-republic": "Czechia",
             "bosnia-herzegovina": "Bosnia and Herzegovina", "saudi-arabia": "Saudi Arabia",
             "new-zealand": "New Zealand", "south-africa": "South Africa",
             "turkey": "Türkiye", "iran": "IR Iran"}


def main():
    c = torch.load(ROOT / "data" / "goalnet.pt", weights_only=False)
    A, nctx, rho = c["A"], c["nctx"], c["rho"]
    mu, sd, cmu, csd = c["mu"], c["sd"], c["cmu"], c["csd"]
    ATTRS = c["attrs"]; aidx = {n: i for i, n in enumerate(ATTRS)}; role_mean = c["role_mean"]
    net = tg.GoalNet(A, nctx); net.load_state_dict(c["state"]); net.eval()
    con = db.connect(); natctx = tg.national_context(con)
    name2cid = {r[1]: r[0] for r in con.execute("SELECT club_id,name FROM club")}
    teams = {}; norm2code = {}
    for f in (WC / "teams").glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))["team"]
        teams[t["code"]] = name2cid.get(tg.NAME_FIX.get(t["name"], t["name"]))
        norm2code[db.norm(t["name"])] = t["code"]
    EDRANK = {3: 10, 4: 9, 1: 8, 5: 7, 10: 6, 2: 5, 6: 4, 7: 3, 8: 2, 9: 1}
    snap = {}
    for sid, fmv, ca, nm in con.execute("SELECT s.snapshot_id,s.fm_version_id,s.ca,p.norm_name FROM player_snapshot s JOIN player p ON p.player_id=s.player_id"):
        r = EDRANK.get(fmv, 0); cur = snap.get(nm)
        if cur is None or r > cur[1] or (r == cur[1] and (ca or 0) > cur[2]): snap[nm] = (sid, r, ca or 0)
    chosen = set(v[0] for v in snap.values()); ab = defaultdict(dict)
    for sid, name, val in con.execute("SELECT snapshot_id,attr_name,attr_value FROM player_attribute"):
        if sid in chosen: ab[sid][name] = val
    def vec(full):
        s = snap.get(db.norm(full));
        if not s: return None
        v = np.zeros(A, np.float32)
        for nm, vl in ab.get(s[0], {}).items():
            j = aidx.get(nm)
            if j is not None: v[j] = vl
        return v
    def side(xi):
        ps = [(tg.pos_role(p.get("pos")), vec(p.get("full", ""))) for p in xi]
        imp = sum(v is None for _, v in ps)
        ps = [(r, v if v is not None else role_mean[r]) for r, v in ps]
        ps.sort(key=lambda t: t[0]); ps = ps[:11] + [(2, role_mean[2])] * max(0, 11 - len(ps))
        return np.stack([v for _, v in ps[:11]]), np.array([r for r, _ in ps[:11]], np.int64), imp
    def T(a):
        a = np.ascontiguousarray(a); tt = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
        return torch.frombuffer(bytearray(a.tobytes()), dtype=tt[a.dtype]).reshape(a.shape)
    L = json.load(open(WC / "lineups.json", encoding="utf-8"))
    rosters = {}
    for f in (WC / "teams").glob("*.json"):
        d = json.load(open(f, encoding="utf-8")); rosters[d["team"]["code"]] = set(db.norm(p["name"]).split()[-1] for p in d["players"])

    def code_of(slug):
        # try longest-match against known multi-word names, else de-slug -> norm
        nm = SLUG2NAME.get(slug)
        if nm: return norm2code.get(db.norm(nm))
        return norm2code.get(db.norm(slug.replace("-", " ")))

    def split_slug(slug):
        # split "germany-ivory-coast" into two team slugs by trying all 2-part splits that both map
        parts = slug.split("-")
        for i in range(1, len(parts)):
            a, b = "-".join(parts[:i]), "-".join(parts[i:])
            if code_of(a) and code_of(b): return code_of(a), code_of(b)
        return None, None

    def find_xis(hc, ac):
        for k in (f"{hc}-{ac}", f"{ac}-{hc}"):
            if k in L:
                g = L[k]; a0, b0 = k.split("-")
                # detect which XI is home team (hc)
                sn = [db.norm(p.get("full", "")).split()[-1] for p in g.get("home_xi", [])]
                home_is_a0 = sum(s in rosters.get(a0, ()) for s in sn) >= sum(s in rosters.get(b0, ()) for s in sn)
                hx = g["home_xi"] if (home_is_a0 == (a0 == hc)) else g["away_xi"]
                ax = g["away_xi"] if hx is g["home_xi"] else g["home_xi"]
                return hx, ax
        return None, None

    def model_lambda(hc, ac):
        hx, ax = find_xis(hc, ac)
        if hx is None: return None
        Xh, Rh, i1 = side(hx); Xa, Ra, i2 = side(ax)
        ctx = tg.ctx_vec(natctx.get(teams.get(hc), (tg.BASE, 1, 0)), natctx.get(teams.get(ac), (tg.BASE, 1, 0)))
        Xh = ((Xh - mu) / sd).astype(np.float32); Xa = ((Xa - mu) / sd).astype(np.float32); cx = ((ctx - cmu) / csd).astype(np.float32)
        with torch.no_grad():
            lh, la = net(T(Xh[None]), T(Rh[None]), T(Xa[None]), T(Ra[None]), T(cx[None]))
        return math.exp(float(lh[0])), math.exp(float(la[0])), i1 + i2

    # MARKET-DC: grid fit lambda to de-vigged 1X2
    gridL = np.linspace(0.15, 4.0, 40); _lg = np.array([math.lgamma(x + 1) for x in range(tg.MAXG + 1)])
    def outp(a, b):
        k = np.arange(tg.MAXG + 1); pa = np.exp(k * np.log(a) - a - _lg); pb = np.exp(k * np.log(b) - b - _lg); P = np.outer(pa, pb)
        return np.array([np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()])
    KL = [(a, b) for a in gridL for b in gridL]; KP = np.array([outp(a, b) for a, b in KL])

    rows = [l.strip().split("|") for l in open(ROOT / "data" / "wc_odds.csv") if l.strip()]
    M = dict(our_pts=0, our_ex=0, our_oc=0, bk_pts=0, bk_ex=0, bk_oc=0, n=0, imp=0)
    ourP, bkP, ys = [], [], []
    miss = []
    for slug, hs, asg, oH, oD, oA in rows:
        hs, asg = int(hs), int(asg); o = np.array([1/float(oH), 1/float(oD), 1/float(oA)]); o = o/o.sum()
        hc, ac = split_slug(slug)
        ml = model_lambda(hc, ac) if hc and ac else None
        if ml is None:
            miss.append(slug); continue
        lh, la, imp = ml; M["imp"] += imp; M["n"] += 1
        ytrue = 0 if hs > asg else (1 if hs == asg else 2); ys.append(ytrue)
        # our model
        Pm = tg.score_matrix(lh, la, rho); ourP.append(tg.hda_from_P(Pm)); pk = tg.ev_pick(Pm)
        pts, lab = tg.grade(pk, hs, asg); M["our_pts"] += pts; M["our_ex"] += lab == "exact"; M["our_oc"] += lab != "wrong"
        # bookmaker
        bkP.append(o); bl = KL[int(((KP - o) ** 2).sum(1).argmin())]
        Pb = tg.score_matrix(bl[0], bl[1], 0.0); bpk = tg.ev_pick(Pb)
        bpts, blab = tg.grade(bpk, hs, asg); M["bk_pts"] += bpts; M["bk_ex"] += blab == "exact"; M["bk_oc"] += blab != "wrong"
    ys = np.array(ys); ourP = np.array(ourP); bkP = np.array(bkP)
    def rps(P): cp = np.cumsum(P, 1); co = np.cumsum(np.eye(3)[ys], 1); return float(np.mean(np.sum((cp - co) ** 2, 1) / 2))
    n = M["n"]
    print(f"\n=== OUR MODEL vs BOOKMAKER — {n} WC2026 games ({M['imp']} imputed starters) ===")
    print(f"  {'':10s}  outcome-correct   RPS      exact   total-pts")
    print(f"  OUR MODEL   {M['our_oc']}/{n} ({M['our_oc']/n*100:.0f}%)   {rps(ourP):.4f}   {M['our_ex']}/{n}   {M['our_pts']}")
    print(f"  BOOKMAKER   {M['bk_oc']}/{n} ({M['bk_oc']/n*100:.0f}%)   {rps(bkP):.4f}   {M['bk_ex']}/{n}   {M['bk_pts']}")
    if miss: print(f"  (unmatched fixtures: {miss})")


if __name__ == "__main__":
    main()
