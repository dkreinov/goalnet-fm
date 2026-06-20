"""Score OUR best 11v11 model (xfmr + national-upweighted context) on the WC2026 games already played,
under the fantasy league scoring (exact score = 3, correct outcome = 1, wrong = 0). Compares to the
naive 'favourite by Elo' baseline and reports exact/correct/wrong/total to stack against the leaderboard.

Pipeline: train the model on players.npz/context.npz (national sample-weight W); compute national-team
Elo/form from fm.db international matches; for each played game assemble both XIs (lineups.json) from
FM26 grades (impute missing starters with the role-group mean), build the matching context, predict
H/D/A, pick a modal scoreline, and score vs the real result (results.json).
Usage: python D:/Programming/claude/FM/src/wc2026_score.py [--w 5] [--epochs 120]
"""
import json
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
NAT_COMPS = (9, 10, 11, 12, 13, 14, 15)
FMV = 3
ROLE = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}
MODAL = {0: (1, 0), 1: (1, 1), 2: (0, 1)}
# worldcup position code -> role bucket
def pos_role(p):
    p = (p or "").upper()
    if p.startswith("G"):
        return 0
    if p[:2] in ("CD", "LB", "RB", "WB") or p.startswith("D"):
        return 1
    if p.startswith("F") or p.startswith("ST") or p in ("LW", "RW") or p.startswith("W"):
        return 3
    return 2     # DM/CM/AM/LM/RM/M and anything else
NAME_FIX = {"IR Iran": "Iran", "Korea Republic": "South Korea", "Côte d'Ivoire": "Ivory Coast",
            "Cabo Verde": "Cape Verde", "Bosnia and Herzegovina": "Bosnia and Herzegovina"}
K, HADV, BASE = 20.0, 60.0, 1500.0


def national_context(con):
    """Elo + recent points-form + recent goal-diff form per national-team club_id, over all intl matches
    chronologically (pre-WC2026 frozen strength). Returns dict cid -> (elo, form, gdform)."""
    rows = con.execute(
        f"""SELECT match_date, home_club_id, away_club_id, home_goals, away_goals FROM match
            WHERE competition_id IN {NAT_COMPS} AND home_goals IS NOT NULL ORDER BY match_date, match_id""").fetchall()
    elo = defaultdict(lambda: BASE)
    form = defaultdict(lambda: deque(maxlen=5)); gd = defaultdict(lambda: deque(maxlen=5))
    for _, hc, ac, hg, ag in rows:
        eh, ea = elo[hc], elo[ac]
        exp = 1.0 / (1.0 + 10 ** (-((eh + HADV) - ea) / 400.0))
        s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo[hc] = eh + K * (s - exp); elo[ac] = ea - K * (s - exp)
        ph = 3 if hg > ag else (1 if hg == ag else 0)
        form[hc].append(ph); form[ac].append(3 - ph if ph != 1 else 1)
        gd[hc].append(hg - ag); gd[ac].append(ag - hg)
    out = {}
    for cid in set(list(elo)):
        out[cid] = (elo[cid], np.mean(form[cid]) if form[cid] else 1.0,
                    np.mean(gd[cid]) if gd[cid] else 0.0)
    return out


def ctx_vector(h, a):
    """Build the 10-feature context [as in build_context.py] for a game from two (elo,form,gd) tuples."""
    eh, fh, gh = h; ea, fa, ga = a
    return np.array([eh / 400.0, ea / 400.0, (eh - ea) / 400.0, fh, fa, fh - fa,
                     gh, ga, 0.5, 0.5], dtype=np.float32)   # rest-days neutral (no per-game rest)


def main():
    def arg(k, d):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    W = float(arg("--w", "5"))
    ep = int(arg("--epochs", "120"))

    # ---- load training data ----
    z = np.load(ROOT / "data" / "players.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]
    Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    ATTRS = [str(a) for a in z["attrs"]]; A = len(ATTRS)
    aidx = {n: i for i, n in enumerate(ATTRS)}
    cz = np.load(ROOT / "data" / "context.npz")
    cctx, cmids = cz["ctx"], cz["mids"]
    cmap = {int(m): cctx[i] for i, m in enumerate(cmids)}
    nctx = cctx.shape[1]
    CTX = np.stack([cmap.get(m, np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)

    con = db.connect()
    NATc = {9, 10, 11, 12, 13, 14, 15}
    comp = {r[0]: r[1] for r in con.execute("SELECT match_id,competition_id FROM match")}
    natl = np.array([comp.get(m, 0) in NATc for m in mids])

    tr = dates < np.datetime64("2024-08-01")
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6
    CTXn = ((CTX - cmu) / csd).astype(np.float32)
    role_mean = {r: Xh[tr][Rh[tr] == r].mean(0) for r in range(4)}   # raw role-mean for imputation

    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a)
        return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t):
        return np.array(t.detach().tolist(), dtype=np.float32)

    # ---- train model (xfmr + ctx, national weight W) ----
    print(f"training xfmr+ctx (national weight W={W})...", flush=True)
    torch.manual_seed(7); np.random.seed(7)
    net = tp.PosNet(A, "xfmr", nctx=nctx)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    lossf = nn.CrossEntropyLoss(reduction="none")
    Xhtr, Rhtr, Xatr, Ratr, Ctr, ytr = (T(Xhn[tr]), T(Rh[tr]), T(Xan[tr]), T(Ra[tr]), T(CTXn[tr]), T(y[tr]))
    wt = T(np.where(natl[tr], W, 1.0).astype(np.float32))
    va = (dates >= np.datetime64("2024-08-01")) & (dates < np.datetime64("2025-08-01"))
    Vh, Vrh, Va_, Vra, Cv = T(Xhn[va]), T(Rh[va]), T(Xan[va]), T(Ra[va]), T(CTXn[va])
    bs, n = 512, Xhtr.size(0); best, bstate, bad = 9, None, 0
    for e in range(ep):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            l = (lossf(net(Xhtr[b], Rhtr[b], Xatr[b], Ratr[b], Ctr[b]), ytr[b]) * wt[b]).mean()
            l.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = tonp(torch.softmax(net(Vh, Vrh, Va_, Vra, Cv), 1))
        r = float(np.mean(np.sum((np.cumsum(pv, 1) - np.cumsum(np.eye(3)[y[va]], 1)) ** 2, 1) / 2))
        if r < best - 1e-4:
            best, bstate, bad = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 20:
            break
    net.load_state_dict(bstate); net.eval()
    print(f"  val rps={best:.4f}", flush=True)

    # ---- national context + team mapping ----
    natctx = national_context(con)
    cname = {r[0]: r[1] for r in con.execute("SELECT club_id,name FROM club")}
    name2cid = {v: k for k, v in cname.items()}
    teams = {}     # team code -> club_id
    for f in (WC / "teams").glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))["team"]
        nm = NAME_FIX.get(t["name"], t["name"])
        teams[t["code"]] = name2cid.get(nm)

    # FM26 grade vectors keyed by norm name (pick highest-CA snapshot per name)
    snap = {}      # norm_name -> snapshot_id (prefer higher ca)
    for sid, pid, ca, nn_ in con.execute(
            "SELECT s.snapshot_id,s.player_id,s.ca,p.norm_name FROM player_snapshot s "
            "JOIN player p ON p.player_id=s.player_id WHERE s.fm_version_id=?", (FMV,)):
        if nn_ not in snap or (ca or 0) > snap[nn_][1]:
            snap[nn_] = (sid, ca or 0)
    attrs_by_sid = defaultdict(dict)
    for sid, cat, name, val in con.execute(
            "SELECT a.snapshot_id,a.category,a.attr_name,a.attr_value FROM player_attribute a "
            "JOIN player_snapshot s ON s.snapshot_id=a.snapshot_id WHERE s.fm_version_id=?", (FMV,)):
        attrs_by_sid[sid][name] = val

    def vec_for(full):
        s = snap.get(db.norm(full))
        if not s:
            return None
        v = np.zeros(A, dtype=np.float32)
        for name, val in attrs_by_sid.get(s[0], {}).items():
            j = aidx.get(name)
            if j is not None:
                v[j] = val
        return v

    def side_tensor(xi):
        """Return (X[11,A], R[11], n_imputed) ordered GK->DEF->MID->ATT, imputing misses with role-mean."""
        players = []
        for p in xi:
            r = pos_role(p.get("pos"))
            v = vec_for(p.get("full", ""))
            players.append((r, v))
        imp = sum(1 for _, v in players if v is None)
        filled = [(r, v if v is not None else role_mean[r]) for r, v in players]
        filled.sort(key=lambda t: t[0])
        filled = filled[:11] + [(2, role_mean[2])] * max(0, 11 - len(filled))
        X = np.stack([v for _, v in filled[:11]]); R = np.array([r for r, _ in filled[:11]], np.int64)
        return X, R, imp

    # ---- score the played games ----
    L = json.load(open(WC / "lineups.json", encoding="utf-8"))
    Rz = json.load(open(WC / "results.json", encoding="utf-8"))
    games = [(k, v) for k, v in Rz.items() if v.get("status") == "finished" and k in L]
    rows = []
    for k, res in games:
        g = L[k]; hc, ac = res["home"], res["away"]
        Xh1, Rh1, ih = side_tensor(g.get("home_xi", []))
        Xa1, Ra1, ia = side_tensor(g.get("away_xi", []))
        hcid, acid = teams.get(hc), teams.get(ac)
        ctx = ctx_vector(natctx.get(hcid, (BASE, 1.0, 0.0)), natctx.get(acid, (BASE, 1.0, 0.0)))
        Xh1 = ((Xh1 - mu) / sd).astype(np.float32); Xa1 = ((Xa1 - mu) / sd).astype(np.float32)
        ctxn = ((ctx - cmu) / csd).astype(np.float32)
        with torch.no_grad():
            p = tonp(torch.softmax(net(T(Xh1[None]), T(Rh1[None]), T(Xa1[None]), T(Ra1[None]),
                                       T(ctxn[None])), 1))[0]
        out = int(p.argmax())
        rows.append((k, hc, ac, res["hs"], res["as"], out, p, ih + ia, natctx.get(hcid), natctx.get(acid)))

    # scoring
    def grade(pred_score, hs, as_):
        ph, pa = pred_score
        if ph == hs and pa == as_:
            return 3, "exact"
        ro = lambda x, y: (1 if x > y else (0 if x == y else -1))
        return (1, "correct") if ro(ph, pa) == ro(hs, as_) else (0, "wrong")

    tot = ex = co = wr = 0; ntot = nex = nco = nwr = 0
    print("\n  game            res   ourpick  pts  imp", flush=True)
    for k, hc, ac, hs, as_, out, p, imp, hh, aa in rows:
        ph, pa = MODAL[out]
        pts, lab = grade((ph, pa), hs, as_)
        tot += pts; ex += lab == "exact"; co += lab == "correct"; wr += lab == "wrong"
        # naive favourite by Elo
        eh = (hh or (BASE,))[0]; ea = (aa or (BASE,))[0]
        nfav = (1, 0) if eh >= ea else (0, 1)
        npts, nlab = grade(nfav, hs, as_)
        ntot += npts; nex += nlab == "exact"; nco += nlab == "correct"; nwr += nlab == "wrong"
        print(f"  {hc}-{ac:<3} {hs}-{as_}   {ph}-{pa} ({'HDA'[out]})  {pts}   {imp}", flush=True)
    n = len(rows)
    print(f"\n=== OUR MODEL (W={W}) on {n} played games ===", flush=True)
    print(f"  exact={ex} correct={co} wrong={wr}  ->  TOTAL = {tot}  (3*{ex}+{co})", flush=True)
    print(f"=== NAIVE favourite-by-Elo ===", flush=True)
    print(f"  exact={nex} correct={nco} wrong={nwr}  ->  TOTAL = {ntot}", flush=True)
    print(f"\n  (leaderboard ref: YOU=19 [0 exact,19 correct], top=31 [6 exact,13 correct])", flush=True)
    tot_imp = sum(r[7] for r in rows)
    print(f"  NOTE: {tot_imp} of {n*22} starters imputed (role-mean) at current grade coverage", flush=True)


if __name__ == "__main__":
    main()
