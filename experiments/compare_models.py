"""Before/after on the played WC2026 games: score a given checkpoint's 5-seed ensemble over every finished
game, under both chalk and contrarian picks. Run for the all-round and the national-specialised models to see
the production delta on the actual target. Usage: python experiments/compare_models.py <ckpt.pt> [ckpt2.pt ...]
"""
import sys, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import db, train_goals as tg
import predict_game as pg   # for pick_strategy
WC = Path(r"D:\Programming\claude\worldcup\team_db")
NATc = {9, 10, 11, 12, 13, 14, 15}


def T(a):
    a = np.ascontiguousarray(a); tt = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    return torch.frombuffer(bytearray(a.tobytes()), dtype=tt[a.dtype]).reshape(a.shape)


def wc_grids(ckpt):
    c = torch.load(ckpt, weights_only=False)
    A, nctx = c["A"], c["nctx"]; mu, sd, cmu, csd, rho = c["mu"], c["sd"], c["cmu"], c["csd"], c["rho"]
    ATTRS = c["attrs"]; aidx = {n: i for i, n in enumerate(ATTRS)}; role_mean = c["role_mean"]
    states = c.get("states") or [c["state"]]; nets = []
    for st in states:
        nt = tg.GoalNet(A, nctx); nt.load_state_dict(st); nt.eval(); nets.append(nt)
    con = db.connect(); natctx = tg.national_context(con)
    name2cid = {r[1]: r[0] for r in con.execute("SELECT club_id,name FROM club")}
    teams = {}
    for f in (WC / "teams").glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))["team"]; teams[t["code"]] = name2cid.get(tg.NAME_FIX.get(t["name"], t["name"]))
    EDRANK = {3: 10, 4: 9, 1: 8, 5: 7, 10: 6, 2: 5, 6: 4, 7: 3, 8: 2, 9: 1}; snap = {}
    for sid, fmv, ca, nm in con.execute("SELECT s.snapshot_id,s.fm_version_id,s.ca,p.norm_name FROM player_snapshot s JOIN player p ON p.player_id=s.player_id"):
        r = EDRANK.get(fmv, 0); cur = snap.get(nm)
        if cur is None or r > cur[1] or (r == cur[1] and (ca or 0) > cur[2]): snap[nm] = (sid, r, ca or 0)
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
    L = json.load(open(WC / "lineups.json", encoding="utf-8")); Rz = json.load(open(WC / "results.json", encoding="utf-8"))
    out = []
    for key, gg in L.items():
        res = Rz.get(key, {})
        if res.get("status") != "finished": continue
        ca0, cb0 = key.split("-"); hc = detect(gg.get("home_xi", []), ca0, cb0); ac = cb0 if hc == ca0 else ca0
        Xh, Rh = side(gg.get("home_xi", [])); Xa, Ra = side(gg.get("away_xi", []))
        ctx = tg.ctx_vec(natctx.get(teams.get(hc), (tg.BASE, 1, 0)), natctx.get(teams.get(ac), (tg.BASE, 1, 0)))
        Xh = ((Xh - mu) / sd).astype(np.float32); Xa = ((Xa - mu) / sd).astype(np.float32); ctxn = ((ctx - cmu) / csd).astype(np.float32)
        grids = []
        with torch.no_grad():
            for nt in nets:
                lh, la = nt(T(Xh[None]), T(Rh[None]), T(Xa[None]), T(Ra[None]), T(ctxn[None]))
                grids.append(tg.score_matrix(math.exp(float(lh[0])), math.exp(float(la[0])), rho))
        P = np.mean(grids, 0); P = P / P.sum()
        out.append((f"{hc}-{ac}", P, int(res["hs"]), int(res["as"])))
    return out


def tally(grids, strat):
    tot = ex = 0
    for key, P, hs, as_ in grids:
        pk = pg.pick_strategy(P, strat, 0.25, 0.6); pts, lab = tg.grade(pk, hs, as_); tot += pts; ex += lab == "exact"
    return tot, ex


def main():
    for ckpt in sys.argv[1:]:
        grids = wc_grids(ckpt); n = len(grids)
        ch = tally(grids, "chalk"); co = tally(grids, "contrarian")
        print(f"{Path(ckpt).name:24s} ({n} games)  chalk={ch[0]} (ex{ch[1]})  contrarian={co[0]} (ex{co[1]})", flush=True)


if __name__ == "__main__":
    main()
