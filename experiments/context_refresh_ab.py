"""A/B: does feeding the tournament results-so-far into GoalNet's context (Elo/form) improve its WC picks?
The model's context currently stops at the pre-tournament DB state (June 2). Here we replay the played WC
games chronologically and predict each one TWICE with the SAME GoalNet weights + lineups, differing only in
context:
  OLD = frozen pre-WC Elo/form (what predict_game uses now).
  NEW = Elo/form updated with the earlier WC games (no leakage: game k sees only games 1..k-1).
Reports exact / chalk-pts / exacts-pts / RPS for OLD vs NEW on the played games.
Usage: python experiments/context_refresh_ab.py
"""
import sys, json, math
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import db, train_goals as tg, predict_game as pg
WC = Path(r"D:\Programming\claude\worldcup\team_db")
NATc = {9, 10, 11, 12, 13, 14, 15}
K, HADV, BASE = tg.K, tg.HADV, tg.BASE


def T(a):
    a = np.ascontiguousarray(a); tt = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    return torch.frombuffer(bytearray(a.tobytes()), dtype=tt[a.dtype]).reshape(a.shape)


def pre_wc_state(con):
    """Running Elo/form/gd dicts after replaying all national matches in the DB (= pre-tournament snapshot)."""
    rows = con.execute(f"""SELECT home_club_id,away_club_id,home_goals,away_goals FROM match
        WHERE competition_id IN {tuple(NATc)} AND home_goals IS NOT NULL ORDER BY match_date,match_id""").fetchall()
    elo = defaultdict(lambda: BASE); form = defaultdict(lambda: deque(maxlen=5)); gd = defaultdict(lambda: deque(maxlen=5))
    for hc, ac, hg, ag in rows:
        update(elo, form, gd, hc, ac, hg, ag)
    return elo, form, gd


def update(elo, form, gd, hc, ac, hg, ag):
    eh, ea = elo[hc], elo[ac]
    exp = 1 / (1 + 10 ** (-((eh + HADV) - ea) / 400.0)); s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    elo[hc] = eh + K * (s - exp); elo[ac] = ea - K * (s - exp)
    ph = 3 if hg > ag else (1 if hg == ag else 0)
    form[hc].append(ph); form[ac].append(3 - ph if ph != 1 else 1)
    gd[hc].append(hg - ag); gd[ac].append(ag - hg)


def tup(elo, form, gd, cid):
    return (elo[cid], np.mean(form[cid]) if form[cid] else 1.0, np.mean(gd[cid]) if gd[cid] else 0.0)


def main():
    c = torch.load(ROOT / "data" / "goalnet.pt", weights_only=False)
    A, nctx = c["A"], c["nctx"]; mu, sd, cmu, csd, rho = c["mu"], c["sd"], c["cmu"], c["csd"], c["rho"]
    ATTRS = c["attrs"]; aidx = {n: i for i, n in enumerate(ATTRS)}; role_mean = c["role_mean"]
    states = c.get("states") or [c["state"]]; nets = []
    for st in states:
        nt = tg.GoalNet(A, nctx); nt.load_state_dict(st); nt.eval(); nets.append(nt)
    con = db.connect()
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
        na = sum(s in rosters.get(a, ()) for s in sn); nb = sum(s in rosters.get(b, ()) for s in sn); return a if na >= nb else b
    L = json.load(open(WC / "lineups.json", encoding="utf-8")); Rz = json.load(open(WC / "results.json", encoding="utf-8"))

    # frozen pre-WC state, and a live copy we update through the tournament
    elo0, form0, gd0 = pre_wc_state(con)
    eloN = defaultdict(lambda: BASE, dict(elo0)); formN = defaultdict(lambda: deque(maxlen=5))
    gdN = defaultdict(lambda: deque(maxlen=5))
    for k in form0: formN[k] = deque(form0[k], maxlen=5)
    for k in gd0: gdN[k] = deque(gd0[k], maxlen=5)

    games = []
    for key, e in L.items():
        res = Rz.get(key, {})
        if res.get("status") != "finished": continue
        games.append((res.get("kickoff", 0), key, e, res))
    games.sort(key=lambda t: t[0])

    def grid_for(Xh, Rh, Xa, Ra, ctx):
        Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
        ctxn = ((ctx - cmu) / csd).astype(np.float32); gs = []
        with torch.no_grad():
            for nt in nets:
                lh, la = nt(T(Xhn[None]), T(Rh[None]), T(Xan[None]), T(Ra[None]), T(ctxn[None]))
                gs.append(tg.score_matrix(math.exp(float(lh[0])), math.exp(float(la[0])), rho))
        P = np.mean(gs, 0); return P / P.sum()

    agg = {"OLD": dict(ex=0, chalk=0, exacts=0, rps=0.0), "NEW": dict(ex=0, chalk=0, exacts=0, rps=0.0)}
    n = 0
    for _, key, e, res in games:
        ca0, cb0 = key.split("-"); hc = detect(e.get("home_xi", []), ca0, cb0); ac = cb0 if hc == ca0 else ca0
        hcid, acid = teams.get(hc), teams.get(ac)
        if hcid is None or acid is None: continue
        Xh, Rh = side(e.get("home_xi", [])); Xa, Ra = side(e.get("away_xi", []))
        hs, as_ = int(res["hs"]), int(res["as"])
        ctx_old = tg.ctx_vec(tup(elo0, form0, gd0, hcid), tup(elo0, form0, gd0, acid))
        ctx_new = tg.ctx_vec(tup(eloN, formN, gdN, hcid), tup(eloN, formN, gdN, acid))
        for tag, ctx in (("OLD", ctx_old), ("NEW", ctx_new)):
            P = grid_for(Xh, Rh, Xa, Ra, ctx)
            ck = pg.pick_strategy(P, "chalk"); ex = pg.pick_strategy(P, "exacts")
            agg[tag]["chalk"] += tg.grade(ck, hs, as_)[0]; agg[tag]["exacts"] += tg.grade(ex, hs, as_)[0]
            agg[tag]["ex"] += int(tg.grade(ck, hs, as_)[1] == "exact")
            P3 = tg.hda_from_P(P); y = 0 if hs > as_ else (1 if hs == as_ else 2); agg[tag]["rps"] += tg.rps(np.array([y]), P3[None])
        update(eloN, formN, gdN, hcid, acid, hs, as_)   # advance live state AFTER predicting (no leakage)
        n += 1
    print(f"=== context-refresh A/B on {n} played WC games (same GoalNet weights + lineups) ===", flush=True)
    print(f"  {'context':6s} {'chalk pts':>10s} {'exacts pts':>11s} {'exact(chalk)':>13s} {'RPS':>8s}", flush=True)
    for tag in ("OLD", "NEW"):
        a = agg[tag]; print(f"  {tag:6s} {a['chalk']:10d} {a['exacts']:11d} {a['ex']:13d} {a['rps']/n:8.4f}", flush=True)


if __name__ == "__main__":
    main()
