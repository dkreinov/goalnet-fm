"""Pure INFERENCE: load the trained GoalNet checkpoint (data/goalnet.pt) and predict any WC2026 game.
No training — just loads weights, builds the FM26 grade lookup + national context from fm.db (fast reads),
runs one forward pass, and prints the scoreline distribution + EV-optimal pick. Run train_goals.py --full
once to produce the checkpoint; after that predictions are instant.
Usage: python D:/Programming/claude/FM/src/predict_game.py NED-SWE [more KEYS...]
"""
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db
import train_goals as tg

WC = Path(r"D:\Programming\claude\worldcup\team_db")
FMV = 3


def _ev_grid(P):
    """EV(fantasy points) for every candidate pick cell (exact=3, correct outcome=1)."""
    ho = tg.hda_from_P(P); M = P.shape[0]; EV = np.zeros_like(P)
    for i in range(M):
        for j in range(M):
            oc = 0 if i > j else (1 if i == j else 2)
            EV[i, j] = 3 * P[i, j] + (ho[oc] - P[i, j])
    return EV


_EMP = None
def _empirical_grid():
    """Historical final-score distribution (cached), capped to the MAXG grid — real scores cluster on 1-1,
    1-0, 2-1, ... which independent double-Poisson under-weights. Used by the exact-hunting strategy."""
    global _EMP
    if _EMP is None:
        con = db.connect(); M = tg.MAXG + 1; E = np.zeros((M, M))
        for h, a in con.execute("SELECT home_goals,away_goals FROM match WHERE home_goals IS NOT NULL"):
            E[min(h, tg.MAXG), min(a, tg.MAXG)] += 1
        _EMP = E / E.sum()
    return _EMP


def pick_strategy(P, strategy="chalk", beta=0.25, q=0.6):
    """chalk = max E(points) (production EV-pick). exact = argmax P. contrarian = max E(points) - beta*field-
    mass (E3): differentiate from a field that picks the chalk cell with prob q else samples the grid. Raises
    P(finish #1) in a chalk-clustered league at a small cost in expected points (see RESULTS_WC2026.md E3)."""
    if strategy == "chalk":
        return tg.ev_pick(P)
    if strategy == "exact":
        i, j = np.unravel_index(np.argmax(P), P.shape); return (int(i), int(j))
    if strategy == "contrarian":
        EV = _ev_grid(P); F = (1 - q) * P.copy()
        ci, cj = tg.ev_pick(P); F[ci, cj] += q
        i, j = np.unravel_index(np.argmax(EV - beta * F), P.shape); return (int(i), int(j))
    if strategy == "exacts":
        # maximise P(exact) on an empirically-corrected grid: shifts toward real common scorelines
        # (1-1 for draws, 2-1 over 2-0). alpha=0.30 measured best on national test (24 vs 22 exacts, 169 vs
        # 165 pts). For a player whose gap is exact-conversion, not outcomes.
        Q = 0.70 * P + 0.30 * _empirical_grid(); Q = Q / Q.sum()
        i, j = np.unravel_index(np.argmax(Q), Q.shape); return (int(i), int(j))
    if strategy == "gamble":
        # differentiated exact: the 2nd-most-likely exact on the corrected grid — decorrelated from the top
        # pick rivals play. For high-multiplier knockout games (QF+) where you must separate from the field.
        Q = 0.70 * P + 0.30 * _empirical_grid(); Q = Q / Q.sum()
        order = np.argsort(Q.ravel())[::-1]; idx = int(order[1] if order.size > 1 else order[0])
        return (idx // Q.shape[1], idx % Q.shape[1])
    raise ValueError(strategy)


def main():
    keys = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not keys:
        print("usage: predict_game.py NED-SWE [KEY ...] [--round group|r32|r16|qf|sf|final] "
              "[--rival unknown|safe|gambling] [--strategy chalk|exacts|contrarian|gamble] [--beta 0.25] [--q 0.6]")
        return
    def _arg(k, d): return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    global STRATEGY, BETA, QFIELD, ROUNDINFO
    BETA = float(_arg("--beta", "0.25")); QFIELD = float(_arg("--q", "0.6"))
    # --round auto-applies the league decision logic (PREDICTION_GUIDE.md): safe exact-hunting through the
    # low-multiplier rounds; from QF up (x8+) gamble (differentiate) UNLESS the rival is already gambling, in
    # which case stay safe and let their variance sink them. Explicit --strategy always overrides.
    ROUND_MULT = {"group": 1, "r32": 2, "r16": 4, "qf": 8, "sf": 16, "final": 32}   # real league (final exact=64)
    rnd = (_arg("--round", "") or "").lower(); rival = (_arg("--rival", "unknown") or "unknown").lower()
    ROUNDINFO = ""
    if "--strategy" in sys.argv:
        STRATEGY = _arg("--strategy", "chalk")
    elif rnd in ROUND_MULT:
        mult = ROUND_MULT[rnd]
        if mult >= 8 and rival != "gambling":
            STRATEGY = "gamble"; why = f"x{mult} {rnd.upper()}, rival={rival} -> differentiate"
        elif mult >= 8:
            STRATEGY = "exacts"; why = f"x{mult} {rnd.upper()}, rival gambling -> stay safe, let them swing"
        else:
            STRATEGY = "exacts"; why = f"x{mult} {rnd.upper()} -> safe exact-hunting"
        ROUNDINFO = f"  [round {rnd.upper()} x{mult}: {why}]"
    else:
        STRATEGY = "exacts"   # league default: hunt exacts
    ck = ROOT / "data" / "goalnet.pt"
    if not ck.exists():
        print("no data/goalnet.pt — run: python src/train_goals.py --full  (once)"); return
    c = torch.load(ck, weights_only=False)
    A, nctx = c["A"], c["nctx"]
    mu, sd, cmu, csd, rho = c["mu"], c["sd"], c["cmu"], c["csd"], c["rho"]
    ATTRS = c["attrs"]; aidx = {n: i for i, n in enumerate(ATTRS)}; role_mean = c["role_mean"]
    states = c.get("states") or [c["state"]]           # seed ensemble when present, else single model
    nets = []
    for st in states:
        nt = tg.GoalNet(A, nctx); nt.load_state_dict(st); nt.eval(); nets.append(nt)

    con = db.connect()
    natctx = tg.national_context(con)
    name2cid = {r[1]: r[0] for r in con.execute("SELECT club_id,name FROM club")}
    teams = {}
    for f in (WC / "teams").glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))["team"]
        teams[t["code"]] = name2cid.get(tg.NAME_FIX.get(t["name"], t["name"]))
    EDRANK = {3: 10, 4: 9, 1: 8, 5: 7, 10: 6, 2: 5, 6: 4, 7: 3, 8: 2, 9: 1}   # FM26 first, else newest
    snap = {}
    for sid, fmv, ca, nm in con.execute("SELECT s.snapshot_id,s.fm_version_id,s.ca,p.norm_name "
                                        "FROM player_snapshot s JOIN player p ON p.player_id=s.player_id"):
        r = EDRANK.get(fmv, 0); cur = snap.get(nm)
        if cur is None or r > cur[1] or (r == cur[1] and (ca or 0) > cur[2]):
            snap[nm] = (sid, r, ca or 0)
    chosen = set(v[0] for v in snap.values())
    ab = defaultdict(dict)
    for sid, name, val in con.execute("SELECT snapshot_id,attr_name,attr_value FROM player_attribute"):
        if sid in chosen:
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
        ps = [(tg.pos_role(p.get("pos")), vec_for(p.get("full", ""))) for p in xi]
        imp = sum(v is None for _, v in ps)
        ps = [(r, v if v is not None else role_mean[r]) for r, v in ps]
        ps.sort(key=lambda t: t[0]); ps = ps[:11] + [(2, role_mean[2])] * max(0, 11 - len(ps))
        return np.stack([v for _, v in ps[:11]]), np.array([r for r, _ in ps[:11]], np.int64), imp

    def T(a):
        a = np.ascontiguousarray(a)
        tt = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
        return torch.frombuffer(bytearray(a.tobytes()), dtype=tt[a.dtype]).reshape(a.shape)

    # squad rosters (norm surname -> for robust team detection; lineups.json labels can be swapped)
    rosters = {}
    for f in (WC / "teams").glob("*.json"):
        d = json.load(open(f, encoding="utf-8"))
        rosters[d["team"]["code"]] = set(db.norm(p["name"]).split()[-1] for p in d["players"])

    def detect(xi, a, b):
        """Which of codes a/b does this XI belong to, by surname overlap with the squad rosters."""
        sn = [db.norm(p.get("full", "")).split()[-1] for p in xi if p.get("full")]
        na = sum(s in rosters.get(a, ()) for s in sn); nb = sum(s in rosters.get(b, ()) for s in sn)
        return a if na >= nb else b

    # --lineups PATH overrides the lineup source (used for fallback / previous-game XIs)
    lp = WC / "lineups.json"
    for i, a in enumerate(sys.argv):
        if a == "--lineups" and i + 1 < len(sys.argv): lp = Path(sys.argv[i + 1])
        elif a.startswith("--lineups="): lp = Path(a.split("=", 1)[1])
    L = json.load(open(lp, encoding="utf-8"))
    Rz = json.load(open(WC / "results.json", encoding="utf-8"))

    def team_prev_xi(code):
        """That team's XI from its most recent FINISHED game in lineups.json (roster-attributed), or None.
        Lets us predict UPCOMING fixtures (not yet in lineups.json) from each side's last lineup."""
        cands = []
        for k, e in L.items():
            fin = e.get("state") == "finished" or (Rz.get(k) or {}).get("status") == "finished"
            if not fin or code not in k.split("-"):
                continue
            hx, ax = e.get("home_xi") or [], e.get("away_xi") or []
            if len(hx) < 11 or len(ax) < 11:
                continue
            cands.append(((Rz.get(k) or {}).get("kickoff", 0), hx, ax))
        if not cands:
            return None
        cands.sort(key=lambda t: t[0], reverse=True)
        _, hx, ax = cands[0]; rs = rosters.get(code, set())
        ov = lambda xi: sum(1 for p in xi if db.norm(p.get("full", "")).split()[-1:] and db.norm(p.get("full", "")).split()[-1] in rs)
        return hx if ov(hx) >= ov(ax) else ax

    for key in keys:
        gg = L.get(key); fallback = False
        if not gg:
            ca0, cb0 = key.split("-")
            hx, ax = team_prev_xi(ca0), team_prev_xi(cb0)
            if hx and ax:
                gg = {"home_xi": hx, "away_xi": ax}; fallback = True
            else:
                miss = [c for c, x in ((ca0, hx), (cb0, ax)) if not x]
                print(f"\n{key}: not in lineups.json and no previous XI for {', '.join(miss)} — can't predict yet")
                continue
        ca0, cb0 = key.split("-")
        hc = detect(gg.get("home_xi", []), ca0, cb0)     # actual team of home_xi (label-swap safe)
        ac = cb0 if hc == ca0 else ca0
        if hc != ca0:
            print(f"  (note: {key} home/away XI were swapped in source data — corrected)")
        Xh, Rh, i1 = side(gg.get("home_xi", [])); Xa, Ra, i2 = side(gg.get("away_xi", []))
        ctx = tg.ctx_vec(natctx.get(teams.get(hc), (tg.BASE, 1, 0)), natctx.get(teams.get(ac), (tg.BASE, 1, 0)))
        Xh = ((Xh - mu) / sd).astype(np.float32); Xa = ((Xa - mu) / sd).astype(np.float32)
        ctxn = ((ctx - cmu) / csd).astype(np.float32)
        grids = []
        with torch.no_grad():
            for nt in nets:
                lh, la = nt(T(Xh[None]), T(Rh[None]), T(Xa[None]), T(Ra[None]), T(ctxn[None]))
                grids.append(tg.score_matrix(math.exp(float(lh[0])), math.exp(float(la[0])), rho))
        P = np.mean(grids, 0); P = P / P.sum()         # ensemble = average score grids across seeds
        lhh = float((P.sum(1) * np.arange(P.shape[0])).sum())   # display xG = grid marginal means
        laa = float((P.sum(0) * np.arange(P.shape[1])).sum())
        ho = tg.hda_from_P(P); pk = pick_strategy(P, STRATEGY, BETA, QFIELD)
        flat = sorted(((P[i, j], i, j) for i in range(tg.MAXG + 1) for j in range(tg.MAXG + 1)), reverse=True)
        res = Rz.get(key, {})
        fbnote = "  [FALLBACK: previous-game XIs, not the confirmed lineup]" if fallback else ""
        print(f"\n=== {hc} (home) vs {ac} (away)  [status={res.get('status','?')}, imputed {i1+i2}/22] ==={ROUNDINFO}{fbnote}")
        print(f"  xG: {hc} {lhh:.2f} - {laa:.2f} {ac}   |   {hc} win {ho[0]*100:.0f}%  draw {ho[1]*100:.0f}%  {ac} win {ho[2]*100:.0f}%")
        print(f"  {STRATEGY} pick: {hc} {pk[0]}-{pk[1]} {ac}   top: " +
              "  ".join(f"{i}-{j} {p*100:.0f}%" for p, i, j in flat[:5]))
        if res.get("status") == "finished":
            print(f"  ACTUAL: {hc} {res['hs']}-{res['as']} {ac}")


if __name__ == "__main__":
    main()
