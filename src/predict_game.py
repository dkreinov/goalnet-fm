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


def main():
    keys = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not keys:
        print("usage: predict_game.py NED-SWE [KEY ...]"); return
    ck = ROOT / "data" / "goalnet.pt"
    if not ck.exists():
        print("no data/goalnet.pt — run: python src/train_goals.py --full  (once)"); return
    c = torch.load(ck, weights_only=False)
    A, nctx = c["A"], c["nctx"]
    mu, sd, cmu, csd, rho = c["mu"], c["sd"], c["cmu"], c["csd"], c["rho"]
    ATTRS = c["attrs"]; aidx = {n: i for i, n in enumerate(ATTRS)}; role_mean = c["role_mean"]
    net = tg.GoalNet(A, nctx); net.load_state_dict(c["state"]); net.eval()

    con = db.connect()
    natctx = tg.national_context(con)
    name2cid = {r[1]: r[0] for r in con.execute("SELECT club_id,name FROM club")}
    teams = {}
    for f in (WC / "teams").glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))["team"]
        teams[t["code"]] = name2cid.get(tg.NAME_FIX.get(t["name"], t["name"]))
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
        ps = [(tg.pos_role(p.get("pos")), vec_for(p.get("full", ""))) for p in xi]
        imp = sum(v is None for _, v in ps)
        ps = [(r, v if v is not None else role_mean[r]) for r, v in ps]
        ps.sort(key=lambda t: t[0]); ps = ps[:11] + [(2, role_mean[2])] * max(0, 11 - len(ps))
        return np.stack([v for _, v in ps[:11]]), np.array([r for r, _ in ps[:11]], np.int64), imp

    def T(a):
        a = np.ascontiguousarray(a)
        tt = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
        return torch.frombuffer(bytearray(a.tobytes()), dtype=tt[a.dtype]).reshape(a.shape)

    L = json.load(open(WC / "lineups.json", encoding="utf-8"))
    Rz = json.load(open(WC / "results.json", encoding="utf-8"))
    for key in keys:
        gg = L.get(key)
        if not gg:
            print(f"\n{key}: not in lineups.json"); continue
        hc, ac = key.split("-")
        Xh, Rh, i1 = side(gg.get("home_xi", [])); Xa, Ra, i2 = side(gg.get("away_xi", []))
        ctx = tg.ctx_vec(natctx.get(teams.get(hc), (tg.BASE, 1, 0)), natctx.get(teams.get(ac), (tg.BASE, 1, 0)))
        Xh = ((Xh - mu) / sd).astype(np.float32); Xa = ((Xa - mu) / sd).astype(np.float32)
        ctxn = ((ctx - cmu) / csd).astype(np.float32)
        with torch.no_grad():
            lh, la = net(T(Xh[None]), T(Rh[None]), T(Xa[None]), T(Ra[None]), T(ctxn[None]))
        lhh, laa = math.exp(float(lh[0])), math.exp(float(la[0]))
        P = tg.score_matrix(lhh, laa, rho); ho = tg.hda_from_P(P); pk = tg.ev_pick(P)
        flat = sorted(((P[i, j], i, j) for i in range(tg.MAXG + 1) for j in range(tg.MAXG + 1)), reverse=True)
        res = Rz.get(key, {})
        print(f"\n=== {hc} (home) vs {ac} (away)  [status={res.get('status','?')}, imputed {i1+i2}/22] ===")
        print(f"  xG: {hc} {lhh:.2f} - {laa:.2f} {ac}   |   {hc} win {ho[0]*100:.0f}%  draw {ho[1]*100:.0f}%  {ac} win {ho[2]*100:.0f}%")
        print(f"  EV pick: {hc} {pk[0]}-{pk[1]} {ac}   top: " +
              "  ".join(f"{i}-{j} {p*100:.0f}%" for p, i, j in flat[:5]))
        if res.get("status") == "finished":
            print(f"  ACTUAL: {hc} {res['hs']}-{res['as']} {ac}")


if __name__ == "__main__":
    main()
