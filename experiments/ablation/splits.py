"""Split/eval-set definitions + frozen WC2026-slate inputs (contract in DESIGN.md).

Splits (masks over a players npz):
  canonical: train <2024-08 | earlystop = val [2024-08,2025-08) | eval = test >=2025-08
  pooled   : train <2024-08 | earlystop = last 10% of train by date (still inside train)
             | eval = [2024-08, 2026-06-11)   (val+test pooled, pre-WC cutoff)
WC slate: experiments/ablation/wc_inputs.npz — RAW (unstandardized) features + truth for all 104
finished WC2026 games, built once from worldcup team_db + FM snapshots (edition-fallback, same
logic as eval_harness.build_wc_cache but saving inputs, not rates). Context = pre-tournament
national Elo/form (DB holds no WC2026 matches — verified; walk-forward context is Phase-6 replay).

Usage: python experiments/ablation/splits.py --report   (builds wc_inputs if missing, checks leakage)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db  # noqa: E402
import train_goals as tg  # noqa: E402

WC = Path(r"D:\Programming\claude\worldcup\team_db")
WC_INPUTS = ROOT / "experiments" / "ablation" / "wc_inputs.npz"
NATc = tg.NATc
TR_END = np.datetime64("2024-08-01")
VA_END = np.datetime64("2025-08-01")
WC_START = np.datetime64("2026-06-11")


def load_dataset(npz="players_imp.npz"):
    z = np.load(ROOT / "data" / npz, allow_pickle=True)
    out = {k: z[k] for k in ("Xh", "Xa", "dates")}
    out["Rh"] = z["Rh"].astype(np.int64); out["Ra"] = z["Ra"].astype(np.int64)
    out["y"] = z["y"].astype(np.int64); out["mids"] = [int(m) for m in z["mids"]]
    out["attrs"] = [str(a) for a in z["attrs"]]
    cz = np.load(ROOT / "data" / "context.npz")
    ctx_arr = cz["ctx"]; ctx_mids = cz["mids"]        # materialize once (NpzFile is lazy)
    cmap = {int(m): ctx_arr[i] for i, m in enumerate(ctx_mids)}
    nctx = ctx_arr.shape[1]
    out["CTX"] = np.stack([cmap.get(m, np.zeros(nctx, np.float32)) for m in out["mids"]]).astype(np.float32)
    con = db.connect()
    meta = {r[0]: (r[1], r[2], r[3]) for r in
            con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    con.close()
    out["natl"] = np.array([meta.get(m, (0, 0, 0))[0] in NATc for m in out["mids"]])
    out["hg"] = np.array([min(meta.get(m, (0, 0, 0))[1] or 0, tg.MAXG) for m in out["mids"]], np.float32)
    out["ag"] = np.array([min(meta.get(m, (0, 0, 0))[2] or 0, tg.MAXG) for m in out["mids"]], np.float32)
    return out


def get_masks(dates, split="pooled"):
    tr = dates < TR_END
    if split == "canonical":
        es = (dates >= TR_END) & (dates < VA_END)          # early-stop on val (historical recipe)
        ev = dates >= VA_END                               # test
    elif split == "pooled":
        trd = np.sort(dates[tr])
        cut = trd[int(len(trd) * 0.9)]                     # last 10% of train (by date) = earlystop
        es = tr & (dates >= cut)
        tr = tr & (dates < cut)
        ev = (dates >= TR_END) & (dates < WC_START)        # val+test pooled, pre-WC
    else:
        raise ValueError(split)
    # leakage assertions (DESIGN.md)
    assert not (tr & es).any() and not (tr & ev).any() and not (es & ev).any()
    assert dates[ev].min() >= TR_END or split == "canonical"
    assert dates[ev].max() < WC_START or split == "canonical"
    return {"train": tr, "earlystop": es, "eval": ev}


def build_wc_inputs(force=False):
    """Freeze raw inputs for all finished WC2026 games (run once; content is final — tournament over)."""
    if WC_INPUTS.exists() and not force:
        return dict(np.load(WC_INPUTS, allow_pickle=True))
    con = db.connect()
    assert con.execute("SELECT COUNT(*) FROM match WHERE competition_id IN (9,10,11,12,13,14,15) "
                       "AND match_date >= '2026-06-11'").fetchone()[0] == 0, \
        "DB now contains WC2026 matches — pre-tournament context assumption broken, revisit DESIGN.md"
    natctx = tg.national_context(con)
    name2cid = {r[1]: r[0] for r in con.execute("SELECT club_id,name FROM club")}
    teams, rosters = {}, {}
    for f in (WC / "teams").glob("*.json"):
        d = json.load(open(f, encoding="utf-8")); t = d["team"]
        teams[t["code"]] = name2cid.get(tg.NAME_FIX.get(t["name"], t["name"]))
        rosters[t["code"]] = set(db.norm(p["name"]).split()[-1] for p in d["players"])
    EDRANK = {3: 10, 4: 9, 1: 8, 5: 7, 10: 6, 2: 5, 6: 4, 7: 3, 8: 2, 9: 1}
    snap = {}
    for sid, fmv, ca, nm in con.execute("SELECT s.snapshot_id,s.fm_version_id,s.ca,p.norm_name "
                                        "FROM player_snapshot s JOIN player p ON p.player_id=s.player_id"):
        r = EDRANK.get(fmv, 0); cur = snap.get(nm)
        if cur is None or r > cur[1] or (r == cur[1] and (ca or 0) > cur[2]):
            snap[nm] = (sid, r, ca or 0)
    chosen = set(v[0] for v in snap.values()); ab = defaultdict(dict)
    for sid, name, val in con.execute("SELECT snapshot_id,attr_name,attr_value FROM player_attribute"):
        if sid in chosen:
            ab[sid][name] = val
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    ATTRS = [str(a) for a in z["attrs"]]; A = len(ATTRS); aidx = {n: i for i, n in enumerate(ATTRS)}
    # role means from the canonical train split (imputation vectors; same convention as production)
    tr = z["dates"] < TR_END
    Rh_ = z["Rh"].astype(np.int64)
    Xh_tr = z["Xh"][tr]; Rh_tr = Rh_[tr]              # materialize once (NpzFile is lazy)
    role_mean = {r: Xh_tr[Rh_tr == r].mean(0) for r in range(4)}

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

    def detect(xi, a, b):
        sn = [db.norm(p.get("full", "")).split()[-1] for p in xi if p.get("full")]
        na = sum(s in rosters.get(a, ()) for s in sn); nb = sum(s in rosters.get(b, ()) for s in sn)
        return a if na >= nb else b

    L = json.load(open(WC / "lineups.json", encoding="utf-8"))
    Rz = json.load(open(WC / "results.json", encoding="utf-8"))
    rows = []
    for key, res in sorted(Rz.items(), key=lambda kv: kv[1].get("kickoff") or 0):
        if res.get("status") != "finished" or key not in L:
            continue
        gg = L[key]
        ca0, cb0 = key.split("-"); hc = detect(gg.get("home_xi", []), ca0, cb0); ac = cb0 if hc == ca0 else ca0
        Xh1, Rh1, i1 = side(gg.get("home_xi", [])); Xa1, Ra1, i2 = side(gg.get("away_xi", []))
        ctx = tg.ctx_vec(natctx.get(teams.get(hc), (tg.BASE, 1, 0)),
                         natctx.get(teams.get(ac), (tg.BASE, 1, 0)))
        rows.append((f"{hc}-{ac}", Xh1, Rh1, Xa1, Ra1, ctx.astype(np.float32),
                     min(int(res["hs"]), tg.MAXG), min(int(res["as"]), tg.MAXG),
                     int(res.get("kickoff") or 0), i1 + i2))
    con.close()
    out = {"keys": np.array([r[0] for r in rows]),
           "Xh": np.stack([r[1] for r in rows]), "Rh": np.stack([r[2] for r in rows]),
           "Xa": np.stack([r[3] for r in rows]), "Ra": np.stack([r[4] for r in rows]),
           "ctx": np.stack([r[5] for r in rows]),
           "hs": np.array([r[6] for r in rows]), "as_": np.array([r[7] for r in rows]),
           "kickoff": np.array([r[8] for r in rows]), "imputed": np.array([r[9] for r in rows])}
    np.savez_compressed(WC_INPUTS, **out)
    print(f"  built {WC_INPUTS.name}: {len(rows)} games, "
          f"{out['imputed'].sum()}/{len(rows)*22} imputed starters", flush=True)
    return out


def report():
    d = load_dataset()
    print(f"dataset players_imp.npz: n={len(d['y']):,}  natl={int(d['natl'].sum()):,}")
    for split in ("canonical", "pooled"):
        m = get_masks(d["dates"], split)
        ev = m["eval"]
        print(f"  {split:9s} train={int(m['train'].sum()):6,} earlystop={int(m['earlystop'].sum()):5,} "
              f"eval={int(ev.sum()):6,} (natl {int((ev & d['natl']).sum()):3d}) "
              f"eval dates {d['dates'][ev].min()} .. {d['dates'][ev].max()}")
    w = build_wc_inputs()
    exp = json.load(open(WC / "results.json", encoding="utf-8"))
    nfin = sum(1 for v in exp.values() if v.get("status") == "finished")
    assert len(w["keys"]) == nfin, f"wc slate {len(w['keys'])} != finished {nfin}"
    pm = get_masks(d["dates"], "pooled")
    assert int((pm["eval"] & d["natl"]).sum()) >= 350, "pooled natl lane too small"
    print(f"  wc_slate: n={len(w['keys'])} (== {nfin} finished), "
          f"imputed {int(np.asarray(w['imputed']).sum())}/{len(w['keys'])*22} starters")
    print("LEAKAGE CHECKS PASS")


if __name__ == "__main__":
    if "--rebuild-wc" in sys.argv:
        build_wc_inputs(force=True)
    if "--report" in sys.argv or len(sys.argv) == 1:
        report()
