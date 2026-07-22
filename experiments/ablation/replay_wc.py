"""Phase-6 walk-forward WC2026 day-by-day replay + candidate comparison.

Replays the 104-game slate matchday-by-matchday (grouped by kickoff date). Every prediction for
matchday D uses only pre-D information, so the walk-forward is leakage-free by construction:
  * the core model is trained on data ending 2026-06-14 (pre-tournament) via run_ablation.train_one;
  * the WC-slate context (wc_inputs.npz) is the frozen pre-tournament snapshot;
  * the FROZEN arm predicts every matchday with those pre-WC weights (== the registry wc_slate eval,
    reproduced here as the driver-correctness tripwire);
  * the FINETUNE arm, after each matchday, fine-tunes on the WC games played STRICTLY earlier, then
    predicts the current matchday (in-tournament learning; the only new information is past results).
Context is NOT recomputed mid-tournament: Phase-3 showed Elo-momentum/trajectory is null over the base
Elo-LEVEL context, and a mid-tournament Elo recompute is the largest leakage surface for ~0 expected
gain; the fine-tune arm carries the in-tournament signal instead. (Documented scope, not an oversight.)

Market layer: candidates optionally carry the ctx-odds FEATURE (trained with data/ctx_odds.npz, and
the WC slate gets data/wc_odds.npz appended — 100% slate coverage) and/or a post-hoc λ-blend of the
score grids toward the de-vigged market (blend_market.rescale). λ is the Phase-4 pre-registered 0.5
(tuned on the non-WC odds-covered earlystop subset) — never tuned on the WC games we score here.

Candidates (× {frozen, finetune}) vs the production-recipe reference (baseline-beta3-w15 wc_slate row,
the honest out-of-tournament production bar on the identical slate; the full-data goalnet.pt cutover
validation is Phase-6 Step 5). Emits one registry row per (candidate, mode) on lane `wc_replay`,
per-matchday cumulative CSVs under diagnostics/, and prints the selection table.

Usage:
  python experiments/ablation/replay_wc.py --seeds 3                # full run
  python experiments/ablation/replay_wc.py --seeds 1 --only core    # smoke
  python experiments/ablation/replay_wc.py --report                 # reprint tables from registry
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

warnings.filterwarnings("ignore")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import metrics  # noqa: E402
import splits  # noqa: E402
import run_ablation as RA  # noqa: E402
import blend_market as BM  # noqa: E402
import train_goals as tg  # noqa: E402

ROOT = RA.ROOT
AB = RA.AB
REG = AB / "registry.jsonl"
DIAG = AB / "diagnostics"
REPLAY_LANE = "wc_replay"
PROD_REF = "baseline-beta3-w15"          # production-recipe reference (registry wc_slate row)
BLEND_LAMBDA = 0.5                        # Phase-4 pre-registered (tuned off-WC); never tuned here

# Candidate set (frozen config, Phase-5→6 handoff). Each runs in both frozen and finetune modes.
CANDIDATES = [
    {"name": "core", "ctx_extra": [], "blend": False},
    {"name": "core-oddsfeat", "ctx_extra": ["ctx_odds.npz"], "blend": False},
    {"name": "core-blend", "ctx_extra": [], "blend": True},
    {"name": "core-oddsfeat-blend", "ctx_extra": ["ctx_odds.npz"], "blend": True},
]

# fine-tune hyperparameters: a deliberately LIGHT touch. Fine-tuning on the ≤104-game slate alone
# catastrophically forgets the 69k-match pretraining if pushed hard (an 8-epoch/matchday probe
# collapsed grid_info 0.27→0.01), so keep it gentle — a few low-LR steps, and only once enough WC
# games have been played that an update is not just fitting 3 games. This is realism-honest (a light
# in-tournament nudge), not tuned to flatter the arm; a neutral/negative verdict is a valid finding.
FT_LR, FT_EPOCHS, FT_WD, FT_MIN_GAMES = 3e-4, 3, 1e-4, 8
# L2-SP (L2-to-init) anti-forgetting variant: penalise λ·‖θ−θ_pretrained‖² during fine-tune so the
# update cannot drift far from the 69k-match base (a small-model alternative to LoRA that targets the
# forgetting directly — no adapter path in the model zoo). λ NOT tuned on the WC slate.
FT_L2SP = 1e-2
MODES = ("frozen", "finetune", "finetune_l2sp")


def build_wc_context(D, ctx_extra):
    """WC-slate tensors standardised with this run's TRAIN stats. Base 10-dim context comes from
    wc_inputs.npz; ctx-odds feature candidates get data/wc_odds.npz (5 dims) appended, aligned to the
    slate keys — mirroring how run_ablation appends ctx_odds.npz to the training context."""
    w = splits.build_wc_inputs()
    keys = [str(k) for k in w["keys"]]
    ctx = w["ctx"].astype(np.float32)
    for name in ctx_extra:
        if name == "ctx_odds.npz":
            wz = np.load(ROOT / "data" / "wc_odds.npz", allow_pickle=True)
            wkeys = [str(k) for k in wz["keys"]]; wfeat = wz["feats"]
            kmap = {k: wfeat[i] for i, k in enumerate(wkeys)}
            edim = wfeat.shape[1]
            EX = np.stack([kmap.get(k, np.zeros(edim, np.float32)) for k in keys]).astype(np.float32)
        else:
            raise ValueError(f"no WC bundle for ctx_extra {name}")
        ctx = np.concatenate([ctx, EX], 1)
    assert ctx.shape[1] == D["nctx"], f"WC ctx dim {ctx.shape[1]} != train nctx {D['nctx']}"
    wXhn = ((w["Xh"] - D["mu"]) / D["sd"]).astype(np.float32)
    wXan = ((w["Xa"] - D["mu"]) / D["sd"]).astype(np.float32)
    wCTXn = ((ctx - D["cmu"]) / D["csd"]).astype(np.float32)
    whg, wag = w["hs"].astype(np.float32), w["as_"].astype(np.float32)
    wy = np.where(whg > wag, 0, np.where(whg == wag, 1, 2))
    return {"keys": keys, "Xhn": wXhn, "Rh": w["Rh"].astype(np.int64), "Xan": wXan,
            "Ra": w["Ra"].astype(np.int64), "CTXn": wCTXn, "hg": whg, "ag": wag, "y": wy,
            "kickoff": w["kickoff"].astype(np.int64)}


def wc_market(keys):
    """De-vigged (pH,pD,pA) per slate key + coverage mask, aligned to `keys`."""
    wz = np.load(ROOT / "data" / "wc_odds.npz", allow_pickle=True)
    wkeys = [str(k) for k in wz["keys"]]; feat = wz["feats"]
    kmap = {k: feat[i] for i, k in enumerate(wkeys)}
    mkt = np.array([kmap.get(k, np.zeros(5, np.float32))[:3] for k in keys], np.float32)
    cov = np.array([k in kmap and kmap[k][4] > 0 for k in keys])
    return mkt, cov


def matchdays(kickoff):
    """Return ordered list of (day_index, boolean mask over games) grouped by kickoff calendar date."""
    days = (kickoff // 86400)                      # UTC day bucket; monotone in kickoff
    order = sorted(set(days.tolist()))
    return [(i, days == d) for i, d in enumerate(order)]


def infer_net(net, WC):
    return RA.infer(net, WC["Xhn"], WC["Rh"], WC["Xan"], WC["Ra"], WC["CTXn"])


def finetune_step(net, tens, beta, theta0=None, l2sp=0.0):
    """A few gradient steps on the cumulative finished-WC tensor dict (Poisson NLL, core beta).
    With l2sp>0, add λ·‖θ−θ0‖² over the trainable params (θ0 = pretrained snapshot) to bound drift."""
    if tens["Xh"].size(0) == 0:
        return
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, weight_decay=FT_WD)
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
    net.train()
    for _ in range(FT_EPOCHS):
        opt.zero_grad()
        lh, la = net(tens["Xh"], tens["Rh"], tens["Xa"], tens["Ra"], tens["C"])
        loss = (pois(lh, tens["hg"]) + pois(la, tens["ag"])).mean()
        if beta:
            loss = loss - beta * RA.exp_points(torch.exp(lh), torch.exp(la), tens["hg"], tens["ag"]).mean()
        if l2sp:
            loss = loss + l2sp * sum(((p - theta0[k]) ** 2).sum()
                                     for k, p in net.named_parameters() if k in theta0)
        loss.backward(); opt.step()
    net.eval()


def walk_finetune(base_state, WC, mds, D, beta, n, l2sp=0.0):
    """Walk-forward fine-tune from base_state: before each matchday, fine-tune on the WC games played
    strictly earlier (once ≥FT_MIN_GAMES), then record that matchday's predictions. Returns (lh,la)."""
    net = RA.models.build_model("goalnet", D["A"], D["nctx"])
    net.load_state_dict({k: v.clone() for k, v in base_state.items()}); net.eval()
    theta0 = {k: v.detach().clone() for k, v in net.named_parameters()} if l2sp else None
    lh_full, la_full = np.zeros(n, np.float32), np.zeros(n, np.float32)
    played = np.zeros(n, bool)
    for _, mask in mds:
        if played.sum() >= FT_MIN_GAMES:
            finetune_step(net, wc_tensor(WC, played), beta, theta0, l2sp)
        lh, la = infer_net(net, WC)                            # predict all; keep only this day's slots
        lh_full[mask], la_full[mask] = lh[mask], la[mask]
        played |= mask
    return lh_full, la_full


def wc_tensor(WC, mask):
    """Torch tensor dict for the WC games under `mask` (fine-tune training data)."""
    idx = np.where(mask)[0]
    return {"Xh": RA.T(WC["Xhn"][idx]), "Rh": RA.T(WC["Rh"][idx]), "Xa": RA.T(WC["Xan"][idx]),
            "Ra": RA.T(WC["Ra"][idx]), "C": RA.T(WC["CTXn"][idx]),
            "hg": RA.T(WC["hg"][idx]), "ag": RA.T(WC["ag"][idx])}


def replay_one(cand, seeds, epochs, modes):
    """Train `seeds` pre-WC models for this candidate ONCE; from each base model produce every
    requested mode's per-matchday grids (frozen / finetune / finetune_l2sp all reuse the same base
    weights, so the mode comparison is on identical models). Returns per-mode seed-averaged grids."""
    beta, W = 0.0, 1.0
    D = RA.load_data("players_imp.npz", "pooled", None, cand["ctx_extra"], None)
    TR, ES = RA.make_split_tensors(D, W)
    WC = build_wc_context(D, cand["ctx_extra"])
    mds = matchdays(WC["kickoff"])
    n = len(WC["keys"])

    seed_rates = {m: [] for m in modes}
    es_rates, seed_rps = [], []
    for s in range(seeds):
        net, brps, ep = RA.train_one(s, D, TR, ES, beta, epochs, arch="goalnet")
        seed_rps.append(brps)
        es_rates.append(RA.infer(net, D["Xhn"][D["es"]], D["Rh"][D["es"]],
                                 D["Xan"][D["es"]], D["Ra"][D["es"]], D["CTXn"][D["es"]]))
        base_state = {k: v.clone() for k, v in net.state_dict().items()}
        if "frozen" in modes:                                   # weights fixed: one slate inference
            seed_rates["frozen"].append(infer_net(net, WC))
        if "finetune" in modes:                                 # light walk-forward fine-tune
            seed_rates["finetune"].append(walk_finetune(base_state, WC, mds, D, beta, n, l2sp=0.0))
        if "finetune_l2sp" in modes:                            # L2-SP anti-forgetting fine-tune
            seed_rates["finetune_l2sp"].append(walk_finetune(base_state, WC, mds, D, beta, n, l2sp=FT_L2SP))
        print(f"    [{cand['name']}] seed {s}: earlystop rps={brps:.4f} (e={ep})", flush=True)

    # DC rho tuned on the earlystop lane by league points (production convention) — off the WC slate
    rho = max(RA.RHOS, key=lambda r: RA.points_of(
        RA.grids_from(es_rates, r), D["hg"][D["es"]], D["ag"][D["es"]]))
    prior = metrics.empirical_prior(D["hg"][D["tr"]], D["ag"][D["tr"]])
    out = {"keys": WC["keys"], "hg": WC["hg"], "ag": WC["ag"], "y": WC["y"], "rho": rho,
           "prior": prior, "kickoff": WC["kickoff"], "mds": mds, "seed_rps": seed_rps}
    for m in modes:
        out[m] = RA.grids_from(seed_rates[m], rho)
    return out


def apply_blend(grids, keys):
    mkt, cov = wc_market(keys)
    mm = BM.outcome_masses(grids)
    target = mm.copy()
    target[cov] = (1 - BLEND_LAMBDA) * mm[cov] + BLEND_LAMBDA * mkt[cov]
    return BM.rescale(grids, target)


def cumulative_curve(grids, WC, mds, prior):
    """Per-matchday cumulative metric rows (grid_info, rps, acc, ece, pts/g)."""
    rows, seen = [], np.zeros(len(WC["y"]), bool)
    for di, mask in mds:
        seen |= mask
        s = metrics.suite(grids[seen], WC["y"][seen], WC["hg"][seen], WC["ag"][seen], prior)
        day = int(WC["kickoff"][mask].min() // 86400)
        rows.append({"matchday": di, "day": day, "games_cum": int(seen.sum()),
                     "grid_info": s["grid_info"], "rps": s["rps"], "acc": s["acc"],
                     "ece": s["ece_outcome"], "pts_g": s["pts_g_31"]})
    return rows


def score_and_register(cand, mode, grids, R, seeds, t0):
    """Score the full slate + write registry row + per-matchday CSV."""
    keys, hg, ag, y, prior = R["keys"], R["hg"], R["ag"], R["y"], R["prior"]
    if cand["blend"]:
        grids = apply_blend(grids, keys)
    s = metrics.suite(grids, y, hg, ag, prior)
    curve = cumulative_curve(grids, {"y": y, "hg": hg, "ag": ag, "kickoff": R["kickoff"]}, R["mds"], prior)
    DIAG.mkdir(exist_ok=True)
    name = f"replay-{cand['name']}-{mode}"
    import csv as _csv
    with open(DIAG / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(curve[0].keys())); w.writeheader(); w.writerows(curve)
    commit, dirty = RA.git_info()
    row = {"name": name, "ts": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "dirty": dirty,
           "config": {"npz": "players_imp.npz", "split": "pooled", "beta": 0.0, "W": 1.0,
                      "seeds": seeds, "epochs": 0, "rho_policy": f"val-tuned:{R['rho']}",
                      "ctx_extra": cand["ctx_extra"], "decay_halflife": None,
                      "flags": {"replay": mode, "blend": (BLEND_LAMBDA if cand["blend"] else None),
                                "candidate": cand["name"]},
                      "notes": f"Phase-6 WC replay: {cand['name']} / {mode}"},
           "data": {"n": len(y), "ctx_dim": None, "seed_earlystop_rps": [round(r, 4) for r in R["seed_rps"]]},
           "metrics": {REPLAY_LANE: s}, "wall_min": round((time.time() - t0) / 60.0, 2)}
    with open(REG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return name, s


def print_table(results, prod_s):
    hdr = f"{'candidate/mode':28s} {'n':>3s} {'grid_info':>10s} {'rps':>7s} {'acc':>6s} {'ece':>6s} {'pts/g':>6s}"
    print("\n=== WC2026 replay — selection table (lane wc_replay) ===", flush=True)
    print(hdr, flush=True)
    if prod_s:
        print(f"{'PROD-REF baseline-b3-w15':28s} {prod_s['n']:>3d} {prod_s['grid_info']:>+10.4f} "
              f"{prod_s['rps']:>7.4f} {prod_s['acc']:>6.3f} {prod_s['ece_outcome']:>6.3f} {prod_s['pts_g_31']:>6.3f}", flush=True)
    for name, s in results:
        di = f"{s['grid_info']:+.4f}"
        if prod_s:
            di += f" ({s['grid_info']-prod_s['grid_info']:+.4f})"
        print(f"{name:28s} {s['n']:>3d} {di:>18s} {s['rps']:>7.4f} {s['acc']:>6.3f} "
              f"{s['ece_outcome']:>6.3f} {s['pts_g_31']:>6.3f}", flush=True)


def prod_reference():
    """Production-recipe wc_slate metrics from the registry (baseline-beta3-w15)."""
    for line in reversed(REG.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            r = json.loads(line)
            if r["name"] == PROD_REF and "wc_slate" in r.get("metrics", {}):
                return r["metrics"]["wc_slate"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--only", default=None, help="comma-separated candidate names to run")
    ap.add_argument("--modes", default=",".join(MODES),
                    help=f"comma-separated replay modes to run (default: {','.join(MODES)})")
    ap.add_argument("--report", action="store_true", help="reprint the table from existing registry rows")
    args = ap.parse_args()
    prod_s = prod_reference()
    modes = [m for m in MODES if m in set(args.modes.split(","))]

    if args.report:
        results = []
        rows = {json.loads(l)["name"]: json.loads(l) for l in REG.read_text(encoding="utf-8").splitlines() if l.strip()}
        for cand in CANDIDATES:
            for mode in MODES:
                nm = f"replay-{cand['name']}-{mode}"
                if nm in rows and REPLAY_LANE in rows[nm]["metrics"]:
                    results.append((nm, rows[nm]["metrics"][REPLAY_LANE]))
        print_table(results, prod_s)
        return

    only = set(args.only.split(",")) if args.only else None
    cands = [c for c in CANDIDATES if not only or c["name"] in only]
    results = []
    for cand in cands:
        t0 = time.time()
        print(f"\n--- candidate {cand['name']} (ctx_extra={cand['ctx_extra']}, blend={cand['blend']}) ---", flush=True)
        R = replay_one(cand, args.seeds, args.epochs, modes)
        for mode in modes:
            name, s = score_and_register(cand, mode, R[mode], R, args.seeds, t0)
            results.append((name, s))
            print(f"  {name}: grid_info={s['grid_info']:+.4f} rps={s['rps']:.4f} acc={s['acc']:.3f} "
                  f"ece={s['ece_outcome']:.3f} pts/g={s['pts_g_31']:.3f}", flush=True)
    print_table(results, prod_s)
    RA.regen_report()


if __name__ == "__main__":
    main()
