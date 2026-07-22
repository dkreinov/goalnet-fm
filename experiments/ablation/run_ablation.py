"""One-command ablation runner (contract: experiments/ablation/DESIGN.md).

  python experiments/ablation/run_ablation.py --name <id> [--npz players_imp.npz]
        [--split pooled|canonical] [--beta 3] [--w 15] [--seeds 5] [--epochs 150]
        [--decay-halflife <years>] [--ctx-extra file.npz ...] [--notes "..."] [--force-rerun]
  python experiments/ablation/run_ablation.py --diagnose <name>   # Step-6 diagnostics section
  python experiments/ablation/run_ablation.py --report            # regenerate RESULTS_ABLATION.md

Trains `seeds` GoalNet models on the split's TRAIN mask (early-stop on the split's earlystop mask),
tunes one shared Dixon-Coles rho on the earlystop lane by league points (production convention),
seed-averages the per-match score grids, scores every lane with the frozen metric suite, appends one
row to registry.jsonl, caches per-seed rates, and regenerates the report. train_goals.py is imported
and never modified; the decision-focused term + tensor helpers below are replicated from its main()
(which defines them as locals) per the Phase-1 no-edit rule.
"""
import argparse
import json
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ablation"))
import train_goals as tg  # noqa: E402
import metrics  # noqa: E402
import splits  # noqa: E402

AB = ROOT / "experiments" / "ablation"
REG = AB / "registry.jsonl"
REPORT = AB / "RESULTS_ABLATION.md"
RATES = AB / "rates"
DIAG = AB / "diagnostics"
BASELINE_NAME = "baseline-beta3-w15"          # pooled reference row for Δ columns
RHOS = [-0.15, -0.1, -0.05, 0.0, 0.05]        # DC-rho grid (train_goals convention)

# ---- decision-focused term, replicated verbatim from train_goals.main() (do NOT edit train_goals) ----
GG, TAU = 7, 0.08
_ii = torch.arange(GG + 1); _I = _ii.view(GG + 1, 1).expand(GG + 1, GG + 1)
_J = _ii.view(1, GG + 1).expand(GG + 1, GG + 1)
_O = torch.where(_I > _J, 0, torch.where(_I == _J, 1, 2)); _lf = torch.lgamma(_ii.float() + 1)


def exp_points(lh, la, th, ta):
    ph = torch.exp(_ii.float().view(1, -1) * torch.log(lh.view(-1, 1).clamp(min=1e-6)) - lh.view(-1, 1) - _lf.view(1, -1))
    pa = torch.exp(_ii.float().view(1, -1) * torch.log(la.view(-1, 1).clamp(min=1e-6)) - la.view(-1, 1) - _lf.view(1, -1))
    P = ph.unsqueeze(2) * pa.unsqueeze(1); P = P / P.sum(dim=[1, 2], keepdim=True).clamp(min=1e-9)
    oprob = torch.stack([torch.tril(P, -1).sum([1, 2]), torch.diagonal(P, dim1=1, dim2=2).sum(1),
                         torch.triu(P, 1).sum([1, 2])], dim=1)
    EV = 2 * P + oprob[:, _O]
    pi = torch.softmax(EV.reshape(EV.size(0), -1) / TAU, dim=1).reshape_as(EV)
    th = th.clamp(max=GG).long(); ta = ta.clamp(max=GG).long()
    exact = (_I.unsqueeze(0) == th.view(-1, 1, 1)) & (_J.unsqueeze(0) == ta.view(-1, 1, 1))
    Otru = torch.where(th > ta, 0, torch.where(th == ta, 1, 2))
    omatch = (_O.unsqueeze(0) == Otru.view(-1, 1, 1))
    return (pi * (3.0 * exact.float() + 1.0 * (omatch & ~exact).float())).sum([1, 2])


# ---- tensor helpers, replicated from train_goals.main() ----
_TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}


def T(a):
    a = np.ascontiguousarray(a)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)


def tonp(t):
    return np.array(t.detach().tolist(), dtype=np.float32)


def _load_extra(name):
    """Load an extra-context npz (Phase-3 feature bundles). Returns (mid->vec dict, dim)."""
    z = np.load(ROOT / "data" / name)
    key = next((k for k in ("feats", "val", "ctx") if k in z.files),
               next(k for k in z.files if k != "mids"))
    arr, mids = z[key], z["mids"]
    return {int(m): arr[i] for i, m in enumerate(mids)}, arr.shape[1]


def load_data(npz, split, decay_halflife=None, ctx_extra=()):
    """Masks + train-standardised tensors + lane truth. Standardisation stats come from TRAIN only."""
    d = splits.load_dataset(npz)
    A = d["Xh"].shape[-1]
    CTX = d["CTX"]
    for name in ctx_extra:                                   # Phase-3 hook (inert unless --ctx-extra given)
        emap, edim = _load_extra(name)
        EX = np.stack([emap.get(m, np.zeros(edim, np.float32)) for m in d["mids"]]).astype(np.float32)
        CTX = np.concatenate([CTX, EX], 1)
    m = splits.get_masks(d["dates"], split)
    tr, es, ev = m["train"], m["earlystop"], m["eval"]
    mu = d["Xh"][tr].reshape(-1, A).mean(0); sd = d["Xh"][tr].reshape(-1, A).std(0) + 1e-6
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6
    Xhn = ((d["Xh"] - mu) / sd).astype(np.float32); Xan = ((d["Xa"] - mu) / sd).astype(np.float32)
    CTXn = ((CTX - cmu) / csd).astype(np.float32)
    # optional exponential time-decay sample weights (Phase-2 hook, off by default; W applied in train_one)
    if decay_halflife:
        age = (d["dates"].max() - d["dates"]) / np.timedelta64(365, "D")
        decay = 0.5 ** (age.astype(np.float32) / float(decay_halflife))
    else:
        decay = np.ones(len(d["mids"]), np.float32)
    return {
        "A": A, "nctx": CTX.shape[1], "mu": mu, "sd": sd, "cmu": cmu, "csd": csd,
        "Xhn": Xhn, "Xan": Xan, "CTXn": CTXn, "Rh": d["Rh"], "Ra": d["Ra"],
        "hg": d["hg"], "ag": d["ag"], "y": d["y"], "natl": d["natl"], "dates": d["dates"],
        "tr": tr, "es": es, "ev": ev, "decay": decay, "npz": npz,
    }


def hda_batch(lh, la, M=tg.MAXG + 1):
    """Vectorised H/D/A probabilities from Poisson rates — identical to
    hda_from_P(score_matrix(lh, la, rho=0)) but over all matches at once (early-stop hot path)."""
    k = np.arange(M)
    lf = np.array([math.lgamma(i + 1) for i in range(M)])
    ph = np.exp(k[None] * np.log(np.maximum(lh[:, None], 1e-6)) - lh[:, None] - lf[None])
    pa = np.exp(k[None] * np.log(np.maximum(la[:, None], 1e-6)) - la[:, None] - lf[None])
    ph /= ph.sum(1, keepdims=True); pa /= pa.sum(1, keepdims=True)   # joint outer is then normalised
    D = (ph * pa).sum(1)                                             # P(h == a)
    Pa_lt = np.concatenate([np.zeros((len(la), 1)), np.cumsum(pa, 1)[:, :-1]], 1)  # P(a < h)
    H = (ph * Pa_lt).sum(1)                                          # P(h > a)
    return np.stack([H, D, 1 - H - D], 1)


def infer(net, Xhn, Rh, Xan, Ra, CTXn, bs=4096):
    """Batched forward -> (lambda_home, lambda_away) as numpy (exp of log-rates)."""
    lhs, las = [], []
    with torch.no_grad():
        for i in range(0, len(Xhn), bs):
            lh, la = net(T(Xhn[i:i + bs]), T(Rh[i:i + bs]), T(Xan[i:i + bs]), T(Ra[i:i + bs]), T(CTXn[i:i + bs]))
            lhs.append(tonp(lh)); las.append(tonp(la))
    return np.exp(np.concatenate(lhs)), np.exp(np.concatenate(las))


def grids_from(lhla_list, rho):
    """Seed-average the per-match score grids at a given DC rho."""
    acc = None
    for lh, la in lhla_list:
        gs = np.stack([tg.score_matrix(a, b, rho) for a, b in zip(lh, la)])
        acc = gs if acc is None else acc + gs
    return acc / len(lhla_list)


def grids_from_arrays(lh2d, la2d, rho):
    """Seed-averaged grids from cached (seeds, n) rate arrays."""
    return grids_from([(lh2d[s], la2d[s]) for s in range(lh2d.shape[0])], rho)


def points_of(grids, hg, ag):
    tot = 0
    for g, H, Aa in zip(grids, hg, ag):
        tot += tg.grade(tg.ev_pick(g), int(H), int(Aa))[0]
    return tot


def make_split_tensors(D, W):
    """Build the seed-independent TRAIN + earlystop tensors ONCE (reused across seeds — the T()
    bytearray copy that NumPy-2/torch compat requires is expensive, so never rebuild per seed)."""
    tr, es = D["tr"], D["es"]
    TR = {"Xh": T(D["Xhn"][tr]), "Rh": T(D["Rh"][tr]), "Xa": T(D["Xan"][tr]), "Ra": T(D["Ra"][tr]),
          "C": T(D["CTXn"][tr]), "hg": T(D["hg"][tr]), "ag": T(D["ag"][tr]),
          "wt": T((np.where(D["natl"][tr], W, 1.0) * D["decay"][tr]).astype(np.float32))}
    ES = {"Xh": T(D["Xhn"][es]), "Rh": T(D["Rh"][es]), "Xa": T(D["Xan"][es]), "Ra": T(D["Ra"][es]),
          "C": T(D["CTXn"][es]), "y": D["y"][es]}
    return TR, ES


def train_one(seed, D, TR, ES, beta, epochs, patience=25):
    """Train one GoalNet on TRAIN (early-stop on earlystop lane by RPS). Mirrors train_goals.main."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xhtr, Rhtr, Xatr, Ratr, Ctr = TR["Xh"], TR["Rh"], TR["Xa"], TR["Ra"], TR["C"]
    hgtr, agtr, wt = TR["hg"], TR["ag"], TR["wt"]
    y_es = ES["y"]
    Vh, Vrh, Va_, Vra, Cv = ES["Xh"], ES["Rh"], ES["Xa"], ES["Ra"], ES["C"]
    net = tg.GoalNet(D["A"], D["nctx"])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
    bs, n = 512, Xhtr.size(0); best, bstate, bad, e = 9.0, None, 0, 0

    def val_rps():
        net.eval()
        with torch.no_grad():
            lh, la = net(Vh, Vrh, Va_, Vra, Cv)
        P = hda_batch(np.exp(tonp(lh)), np.exp(tonp(la)))
        return tg.rps(y_es, P)

    for e in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; opt.zero_grad()
            lh, la = net(Xhtr[b], Rhtr[b], Xatr[b], Ratr[b], Ctr[b])
            loss = ((pois(lh, hgtr[b]) + pois(la, agtr[b])) * wt[b]).mean()
            if beta:
                loss = loss - beta * (exp_points(torch.exp(lh), torch.exp(la), hgtr[b], agtr[b]) * wt[b]).mean()
            loss.backward(); opt.step()
        sched.step()
        r = val_rps()
        if r < best - 1e-4:
            best, bstate, bad = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break
    net.load_state_dict(bstate); net.eval()
    return net, best, e


def git_info():
    def q(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return q("rev-parse", "HEAD"), bool(q("status", "--porcelain"))


def run(args):
    if not args.force_rerun and REG.exists():
        for line in REG.read_text().splitlines():
            if line.strip() and json.loads(line)["name"] == args.name:
                sys.exit(f"refusing: name '{args.name}' already in registry (use --force-rerun to add a rerun row)")
    t0 = time.time()
    D = load_data(args.npz, args.split, args.decay_halflife, args.ctx_extra)
    print(f"data={args.npz} split={args.split} A={D['A']} nctx={D['nctx']} "
          f"train={int(D['tr'].sum()):,} earlystop={int(D['es'].sum()):,} eval={int(D['ev'].sum()):,}", flush=True)

    # WC slate (raw frozen inputs) -> standardise with this run's train stats. The frozen wc_inputs.npz
    # carries only the 10-dim BASE context; --ctx-extra feature bundles aren't available for the WC teams,
    # so the WC-slate lane is skipped for ctx-extra runs (same reason train_goals skips WC for --value/--venue).
    wc_ok = not args.ctx_extra
    if wc_ok:
        w = splits.build_wc_inputs()
        wXhn = ((w["Xh"] - D["mu"]) / D["sd"]).astype(np.float32); wXan = ((w["Xa"] - D["mu"]) / D["sd"]).astype(np.float32)
        wCTXn = ((w["ctx"] - D["cmu"]) / D["csd"]).astype(np.float32)
        wRh, wRa = w["Rh"].astype(np.int64), w["Ra"].astype(np.int64)
        whg, wag = w["hs"].astype(np.float32), w["as_"].astype(np.float32)
        wy = np.where(whg > wag, 0, np.where(whg == wag, 1, 2))
    else:
        print("  --ctx-extra set: WC-slate lane skipped (wc_inputs has base ctx only)", flush=True)

    ev = D["ev"]
    TR, ES = make_split_tensors(D, args.w)
    RATES.mkdir(exist_ok=True)
    es_lhla, ev_lhla, wc_lhla, seed_rps = [], [], [], []
    for s in range(args.seeds):
        sc = RATES / f"{args.name}.s{s}.npz"                 # per-seed checkpoint (resume across kills)
        if sc.exists():
            zc = np.load(sc)
            es_lhla.append((zc["es_lh"], zc["es_la"])); ev_lhla.append((zc["ev_lh"], zc["ev_la"]))
            if wc_ok:
                wc_lhla.append((zc["wc_lh"], zc["wc_la"]))
            seed_rps.append(float(zc["brps"]))
            print(f"  seed {s}: resumed from cache (earlystop rps={float(zc['brps']):.4f})", flush=True)
            continue
        net, brps, ep = train_one(s, D, TR, ES, args.beta, args.epochs)
        esr = infer(net, D["Xhn"][D["es"]], D["Rh"][D["es"]], D["Xan"][D["es"]], D["Ra"][D["es"]], D["CTXn"][D["es"]])
        evr = infer(net, D["Xhn"][ev], D["Rh"][ev], D["Xan"][ev], D["Ra"][ev], D["CTXn"][ev])
        cache = {"es_lh": esr[0], "es_la": esr[1], "ev_lh": evr[0], "ev_la": evr[1], "brps": brps}
        if wc_ok:
            wcr = infer(net, wXhn, wRh, wXan, wRa, wCTXn); wc_lhla.append(wcr)
            cache.update(wc_lh=wcr[0], wc_la=wcr[1])
        np.savez_compressed(sc, **cache)
        es_lhla.append(esr); ev_lhla.append(evr); seed_rps.append(brps)
        print(f"  seed {s}: earlystop rps={brps:.4f} (stopped e={ep}); cached", flush=True)

    # shared DC-rho tuned on earlystop lane by league points (production convention)
    best_rho = max(RHOS, key=lambda r: points_of(grids_from(es_lhla, r), D["hg"][D["es"]], D["ag"][D["es"]]))
    print(f"  DC rho={best_rho} (points-tuned on earlystop, {args.seeds}-seed avg)", flush=True)

    prior = metrics.empirical_prior(D["hg"][D["tr"]], D["ag"][D["tr"]])
    ev_grids = grids_from(ev_lhla, best_rho)
    natl_ev = D["natl"][ev]
    all_tag = "canonical_test" if args.split == "canonical" else "eval"
    lanes = {
        f"{all_tag}_all": (ev_grids, D["y"][ev], D["hg"][ev], D["ag"][ev]),
        f"{all_tag}_natl": (ev_grids[natl_ev], D["y"][ev][natl_ev], D["hg"][ev][natl_ev], D["ag"][ev][natl_ev]),
    }
    if wc_ok:
        wc_grids = grids_from(wc_lhla, best_rho)
        lanes["wc_slate"] = (wc_grids, wy, whg, wag)
    M = {lane: metrics.suite(g, y, hg, ag, prior) for lane, (g, y, hg, ag) in lanes.items()}
    for lane, s in M.items():
        print(f"  {lane:20s} n={s['n']:5d} grid_nll={s['grid_nll']:.4f} grid_info={s['grid_info']:+.4f} "
              f"rps={s['rps']:.4f} acc={s['acc']:.3f} ece={s['ece_outcome']:.3f} "
              f"exact_lift={s['exact_lift']:.2f} pts/g={s['pts_g_31']:.3f}", flush=True)

    # per-seed rates cache (diagnostics / pick-layer never retrain)
    RATES.mkdir(exist_ok=True)
    cache = {"rho": best_rho, "split": args.split,
             "ev_lh": np.stack([a for a, _ in ev_lhla]), "ev_la": np.stack([b for _, b in ev_lhla]),
             "ev_y": D["y"][ev], "ev_hg": D["hg"][ev], "ev_ag": D["ag"][ev], "ev_natl": natl_ev,
             "prior": prior, "tr_hg": D["hg"][D["tr"]], "tr_ag": D["ag"][D["tr"]]}
    if wc_ok:
        cache.update(wc_lh=np.stack([a for a, _ in wc_lhla]), wc_la=np.stack([b for _, b in wc_lhla]),
                     wc_y=wy, wc_hg=whg, wc_ag=wag, wc_keys=w["keys"])
    np.savez_compressed(RATES / f"{args.name}.npz", **cache)

    commit, dirty = git_info()
    npz_mtime = datetime.fromtimestamp((ROOT / "data" / args.npz).stat().st_mtime, timezone.utc).isoformat()
    row = {
        "name": args.name, "ts": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "dirty": dirty,
        "config": {"npz": args.npz, "split": args.split, "beta": args.beta, "W": args.w,
                   "seeds": args.seeds, "epochs": args.epochs, "rho_policy": f"val-tuned:{best_rho}",
                   "ctx_extra": list(args.ctx_extra), "decay_halflife": args.decay_halflife,
                   "flags": {}, "notes": args.notes},
        "data": {"npz_mtime": npz_mtime, "n": int(len(D["y"])), "ctx_dim": int(D["nctx"]),
                 "seed_earlystop_rps": [round(r, 4) for r in seed_rps]},
        "metrics": M, "wall_min": round((time.time() - t0) / 60.0, 2),
    }
    with open(REG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  appended registry row '{args.name}' ({row['wall_min']} min)", flush=True)
    regen_report()


# ---------------------------------------------------------------------------------------------------
REPORT_LANES = ["eval_all", "eval_natl", "canonical_test_all", "canonical_test_natl", "wc_slate"]
REPORT_METRICS = ["grid_info", "grid_nll", "rps", "acc", "ece_outcome", "exact_lift", "pts_g_31"]


def _latest_rows():
    if not REG.exists():
        return []
    rows = [json.loads(l) for l in REG.read_text().splitlines() if l.strip()]
    latest, order = {}, []
    for r in rows:
        if r["name"] not in latest:
            order.append(r["name"])
        latest[r["name"]] = r
    return [latest[n] for n in order]


def regen_report():
    rows = _latest_rows()
    base = next((r for r in rows if r["name"] == BASELINE_NAME), None)
    L = ["# Ablation results (generated — do not hand-edit; `run_ablation.py --report` regenerates)",
         "",
         "Core model = calibrated scoreline distribution P(home,away); 3/1 points is a REFERENCE column, "
         "never a gate. Metrics per DESIGN.md. `grid_info` = nats of score-level information over the "
         "train-empirical-prior null (>0 = model beats the modal-score prior). `exact_lift` = EV-pick "
         "exact rate ÷ always-modal exact rate (1.0 = no better than always guessing the mode).",
         "",
         "**Lanes.** `eval_*` = the split's pooled val∪test eval set (pooled split). "
         "`canonical_test_*` = the historical test lane (canonical split, for continuity). "
         "`wc_slate` = the frozen 104-game WC2026 benchmark, scored from TRAIN-split seeds "
         "(honest out-of-tournament view — differs from the full-data production goalnet.pt WC numbers). "
         "`_natl` restricts to national-team competitions (ids 9-15).",
         "",
         f"Δ columns (on pooled `eval_*` lanes) are vs baseline **{BASELINE_NAME}**"
         f"{' (not yet registered)' if base is None else ''}.",
         ""]
    for r in rows:
        c = r["config"]
        L.append(f"## {r['name']}  ·  {c['split']} · β={c['beta']} W={c['W']} "
                 f"seeds={c['seeds']} ep={c['epochs']} · {r['wall_min']}min · `{r['git_commit'][:8]}`"
                 f"{' ⚠dirty' if r['dirty'] else ''}")
        if c.get("notes"):
            L.append(f"_{c['notes']}_")
        L.append("")
        L.append("| lane | n | " + " | ".join(REPORT_METRICS) + " |")
        L.append("|" + "---|" * (len(REPORT_METRICS) + 2))
        for lane in REPORT_LANES:
            if lane not in r["metrics"]:
                continue
            s = r["metrics"][lane]
            cells = []
            for mname in REPORT_METRICS:
                v = s.get(mname)
                cell = "—" if v is None else (f"{v:.4f}" if mname in ("grid_info", "grid_nll", "rps", "ece_outcome")
                                              else f"{v:.3f}" if mname in ("acc", "pts_g_31")
                                              else f"{v:.2f}")
                if base and lane.startswith("eval") and lane in base["metrics"] and v is not None:
                    bv = base["metrics"][lane].get(mname)
                    if bv is not None and r["name"] != BASELINE_NAME:
                        cell += f" ({v - bv:+.4f})" if mname in ("grid_info", "grid_nll", "rps", "ece_outcome") else f" ({v - bv:+.3f})"
                cells.append(cell)
            L.append(f"| {lane} | {s['n']} | " + " | ".join(cells) + " |")
        L.append("")
    # fold in any per-run diagnostics sections (written by --diagnose) for registered runs
    names = {r["name"] for r in rows}
    diag_files = sorted(DIAG.glob("*.md")) if DIAG.exists() else []
    diags = [f for f in diag_files if f.stem in names]
    if diags:
        L += ["---", "", "# Diagnostics", ""]
        for f in diags:
            L.append(f.read_text(encoding="utf-8")); L.append("")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"  regenerated {REPORT.name} ({len(rows)} run{'s' if len(rows) != 1 else ''})", flush=True)


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def diagnose(name):
    """Prior-coasting diagnostics for a registered run (grid-NLL vs null, per-scoreline lift,
    calibration reliability, sharpness) written to diagnostics/<name>.md and folded into the report."""
    cache = RATES / f"{name}.npz"
    if not cache.exists():
        sys.exit(f"no rates cache for '{name}' at {cache} (run the experiment first)")
    z = np.load(cache, allow_pickle=True)
    rho = float(z["rho"]); split = str(z["split"])
    prior = z["prior"]
    modal = np.unravel_index(prior.argmax(), prior.shape)
    all_tag = "canonical_test" if split == "canonical" else "eval"
    ev_grids = grids_from_arrays(z["ev_lh"], z["ev_la"], rho)
    natl = z["ev_natl"]
    lanes = {
        f"{all_tag}_all": (ev_grids, z["ev_y"], z["ev_hg"], z["ev_ag"]),
        f"{all_tag}_natl": (ev_grids[natl], z["ev_y"][natl], z["ev_hg"][natl], z["ev_ag"][natl]),
    }
    if "wc_lh" in z.files:                                   # absent for --ctx-extra runs (WC lane skipped)
        wc_grids = grids_from_arrays(z["wc_lh"], z["wc_la"], rho)
        lanes["wc_slate"] = (wc_grids, z["wc_y"], z["wc_hg"], z["wc_ag"])
    S = {ln: metrics.suite(g, y, hg, ag, prior) for ln, (g, y, hg, ag) in lanes.items()}

    L = [f"## Diagnostics: {name}", "",
         f"Prior null = train-split empirical score grid; modal scoreline = **{modal[0]}-{modal[1]}** "
         f"(P={prior[modal]:.3f}). DC rho={rho}. Does the model add score-level information, or coast on "
         "the modal-score prior?", "",
         "**(1) Score-level information — grid-NLL vs the empirical-prior null**", ""]
    L += _md_table(["lane", "n", "grid_nll", "grid_nll_prior", "grid_info (nats)", "sharpness (nats)", "ece"],
                   [[ln, S[ln]["n"], f"{S[ln]['grid_nll']:.4f}", f"{S[ln]['grid_nll_prior']:.4f}",
                     f"{S[ln]['grid_info']:+.4f}", f"{S[ln]['sharpness']:.3f}", f"{S[ln]['ece_outcome']:.4f}"]
                    for ln in lanes])
    # per-scoreline lift on the on-target lanes
    for ln in [f"{all_tag}_natl"] + (["wc_slate"] if "wc_slate" in lanes else []):
        g, y, hg, ag = lanes[ln]
        pred_rows, true_rows = metrics.lift_table(g, hg, ag, prior)
        L += ["", f"**(2) Per-scoreline lift — {ln}.** EV-pick precision vs prior cell prob "
              "(off-modal picks with precision > prior_p = genuine score information):", ""]
        L += _md_table(["EV-pick", "picked", "hits", "precision", "prior_p", "off-modal?"],
                       [[r["score"], r["picked"], r["hit"], f"{r['precision']:.3f}", f"{r['prior_p']:.3f}",
                         "" if (int(r["score"].split("-")[0]), int(r["score"].split("-")[1])) == modal else "yes"]
                        for r in pred_rows[:12]])
        L += ["", f"Top-3-mass recall by true scoreline — {ln}:", ""]
        L += _md_table(["true score", "n", "top3_recall"],
                       [[r["score"], r["n"], f"{r['top3_recall']:.3f}"] for r in true_rows[:10]])
    # calibration reliability on the national lane
    ln = f"{all_tag}_natl"
    g, y, hg, ag = lanes[ln]
    rel = metrics.reliability(g, y, hg, ag)
    L += ["", f"**(3) Calibration reliability — {ln}** (outcome max-prob & exact-cell):", ""]
    L += _md_table(["kind", "pred_bin", "mean_pred", "observed", "n"],
                   [[k, f"{r['lo']:.2f}-{r['hi']:.2f}", f"{r['pred']:.3f}", f"{r['obs']:.3f}", r["n"]]
                    for k in ("outcome", "exact") for r in rel[k]])
    # verdict
    nat = S[f"{all_tag}_natl"]; alll = S[f"{all_tag}_all"]; wc = S.get("wc_slate")
    ng, nhg, nag = lanes[f"{all_tag}_natl"][0], lanes[f"{all_tag}_natl"][2], lanes[f"{all_tag}_natl"][3]
    nat_pred_rows = metrics.lift_table(ng, nhg, nag, prior)[0]
    off = [r for r in nat_pred_rows
           if (int(r["score"].split("-")[0]), int(r["score"].split("-")[1])) != modal]
    off_prec = sum(r["hit"] for r in off) / max(1, sum(r["picked"] for r in off))
    off_prior = np.mean([r["prior_p"] for r in off]) if off else float("nan")
    L += ["", "**(4) Verdict.**",
          f"On the national lane the model adds **{nat['grid_info']:+.4f} nats** of score-level "
          f"information over the modal-score prior (grid_nll {nat['grid_nll']:.3f} vs prior "
          f"{nat['grid_nll_prior']:.3f}); exact-score lift **{nat['exact_lift']:.2f}×** the always-modal "
          f"rate; outcome ECE {nat['ece_outcome']:.3f}; sharpness {nat['sharpness']:.2f} nats. "
          f"Off-modal EV-picks ({sum(r['picked'] for r in off)} of them) hit at precision "
          f"**{off_prec:.3f}** vs mean prior cell prob {off_prior:.3f} — "
          f"{'the model is extracting genuine off-modal score signal' if nat['grid_info'] > 0.02 and off_prec > off_prior else 'the model is largely coasting on the modal prior'}. "
          + (f"WC-slate grid_info {wc['grid_info']:+.4f}, exact_lift {wc['exact_lift']:.2f}×. " if wc else "WC-slate lane N/A (--ctx-extra run). ")
          + f"On the broad all-competitions lane, by contrast, grid_info is "
          f"**{alll['grid_info']:+.4f}** (exact_lift {alll['exact_lift']:.2f}×) — "
          f"{'the model does NOT beat the empirical prior at the exact-cell level there' if alll['grid_info'] <= 0 else 'the model modestly beats the prior there'}: "
          "its score-level edge is concentrated on the national/WC lanes it is upweighted (W) for, "
          "which is the intended target. So the network is not merely echoing the modal score on the "
          "games that matter, but it adds little scoreline information on club-heavy fixtures.", ""]

    DIAG.mkdir(exist_ok=True)
    (DIAG / f"{name}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote diagnostics/{name}.md")
    regen_report()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--npz", default="players_imp.npz")
    ap.add_argument("--split", default="pooled", choices=["pooled", "canonical"])
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--w", type=float, default=15.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--decay-halflife", type=float, default=None)
    ap.add_argument("--ctx-extra", nargs="*", default=[])
    ap.add_argument("--notes", default="")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--diagnose")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        regen_report()
    elif args.diagnose:
        diagnose(args.diagnose)
    elif args.name:
        run(args)
    else:
        ap.error("one of --name / --diagnose / --report is required")


if __name__ == "__main__":
    main()
