"""Architecture sweep for the 11v11 result model. Same data/split/metric as train_pos.py, but compares
several team encoders and an ensemble, logging val/test RPS for each so we can see "what works".

Variants (--arch):
  mean : per-role mean-pool (the train_pos.py baseline)
  attn : per-role attention pool (learned query picks the important players in each role)
  diff : mean-pool encoder, but the result head sees difference features [h-a, h*a, h, a, adv]
  xfmr : a small self-attention encoder over the 11 players (lets defenders 'see' the attack) then pool

Each arch is trained with N seeds; we report the single best-by-val model AND the seed-ensemble
(mean of softmax probs), since ensembling usually lowers RPS. Time split, early stop on val RPS.
Usage: python D:/Programming/claude/FM/src/train_pos2.py [--arch mean,attn,diff,xfmr] [--seeds 3] [--epochs 150]
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["H", "D", "A"]


def rps(y_idx, proba):
    cum_p = np.cumsum(proba, axis=1); cum_o = np.cumsum(np.eye(3)[y_idx], axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def report(name, y_idx, proba):
    from sklearn.metrics import accuracy_score, log_loss
    pred = proba.argmax(1)
    r = rps(y_idx, proba)
    print(f"  {name:30s} acc={accuracy_score(y_idx,pred):.4f} "
          f"logloss={log_loss(y_idx,proba,labels=[0,1,2]):.4f} rps={r:.4f}", flush=True)
    return r


class Encoder(nn.Module):
    """X (B,11,A), R (B,11) -> team vector (B,h). Pooling mode set by `arch`."""
    def __init__(self, A, arch, d=64, h=128, p=0.3):
        super().__init__()
        self.arch = arch
        self.player = nn.Sequential(nn.Linear(A, 128), nn.ReLU(), nn.LayerNorm(128),
                                    nn.Dropout(p), nn.Linear(128, d))
        self.role = nn.Embedding(4, d)
        if arch == "attn":
            self.q = nn.Parameter(torch.randn(4, d) * 0.1)       # one query per role
        if arch == "xfmr":
            layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                               dropout=p, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.team = nn.Sequential(nn.Linear(4 * d, h), nn.ReLU(), nn.LayerNorm(h),
                                  nn.Dropout(p), nn.Linear(h, h))
        self.d = d

    def forward(self, X, R):
        pe = self.player(X) + self.role(R)                       # (B,11,d)
        if self.arch == "xfmr":
            pe = self.enc(pe)
        pools = []
        for r in range(4):
            m = (R == r)                                         # (B,11)
            mf = m.unsqueeze(-1).float()
            if self.arch == "attn":
                score = (pe * self.q[r]).sum(-1)                 # (B,11)
                score = score.masked_fill(~m, -1e9)
                w = torch.softmax(score, -1).unsqueeze(-1)       # (B,11,1)
                pools.append((pe * w).sum(1))
            else:
                pools.append((pe * mf).sum(1) / mf.sum(1).clamp(min=1))
        return self.team(torch.cat(pools, -1))                   # (B,h)


class PosNet(nn.Module):
    def __init__(self, A, arch, d=64, h=128, p=0.3, nctx=0):
        super().__init__()
        self.enc = Encoder(A, arch, d, h, p)
        self.diff = (arch == "diff")
        self.nctx = nctx
        feat = (4 * h + 1 if self.diff else 2 * h + 1) + nctx
        self.head = nn.Sequential(nn.Linear(feat, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 3))

    def forward(self, Xh, Rh, Xa, Ra, C=None):
        h = self.enc(Xh, Rh); a = self.enc(Xa, Ra)
        adv = torch.ones(h.size(0), 1, device=h.device)
        z = torch.cat([h - a, h * a, h, a, adv], -1) if self.diff else torch.cat([h, a, adv], -1)
        if self.nctx:
            z = torch.cat([z, C], -1)
        return self.head(z)


def main():
    def arg(k, default):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default
    archs = arg("--arch", "mean,attn,diff,xfmr").split(",")
    seeds = int(arg("--seeds", "3"))
    ep = int(arg("--epochs", "150"))

    npz = arg("--npz", "players.npz")
    z = np.load(ROOT / "data" / npz, allow_pickle=True)
    print(f"data={npz}", flush=True)
    Xh, Xa = z["Xh"], z["Xa"]
    Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates = z["y"].astype(np.int64), z["dates"]
    mids = z["mids"]
    A = Xh.shape[2]
    print(f"matches {len(y):,}  attrs {A}  H/D/A={np.bincount(y)/len(y)}", flush=True)

    use_ctx = "--ctx" in sys.argv
    CTX = None; nctx = 0
    if use_ctx:
        cz = np.load(ROOT / "data" / "context.npz")
        cctx, cmids = cz["ctx"], cz["mids"]          # materialize once (NpzFile indexing is lazy)
        cmap = {int(m): cctx[i] for i, m in enumerate(cmids)}
        nctx = cctx.shape[1]
        CTX = np.stack([cmap.get(int(m), np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)
        miss = sum(int(m) not in cmap for m in mids)
        print(f"ctx: {nctx} features, {miss} matches missing context", flush=True)

    tr = dates < np.datetime64("2024-08-01")
    va = (dates >= np.datetime64("2024-08-01")) & (dates < np.datetime64("2025-08-01"))
    te = dates >= np.datetime64("2025-08-01")
    print(f"train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}", flush=True)

    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xh = ((Xh - mu) / sd).astype(np.float32); Xa = ((Xa - mu) / sd).astype(np.float32)
    if use_ctx:
        cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6
        CTX = ((CTX - cmu) / csd).astype(np.float32)

    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a)
        return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t):
        return np.array(t.detach().tolist(), dtype=np.float32)
    pack = lambda m: (T(Xh[m]), T(Rh[m]), T(Xa[m]), T(Ra[m]), T(y[m]))
    Xhtr, Rhtr, Xatr, Ratr, ytr = pack(tr)
    Vh, Vrh, Va, Vra, _ = pack(va)
    Eh, Erh, Ea, Era, _ = pack(te)
    Ctr = T(CTX[tr]) if use_ctx else None
    Cva = T(CTX[va]) if use_ctx else None
    Cte = T(CTX[te]) if use_ctx else None

    prior = np.bincount(y[tr], minlength=3) / tr.sum()
    print("baseline:")
    report("majority/prior (test)", y[te], np.tile(prior, (te.sum(), 1)))

    def proba(net, Xh_, Rh_, Xa_, Ra_, C_=None):
        net.eval()
        with torch.no_grad():
            return tonp(torch.softmax(net(Xh_, Rh_, Xa_, Ra_, C_), 1))

    def train_one(arch, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        net = PosNet(A, arch, nctx=nctx)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        lossf = nn.CrossEntropyLoss()
        bs, n = 512, Xhtr.size(0)
        best, best_state, bad = 9, None, 0
        for e in range(ep):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                Cb = Ctr[b] if use_ctx else None
                loss = lossf(net(Xhtr[b], Rhtr[b], Xatr[b], Ratr[b], Cb), ytr[b])
                loss.backward(); opt.step()
            sched.step()
            r = rps(y[va], proba(net, Vh, Vrh, Va, Vra, Cva))
            if r < best - 1e-4:
                best, best_state, bad = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
            if bad >= 20:
                break
        net.load_state_dict(best_state)
        return net, best

    summary = []
    best_overall = (9, None, None)
    for arch in archs:
        print(f"\n=== arch={arch} ===", flush=True)
        val_ps, test_ps, val_rs = [], [], []
        for s in range(seeds):
            net, vbest = train_one(arch, 7 + s)
            pv = proba(net, Vh, Vrh, Va, Vra, Cva); pe = proba(net, Eh, Erh, Ea, Era, Cte)
            val_ps.append(pv); test_ps.append(pe); val_rs.append(vbest)
            rt = report(f"{arch} seed{s} (test)", y[te], pe)
            if vbest < best_overall[0]:
                best_overall = (vbest, arch, {k: v.clone() for k, v in net.state_dict().items()})
        ens_v = np.mean(val_ps, 0); ens_t = np.mean(test_ps, 0)
        rv = report(f"{arch} ENSEMBLE (val)", y[va], ens_v)
        rt = report(f"{arch} ENSEMBLE (test)", y[te], ens_t)
        summary.append((arch, np.mean(val_rs), rv, rt))

    print("\n=== SUMMARY (sorted by ensemble val rps) ===", flush=True)
    for arch, mv, rv, rt in sorted(summary, key=lambda x: x[2]):
        print(f"  {arch:6s} mean-seed val_rps={mv:.4f}  ensemble val={rv:.4f}  test={rt:.4f}", flush=True)

    vbest, barch, bstate = best_overall
    torch.save({"state": bstate, "arch": barch, "mu": mu, "sd": sd, "A": A},
               ROOT / "data" / "posnet_best.pt")
    print(f"\nbest single model: arch={barch} val_rps={vbest:.4f} -> data/posnet_best.pt", flush=True)


if __name__ == "__main__":
    main()
