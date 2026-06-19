"""Position-ordered 11v11 model: predict H/D/A from the 22 starters' FM attribute vectors, structured
by role (GK -> DEF -> MID -> ATT). Per-player encoder + role embedding -> mean-pool within each role ->
team vector -> [home, away] head -> 3-class result. Time split (no leakage), early stop on val RPS.
Usage: python D:/Programming/claude/FM/src/train_pos.py [--epochs N] [--dim D]
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(7); np.random.seed(7)
CLASSES = ["H", "D", "A"]


def rps(y_idx, proba):
    cum_p = np.cumsum(proba, axis=1); cum_o = np.cumsum(np.eye(3)[y_idx], axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def metrics(name, y_idx, proba):
    from sklearn.metrics import accuracy_score, log_loss
    pred = proba.argmax(1)
    print(f"  {name:26s} acc={accuracy_score(y_idx,pred):.4f} "
          f"logloss={log_loss(y_idx,proba,labels=[0,1,2]):.4f} rps={rps(y_idx,proba):.4f}", flush=True)
    return rps(y_idx, proba)


class PosNet(nn.Module):
    def __init__(self, A, d=64, h=128, p=0.3):
        super().__init__()
        self.player = nn.Sequential(nn.Linear(A, 128), nn.ReLU(), nn.LayerNorm(128),
                                    nn.Dropout(p), nn.Linear(128, d))
        self.role = nn.Embedding(4, d)
        self.team = nn.Sequential(nn.Linear(4 * d, h), nn.ReLU(), nn.LayerNorm(h),
                                  nn.Dropout(p), nn.Linear(h, h))
        self.head = nn.Sequential(nn.Linear(2 * h + 1, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 3))

    def team_repr(self, X, R):
        pe = self.player(X) + self.role(R)                 # (B,11,d)
        pools = []
        for r in range(4):
            mask = (R == r).unsqueeze(-1).float()
            pools.append((pe * mask).sum(1) / mask.sum(1).clamp(min=1))
        return self.team(torch.cat(pools, -1))             # (B,h)

    def forward(self, Xh, Rh, Xa, Ra):
        h = self.team_repr(Xh, Rh); a = self.team_repr(Xa, Ra)
        ha = torch.ones(h.size(0), 1, device=h.device)     # home-advantage constant
        return self.head(torch.cat([h, a, ha], -1))


def main():
    ep = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 120
    d = int(sys.argv[sys.argv.index("--dim") + 1]) if "--dim" in sys.argv else 64
    z = np.load(ROOT / "data" / "players.npz", allow_pickle=True)
    Xh, Xa, Rh, Ra, y, dates = z["Xh"], z["Xa"], z["Rh"].astype(np.int64), z["Ra"].astype(np.int64), z["y"].astype(np.int64), z["dates"]
    A = Xh.shape[2]
    print(f"matches {len(y):,}  attrs {A}  result dist H/D/A = {np.bincount(y)/len(y)}", flush=True)

    # time split: train < 2024-08, val 2024-25, test 2025-26
    tr = dates < np.datetime64("2024-08-01")
    va = (dates >= np.datetime64("2024-08-01")) & (dates < np.datetime64("2025-08-01"))
    te = dates >= np.datetime64("2025-08-01")
    print(f"train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}", flush=True)

    # standardize attrs on train (both teams share stats)
    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    def norm(X): return ((X - mu) / sd).astype(np.float32)
    Xh, Xa = norm(Xh), norm(Xa)

    # torch 2.2 can't read numpy 2.4 arrays (from_numpy/tensor fail) -> go through a raw byte buffer
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64, np.dtype("int8"): torch.int8}
    def T(a):
        a = np.ascontiguousarray(a)
        return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t):                                        # torch -> numpy also broken on 2.4 -> via list
        return np.array(t.detach().tolist(), dtype=np.float32)
    dev = "cpu"
    pack = lambda m: (T(Xh[m]), T(Rh[m]), T(Xa[m]), T(Ra[m]), T(y[m]))
    Xhtr, Rhtr, Xatr, Ratr, ytr = pack(tr)
    Xhva, Rhva, Xava, Rava, yva = pack(va)
    Xhte, Rhte, Xate, Rate, yte = pack(te)

    # baseline
    prior = np.bincount(y[tr], minlength=3) / tr.sum()
    print("baselines (test):")
    metrics("majority/prior", y[te], np.tile(prior, (te.sum(), 1)))

    net = PosNet(A, d=d).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    lossf = nn.CrossEntropyLoss()
    bs = 512
    n = Xhtr.size(0)
    best_rps, best_state, bad = 9, None, 0
    for e in range(ep):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xhtr[b], Rhtr[b], Xatr[b], Ratr[b])
            loss = lossf(out, ytr[b])
            loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = tonp(torch.softmax(net(Xhva, Rhva, Xava, Rava), 1))
        r = rps(y[va], pv)
        if r < best_rps - 1e-4:
            best_rps, best_state, bad = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if e % 10 == 0 or bad == 0:
            print(f"  epoch {e:3d}  val_rps={r:.4f}  best={best_rps:.4f}", flush=True)
        if bad >= 20:
            print(f"  early stop @ {e}", flush=True); break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pte = tonp(torch.softmax(net(Xhte, Rhte, Xate, Rate), 1))
        pva = tonp(torch.softmax(net(Xhva, Rhva, Xava, Rava), 1))
    print("PosNet results:")
    metrics("PosNet (val)", y[va], pva)
    metrics("PosNet (test)", y[te], pte)
    torch.save({"state": best_state, "mu": mu, "sd": sd, "A": A, "dim": d}, ROOT / "data" / "posnet.pt")
    print("saved data/posnet.pt", flush=True)


if __name__ == "__main__":
    main()
