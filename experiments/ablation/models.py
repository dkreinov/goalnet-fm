"""Model zoo for Phase-5 architecture ablations (contract: experiments/ablation/DESIGN.md).

`build_model(arch, A, nctx)` is the single construction point used by run_ablation.train_one.
PARITY RULE: for arch="goalnet" this must be *bit-for-bit* identical to constructing
`tg.GoalNet(A, nctx)` inline — train_one seeds torch RNG immediately before building the net, so
the factory must not consume any RNG draws before (or instead of) the exact same module-creation
sequence. New variants get their own arch key; production src/train_goals.py is never modified.

Input contract shared by ALL variants (keeps the frozen WC-slate lane usable):
    forward(Xh (B,11,A), Rh (B,11) int64, Xa, Ra, C (B,nctx)) -> (log_lambda_home, log_lambda_away)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import train_goals as tg  # noqa: E402

ARCHS = ["goalnet", "cross22", "latecross"]


class Cross22GoalNet(nn.Module):
    """Cross-team attention variant: one transformer over all 22 players (both XIs + team
    embedding), so each player token attends to opponents as well as teammates (HIGFormer-style
    match-comparison). Per-team/per-role masked-mean pooling then feeds the SAME team->att/def
    head shape and rate equation as GoalNet. d=64/2 layers/nhead=4 keeps params ~GoalNet-sized."""

    def __init__(self, A, nctx, d=64, h=128, p=0.3):
        super().__init__()
        self.player = nn.Sequential(nn.Linear(A, 128), nn.ReLU(), nn.LayerNorm(128),
                                    nn.Dropout(p), nn.Linear(128, d))
        self.role = nn.Embedding(4, d)
        self.side = nn.Embedding(2, d)                        # team-membership embedding
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                           dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.team = nn.Sequential(nn.Linear(4 * d, h), nn.ReLU(), nn.LayerNorm(h),
                                  nn.Dropout(p), nn.Linear(h, h))
        self.ad = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 2))
        self.ctx = nn.Sequential(nn.Linear(nctx, 32), nn.ReLU(), nn.Linear(32, 2))
        self.home_adv = nn.Parameter(torch.tensor(0.25))

    def _pool(self, pe, R):
        """(B,11,d) tokens + (B,11) roles -> (B,4d) role-pooled team vector (masked mean)."""
        pools = []
        for r in range(4):
            mf = (R == r).unsqueeze(-1).float()
            pools.append((pe * mf).sum(1) / mf.sum(1).clamp(min=1))
        return torch.cat(pools, -1)

    def forward(self, Xh, Rh, Xa, Ra, C):
        eh = self.player(Xh) + self.role(Rh) + self.side.weight[0]
        ea = self.player(Xa) + self.role(Ra) + self.side.weight[1]
        pe = self.enc(torch.cat([eh, ea], 1))                 # (B,22,d) — cross-team attention here
        th = self.team(self._pool(pe[:, :11], Rh))
        ta = self.team(self._pool(pe[:, 11:], Ra))
        adh, ada = self.ad(th), self.ad(ta)
        ch, ca = self.ctx(C).unbind(-1)
        logh = self.home_adv + adh[:, 0] - ada[:, 1] + ch
        loga = ada[:, 0] - adh[:, 1] + ca
        return logh, loga


class LateCrossGoalNet(nn.Module):
    """Fallback Arm-X variant: keep GoalNet's per-team encoding philosophy (11-token self
    transformer per XI), then ONE late cross-attention block where each team's tokens attend to
    the opponent's (residual), before role-pooling and the same team/ad/ctx/rate heads."""

    def __init__(self, A, nctx, d=64, h=128, p=0.3):
        super().__init__()
        self.player = nn.Sequential(nn.Linear(A, 128), nn.ReLU(), nn.LayerNorm(128),
                                    nn.Dropout(p), nn.Linear(128, d))
        self.role = nn.Embedding(4, d)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                           dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.xattn = nn.MultiheadAttention(d, 4, dropout=p, batch_first=True)
        self.xnorm = nn.LayerNorm(d)
        self.team = nn.Sequential(nn.Linear(4 * d, h), nn.ReLU(), nn.LayerNorm(h),
                                  nn.Dropout(p), nn.Linear(h, h))
        self.ad = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 2))
        self.ctx = nn.Sequential(nn.Linear(nctx, 32), nn.ReLU(), nn.Linear(32, 2))
        self.home_adv = nn.Parameter(torch.tensor(0.25))

    def _pool(self, pe, R):
        pools = []
        for r in range(4):
            mf = (R == r).unsqueeze(-1).float()
            pools.append((pe * mf).sum(1) / mf.sum(1).clamp(min=1))
        return torch.cat(pools, -1)

    def _side(self, own, opp, Rown):
        x = self.xnorm(own + self.xattn(own, opp, opp, need_weights=False)[0])
        return self.team(self._pool(x, Rown))

    def forward(self, Xh, Rh, Xa, Ra, C):
        ph = self.enc(self.player(Xh) + self.role(Rh))
        pa = self.enc(self.player(Xa) + self.role(Ra))
        th = self._side(ph, pa, Rh); ta = self._side(pa, ph, Ra)
        adh, ada = self.ad(th), self.ad(ta)
        ch, ca = self.ctx(C).unbind(-1)
        logh = self.home_adv + adh[:, 0] - ada[:, 1] + ch
        loga = ada[:, 0] - adh[:, 1] + ca
        return logh, loga


def build_model(arch, A, nctx):
    if arch == "goalnet":
        return tg.GoalNet(A, nctx)
    if arch == "cross22":
        return Cross22GoalNet(A, nctx)
    if arch == "latecross":
        return LateCrossGoalNet(A, nctx)
    raise ValueError(f"unknown arch '{arch}' (known: {ARCHS})")
