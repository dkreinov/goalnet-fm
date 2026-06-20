"""Feature-ablation experiment on the scoreline model. Toggles, one at a time and combined:
  detailed position embedding (9 vs 4), competition embedding, match metadata (kickoff/attendance/
  formation), and an attendance AUXILIARY head. Each config trains on all seasons except a held-out
  test season (default 2024-25, interior -> non-adjacent style) and reports acc / rps / fantasy-pts.
Usage: python D:/Programming/claude/FM/src/train_enr.py [--test 2024-25] [--epochs 45]
"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db, train_goals as tg
from build_player_dataset_pos import DET2ROLE
NATc = {9, 10, 11, 12, 13, 14, 15}
NCOMP, NFORM = 54, 24


def season_of(d):
    y = d.astype("datetime64[Y]").astype(int) + 1970
    m = d.astype("datetime64[M]").astype(int) % 12 + 1
    s = y if m >= 8 else y - 1
    return f"{s}-{str(s + 1)[2:]}"


class Enc2(nn.Module):
    def __init__(self, A, npos, d=64, h=128, p=0.3):
        super().__init__()
        self.npos = npos
        self.player = nn.Sequential(nn.Linear(A, 128), nn.ReLU(), nn.LayerNorm(128), nn.Dropout(p), nn.Linear(128, d))
        self.pos = nn.Embedding(npos, d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.team = nn.Sequential(nn.Linear(4 * d, h), nn.ReLU(), nn.LayerNorm(h), nn.Dropout(p), nn.Linear(h, h))
        self.register_buffer("d2r", torch.tensor([int(x) for x in DET2ROLE], dtype=torch.long))

    def forward(self, X, Rdet):
        emb = self.pos(Rdet if self.npos == 9 else self.d2r[Rdet])
        pe = self.enc(self.player(X) + emb)
        role = self.d2r[Rdet]
        pools = []
        for r in range(4):
            m = (role == r).unsqueeze(-1).float()
            pools.append((pe * m).sum(1) / m.sum(1).clamp(min=1))
        return self.team(torch.cat(pools, -1))


class GN2(nn.Module):
    def __init__(self, A, nctx, cfg, d=64, h=128, p=0.3):
        super().__init__()
        self.cfg = cfg
        self.enc = Enc2(A, 9 if cfg["det"] else 4, d, h, p)
        self.ad = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 2))
        self.ctx = nn.Sequential(nn.Linear(nctx, 32), nn.ReLU(), nn.Linear(32, 2))
        self.home_adv = nn.Parameter(torch.tensor(0.25))
        if cfg["comp"]:
            self.comp = nn.Embedding(NCOMP, 2)
            nn.init.zeros_(self.comp.weight)
        if cfg["form"]:
            self.form = nn.Embedding(NFORM, 1); nn.init.zeros_(self.form.weight)
        if cfg["meta"]:
            self.meta = nn.Linear(3, 2); nn.init.zeros_(self.meta.weight); nn.init.zeros_(self.meta.bias)
        if cfg["aux"]:
            self.att = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, Xh, Rh, Xa, Ra, C, comp, meta, hf, af):
        th = self.enc(Xh, Rh); ta = self.enc(Xa, Ra)
        adh, ada = self.ad(th), self.ad(ta)
        ch, ca = self.ctx(C).unbind(-1)
        logh = self.home_adv + adh[:, 0] - ada[:, 1] + ch
        loga = ada[:, 0] - adh[:, 1] + ca
        if self.cfg["comp"]:
            co = self.comp(comp); logh = logh + co[:, 0]; loga = loga + co[:, 1]
        if self.cfg["form"]:
            logh = logh + self.form(hf).squeeze(-1); loga = loga + self.form(af).squeeze(-1)
        if self.cfg["meta"]:
            mo = self.meta(meta); logh = logh + mo[:, 0]; loga = loga + mo[:, 1]
        att = self.att(torch.cat([th, ta], -1)).squeeze(-1) if self.cfg["aux"] else None
        return logh, loga, att


def main():
    def arg(k, d):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    test_season = arg("--test", "2024-25"); ep = int(arg("--epochs", "45"))
    z = np.load(ROOT / "data" / "players_pos.npz", allow_pickle=True)
    Xh, Xa = z["Xh"], z["Xa"]; Rh, Ra = z["Rh"].astype(np.int64), z["Ra"].astype(np.int64)
    y, dates, mids = z["y"].astype(np.int64), z["dates"], [int(m) for m in z["mids"]]
    A = Xh.shape[2]
    cz = np.load(ROOT / "data" / "context.npz"); _cc, _cm = cz["ctx"], cz["mids"]
    cm = {int(m): _cc[i] for i, m in enumerate(_cm)}
    nctx = _cc.shape[1]
    CTX = np.stack([cm.get(m, np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)
    mz = np.load(ROOT / "data" / "meta.npz")
    M = {int(mz["mids"][i]): (int(mz["comp"][i]), mz["kh"][i], mz["logatt"][i], mz["hasatt"][i],
                              int(mz["hform"][i]), int(mz["aform"][i])) for i in range(len(mz["mids"]))}
    comp = np.array([M.get(m, (0,))[0] for m in mids], np.int64)
    meta = np.array([[M.get(m, (0, -1, 0, 0))[1], M.get(m, (0, 0, 0))[2], M.get(m, (0, 0, 0, 0))[3]] for m in mids], np.float32)
    hf = np.array([M.get(m, (0, 0, 0, 0, 0))[4] for m in mids], np.int64)
    af = np.array([M.get(m, (0, 0, 0, 0, 0, 0))[5] for m in mids], np.int64)
    logatt = meta[:, 1].copy()
    con = db.connect()
    meta_db = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,competition_id,home_goals,away_goals FROM match")}
    natl = np.array([meta_db.get(m, (0, 0, 0))[0] in NATc for m in mids])
    hg = np.array([min(meta_db.get(m, (0, 0, 0))[1] or 0, tg.MAXG) for m in mids], np.float32)
    ag = np.array([min(meta_db.get(m, (0, 0, 0))[2] or 0, tg.MAXG) for m in mids], np.float32)
    season = np.array([season_of(d) for d in dates])
    te = season == test_season; tr = ~te
    print(f"test={test_season} (n={te.sum()})  train n={tr.sum()}", flush=True)

    mu = Xh[tr].reshape(-1, A).mean(0); sd = Xh[tr].reshape(-1, A).std(0) + 1e-6
    Xhn = ((Xh - mu) / sd).astype(np.float32); Xan = ((Xa - mu) / sd).astype(np.float32)
    cmu = CTX[tr].mean(0); csd = CTX[tr].std(0) + 1e-6; CTXn = ((CTX - cmu) / csd).astype(np.float32)
    am = logatt[tr][logatt[tr] > 0].mean(); asd = logatt[tr][logatt[tr] > 0].std() + 1e-6
    meta[:, 1] = np.where(meta[:, 1] > 0, (meta[:, 1] - am) / asd, 0)   # z-score logatt feature
    attz = np.where(logatt > 0, (logatt - am) / asd, 0).astype(np.float32)   # aux target
    _TT = {np.dtype("float32"): torch.float32, np.dtype("int64"): torch.int64}
    def T(a):
        a = np.ascontiguousarray(a); return torch.frombuffer(bytearray(a.tobytes()), dtype=_TT[a.dtype]).reshape(a.shape)
    def tonp(t): return np.array(t.detach().tolist(), dtype=np.float32)
    def pk(m):
        return [T(Xhn[m]), T(Rh[m]), T(Xan[m]), T(Ra[m]), T(CTXn[m]), T(comp[m]), T(meta[m]), T(hf[m]), T(af[m])]
    Tr = pk(tr); Te = pk(te); hgt, agt = T(hg[tr]), T(ag[tr]); wt = T(np.where(natl[tr], 5.0, 1.0).astype(np.float32))
    attt = T(attz[tr]); hast = T((logatt[tr] > 0).astype(np.float32))

    def run(name, cfg):
        torch.manual_seed(7); np.random.seed(7)
        net = GN2(A, nctx, cfg)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        pois = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
        bs, n = 512, Tr[0].size(0)
        for e in range(ep):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                lh, la, att = net(Tr[0][b], Tr[1][b], Tr[2][b], Tr[3][b], Tr[4][b], Tr[5][b], Tr[6][b], Tr[7][b], Tr[8][b])
                loss = ((pois(lh, hgt[b]) + pois(la, agt[b])) * wt[b]).mean()
                if cfg["aux"]:
                    loss = loss + 0.1 * (((att - attt[b]) ** 2) * hast[b]).mean()
                loss.backward(); opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            lh, la, _ = net(*Te)
        lh, la = np.exp(tonp(lh)), np.exp(tonp(la))
        ye, hge, age_ = y[te], hg[te], ag[te]
        P = np.array([tg.hda_from_P(tg.score_matrix(a, b)) for a, b in zip(lh, la)])
        acc = float((P.argmax(1) == ye).mean()); r = tg.rps(ye, P)
        tot = ex = 0
        for a, b, H, Aa in zip(lh, la, hge, age_):
            pkk = tg.ev_pick(tg.score_matrix(a, b)); pts, lab = tg.grade(pkk, int(H), int(Aa)); tot += pts; ex += lab == "exact"
        nm = natl[te]
        accn = float((P[nm].argmax(1) == ye[nm]).mean()) if nm.sum() else 0
        print(f"  {name:22s} acc={acc:.3f} rps={r:.4f} pts/g={tot/te.sum():.3f} exact%={ex/te.sum()*100:.1f}  natl_acc={accn:.3f}", flush=True)

    base = dict(det=False, comp=False, form=False, meta=False, aux=False)
    print("\nfeature ablation (held-out season):", flush=True)
    run("baseline(role4)", base)
    run("+detailed_pos", {**base, "det": True})
    run("+competition", {**base, "comp": True})
    run("+formation", {**base, "form": True})
    run("+metadata(kick/att)", {**base, "meta": True})
    run("+attendance_aux", {**base, "aux": True})
    run("ALL", dict(det=True, comp=True, form=True, meta=True, aux=True))


if __name__ == "__main__":
    main()
