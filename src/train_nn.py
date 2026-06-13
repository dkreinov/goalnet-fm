"""Phase 2 model: per-player attention encoder + Poisson goal-rate head.

Each starter -> feature vector (47 FM attributes on 1-20 scale /20, + presence mask + position group).
A shared MLP encodes each player; masked attention pooling (+ mean) collapses the 11 starters into a
team vector; the two team vectors + context feed an MLP that emits two Poisson rates (lambda_home,
lambda_away). H/D/A probabilities are derived from the bivariate Poisson score matrix and scored with
accuracy / log-loss / RPS against the same baselines as train.py, on the 2025-26 time-split test set.

torch 2.2 is incompatible with this machine's numpy 2.4 *bridge*; we therefore never hand a numpy array
to torch — all tensors are built from Python lists. numpy is still used for the score matrix only.

Usage: python D:/Programming/claude/FM/src/train_nn.py
"""
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as bd

import torch
import torch.nn as nn

# ---- canonical attribute vocabulary (fminside 47) + source name aliases -------------------
ATTR_VOCAB = [
    # technical (10)
    "crossing", "dribbling", "finishing", "first-touch", "heading", "long-shots", "marking",
    "passing", "tackling", "technique",
    # set pieces (4)
    "corners", "free-kick-taking", "long-throws", "penalty-taking",
    # mental (14)
    "aggression", "anticipation", "bravery", "composure", "concentration", "decisions",
    "determination", "flair", "leadership", "off-the-ball", "positioning", "teamwork", "vision",
    "work-rate",
    # physical (8)
    "acceleration", "agility", "balance", "jumping-reach", "natural-fitness", "pace", "stamina",
    "strength",
    # goalkeeping (11)
    "aerial-reach", "command-of-area", "communication", "eccentricity", "handling", "kicking",
    "one-on-ones", "punching-tendency", "reflexes", "rushing-out-tendency", "throwing",
]
ATTR_IDX = {a: i for i, a in enumerate(ATTR_VOCAB)}
NAME_ALIAS = {"free-kicks": "free-kick-taking", "penalties": "penalty-taking"}
POS_GROUPS = ["GK", "DEF", "MID", "ATT"]
POS_IDX = {p: i for i, p in enumerate(POS_GROUPS)}
NA = len(ATTR_VOCAB)                 # 47
PLAYER_DIM = NA * 2 + len(POS_GROUPS)  # values + mask + pos one-hot = 98
MAX_PLAYERS = 11
CTX = 8
CLASSES = ["H", "D", "A"]
MAXG = 10                            # truncate Poisson score grid at 10 goals


def player_vector(attr_map, pos_group):
    """attr_map: {(category, attr_name): value 1-20}. Returns (values+mask+pos) length 98 list."""
    vals = [0.0] * NA
    mask = [0.0] * NA
    for (_cat, name), v in attr_map.items():
        name = NAME_ALIAS.get(name, name)
        j = ATTR_IDX.get(name)
        if j is not None:
            vals[j] = v / 20.0
            mask[j] = 1.0
    pos = [0.0] * len(POS_GROUPS)
    pos[POS_IDX.get(pos_group, POS_IDX["MID"])] = 1.0
    return vals + mask + pos


def build_examples():
    """Return list of dicts: {date, home[11x98], home_mask[11], away[...], ctx[8], hg, ag, result, b365}."""
    con = db.connect()
    snaps = bd.load_snapshots(con)
    attrs = bd.load_attrs(con)
    idx, has_snap = bd.name_index(con)
    bd.build_fallback(idx, con)
    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}

    matches = con.execute(
        "SELECT match_id, match_date, home_club_id, away_club_id, home_goals, away_goals "
        "FROM match ORDER BY match_date").fetchall()
    ctxd = bd.elo_and_form([tuple(m) for m in matches])

    lineups = defaultdict(list)
    for r in con.execute(
            "SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        lineups[(r[0], r[2])].append((r[1], r[3]))

    examples = []
    for mid, date, hcid, acid, hg, ag in matches:
        mrow = con.execute("SELECT b365h, b365d, b365a FROM match WHERE match_id=?", (mid,)).fetchone()

        def side_players(cid):
            out = []
            for pid, pos in lineups.get((mid, cid), []):
                rpid = bd.resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
                if rpid is None:
                    continue
                snap = bd.latest_before(snaps.get(rpid, []), date)
                if snap is None:
                    continue
                sid = snap[1]
                pg = bd.POS_GROUP.get((pos or " ")[0], "MID")
                out.append(player_vector(attrs.get(sid, {}), pg))
                if len(out) == MAX_PLAYERS:
                    break
            return out

        hp, ap = side_players(hcid), side_players(acid)
        if len(hp) < 8 or len(ap) < 8:        # need most of both XIs
            continue
        c = ctxd[mid]
        ctx = [
            1.0,                                              # home indicator (always 1; asymmetry is in ordering)
            (c["elo_home"] - c["elo_away"]) / 400.0,
            c["form_pts_home"] / 3.0, c["form_pts_away"] / 3.0,
            c["form_gd_home"] / 3.0, c["form_gd_away"] / 3.0,
            (c["rest_home"] if c["rest_home"] is not None else 4) / 7.0,
            (c["rest_away"] if c["rest_away"] is not None else 4) / 7.0,
        ]
        examples.append({
            "date": date, "home": hp, "away": ap, "ctx": ctx,
            "hg": hg, "ag": ag,
            "result": "H" if hg > ag else ("A" if ag > hg else "D"),
            "b365": [mrow[0], mrow[1], mrow[2]],
        })
    con.close()
    return examples


def pad(players):
    """players: list of <=11 vectors length 98 -> (padded 11x98 list, mask length 11 list)."""
    m = [1.0] * len(players) + [0.0] * (MAX_PLAYERS - len(players))
    p = players + [[0.0] * PLAYER_DIM for _ in range(MAX_PLAYERS - len(players))]
    return p, m


# ---- model ---------------------------------------------------------------------------------
class TeamEncoder(nn.Module):
    def __init__(self, d_player=PLAYER_DIM, d_hid=96, d_emb=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_player, d_hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d_hid, d_emb), nn.ReLU())
        self.attn = nn.Linear(d_emb, 1)
        self.d_emb = d_emb

    def forward(self, players, mask):
        # players: (B, 11, 98); mask: (B, 11)
        h = self.enc(players)                                  # (B,11,d)
        score = self.attn(h).squeeze(-1)                        # (B,11)
        score = score.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(-1)           # (B,11,1)
        attn_vec = (w * h).sum(dim=1)                           # (B,d)
        msum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_vec = (h * mask.unsqueeze(-1)).sum(dim=1) / msum   # (B,d)
        return torch.cat([attn_vec, mean_vec], dim=1)           # (B,2d)


class MatchNet(nn.Module):
    def __init__(self, d_emb=64, ctx=CTX):
        super().__init__()
        self.team = TeamEncoder(d_emb=d_emb)
        t = d_emb * 2
        self.head = nn.Sequential(
            nn.Linear(t * 3 + ctx, 96), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 2))

    def forward(self, hp, hm, ap, am, ctx):
        H = self.team(hp, hm)
        A = self.team(ap, am)
        feat = torch.cat([H, A, H - A, ctx], dim=1)
        lam = torch.nn.functional.softplus(self.head(feat)) + 0.05  # (B,2) positive rates
        return lam


# ---- Poisson score matrix -> H/D/A ---------------------------------------------------------
def hda_from_rates(lh, la):
    import numpy as np
    i = np.arange(MAXG + 1)
    ph = np.exp(-lh) * lh ** i / np.array([math.factorial(k) for k in i])
    pa = np.exp(-la) * la ** i / np.array([math.factorial(k) for k in i])
    M = np.outer(ph, pa)
    M = M / M.sum()
    home = np.tril(M, -1).sum()
    draw = np.trace(M)
    away = np.triu(M, 1).sum()
    return [home, draw, away]


def rps(y_idx, proba):
    import numpy as np
    proba = np.array(proba)
    cum_p = np.cumsum(proba, axis=1)
    cum_o = np.cumsum(np.eye(3)[y_idx], axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def report(name, y_idx, proba):
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss
    pred = np.argmax(proba, axis=1)
    print(f"{name:30s} acc={accuracy_score(y_idx, pred):.4f} "
          f"logloss={log_loss(y_idx, proba, labels=[0,1,2]):.4f} rps={rps(y_idx, proba):.4f}")


def main():
    torch.manual_seed(7)
    ex = build_examples()
    print(f"usable matches (>=8 starters/side): {len(ex)}")
    train = [e for e in ex if e["date"] < "2025-08-01"]
    test = [e for e in ex if e["date"] >= "2025-08-01"]
    print(f"train {len(train)}  test {len(test)}")
    if len(test) < 50:
        print("not enough test data"); return

    def batchify(items):
        HP, HM, AP, AM, CX, HG, AG = [], [], [], [], [], [], []
        for e in items:
            p, m = pad(e["home"]); HP.append(p); HM.append(m)
            p, m = pad(e["away"]); AP.append(p); AM.append(m)
            CX.append(e["ctx"]); HG.append(e["hg"]); AG.append(e["ag"])
        T = torch.tensor
        return (T(HP), T(HM), T(AP), T(AM), T(CX, dtype=torch.float32),
                T(HG, dtype=torch.float32), T(AG, dtype=torch.float32))

    tr = batchify(train)
    te = batchify(test)
    net = MatchNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    poisson = nn.PoissonNLLLoss(log_input=False, full=True)

    n = len(train)
    bs = 64
    best_state, best_val = None, 1e9
    val_cut = int(n * 0.85)
    for epoch in range(160):
        net.train()
        perm = torch.randperm(val_cut)
        for i in range(0, val_cut, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lam = net(tr[0][b], tr[1][b], tr[2][b], tr[3][b], tr[4][b])
            loss = poisson(lam[:, 0], tr[5][b]) + poisson(lam[:, 1], tr[6][b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vi = torch.arange(val_cut, n)
            lam = net(tr[0][vi], tr[1][vi], tr[2][vi], tr[3][vi], tr[4][vi])
            vloss = (poisson(lam[:, 0], tr[5][vi]) + poisson(lam[:, 1], tr[6][vi])).item()
        if vloss < best_val:
            best_val, best_state = vloss, {k: v.clone() for k, v in net.state_dict().items()}
    if best_state:
        net.load_state_dict(best_state)

    # evaluate on test
    net.eval()
    y_idx = [CLASSES.index(e["result"]) for e in test]
    with torch.no_grad():
        lam = net(te[0], te[1], te[2], te[3], te[4])
    proba_nn = [hda_from_rates(float(lam[k, 0]), float(lam[k, 1])) for k in range(len(test))]

    print("\n--- test = 2025-26 (time split, no leakage) ---")
    import numpy as np
    prior = np.bincount(y_idx, minlength=3) / len(y_idx)
    report("majority/prior", y_idx, [prior.tolist()] * len(y_idx))
    inv = 1.0 / np.array([e["b365"] for e in test], dtype=float)
    report("bookmaker (B365)", y_idx, (inv / inv.sum(axis=1, keepdims=True)).tolist())
    report("PlayerAttn + Poisson (NN)", y_idx, proba_nn)

    # report mean predicted rates sanity
    print(f"\nmean lambda_home={float(lam[:,0].mean()):.2f} lambda_away={float(lam[:,1].mean()):.2f} "
          f"(actual mean goals H={np.mean([e['hg'] for e in test]):.2f} "
          f"A={np.mean([e['ag'] for e in test]):.2f})")
    out = db.ROOT / "data" / "model_nn.pt"
    torch.save(net.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
