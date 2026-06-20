"""Ablation: how good is the result model from team context (Elo/form) ALONE, with no FM player grades?
Compares against the grades+context model (0.2100) to isolate what the FM 11v11 grades actually add.
Same matches, same time split, same RPS metric as train_pos2.py. Uses sklearn (no torch needed).
Variants: Elo-diff only · all-Elo (3) · full 10-ctx — each as multinomial logistic + gradient boosting.
Usage: python D:/Programming/claude/FM/src/ctx_only.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent


def rps(y, p):
    cp = np.cumsum(p, 1); co = np.cumsum(np.eye(3)[y], 1)
    return float(np.mean(np.sum((cp - co) ** 2, 1) / 2))


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss

    pz = np.load(ROOT / "data" / "players.npz", allow_pickle=True)
    y = pz["y"].astype(np.int64); dates = pz["dates"]; mids = pz["mids"]
    cz = np.load(ROOT / "data" / "context.npz")
    cctx, cmids = cz["ctx"], cz["mids"]
    cmap = {int(m): cctx[i] for i, m in enumerate(cmids)}
    nctx = cctx.shape[1]
    CTX = np.stack([cmap.get(int(m), np.zeros(nctx, np.float32)) for m in mids]).astype(np.float32)
    # column layout from build_context.py:
    # 0 home_elo 1 away_elo 2 elo_diff 3 home_form 4 away_form 5 form_diff
    # 6 home_gdform 7 away_gdform 8 home_rest 9 away_rest

    tr = dates < np.datetime64("2024-08-01")
    va = (dates >= np.datetime64("2024-08-01")) & (dates < np.datetime64("2025-08-01"))
    te = dates >= np.datetime64("2025-08-01")
    print(f"matches {len(y):,}  train {tr.sum():,} val {va.sum():,} test {te.sum():,}", flush=True)

    prior = np.bincount(y[tr], minlength=3) / tr.sum()
    print("\nfeature set                 model   val_rps  test_rps  test_acc")
    print(f"  {'majority/prior':24s} {'-':6s}  {'-':7s}  "
          f"{rps(y[te], np.tile(prior,(te.sum(),1))):.4f}   "
          f"{accuracy_score(y[te], np.full(te.sum(), prior.argmax())):.4f}")

    feats = {"elo_diff only": [2], "all-Elo (3)": [0, 1, 2], "full 10-ctx": list(range(nctx))}
    for fname, cols in feats.items():
        Xtr, Xva, Xte = CTX[tr][:, cols], CTX[va][:, cols], CTX[te][:, cols]
        # logistic
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(Xtr, y[tr])
        pv, pt = lr.predict_proba(Xva), lr.predict_proba(Xte)
        print(f"  {fname:24s} {'logit':6s}  {rps(y[va],pv):.4f}  {rps(y[te],pt):.4f}    "
              f"{accuracy_score(y[te],pt.argmax(1)):.4f}")
        # gradient boosting
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                            max_depth=3, l2_regularization=1.0,
                                            validation_fraction=None, random_state=7)
        gb.fit(Xtr, y[tr])
        pv, pt = gb.predict_proba(Xva), gb.predict_proba(Xte)
        print(f"  {fname:24s} {'gbdt':6s}  {rps(y[va],pv):.4f}  {rps(y[te],pt):.4f}    "
              f"{accuracy_score(y[te],pt.argmax(1)):.4f}")

    print("\ncompare: grades + context (xfmr ensemble)  test_rps=0.2100  test_acc=0.4967", flush=True)


if __name__ == "__main__":
    main()
