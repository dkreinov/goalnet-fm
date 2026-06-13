"""POC training: predict H/D/A from FM lineup ratings + context.
Baselines (majority, odds, Elo-logistic) vs HistGradientBoosting vs MLP (torch).
Time split: train = 2023-24 + 2024-25, test = 2025-26. Metrics: accuracy, log-loss, RPS.
Usage: python D:/Programming/claude/FM/src/train.py [--with-odds]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["H", "D", "A"]


def rps(y_true_idx, proba):
    """Ranked probability score over ordered outcomes H, D, A (lower better)."""
    cum_p = np.cumsum(proba, axis=1)
    cum_o = np.cumsum(np.eye(3)[y_true_idx], axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def report(name, y_idx, proba):
    pred = np.argmax(proba, axis=1)
    print(f"{name:28s} acc={accuracy_score(y_idx, pred):.4f} "
          f"logloss={log_loss(y_idx, proba, labels=[0,1,2]):.4f} rps={rps(y_idx, proba):.4f}")


def main():
    with_odds = "--with-odds" in sys.argv
    df = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    df = df[df["complete"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    print(f"complete matches: {len(df)}")

    test_mask = df["date"] >= "2025-08-01"
    train, test = df[~test_mask], df[test_mask]
    print(f"train {len(train)}, test {len(test)}")
    if len(test) < 50 or len(train) < 200:
        print("not enough data yet — rerun after scrape completes")
        return

    y_tr = train["result"].map(CLASSES.index).to_numpy()
    y_te = test["result"].map(CLASSES.index).to_numpy()

    drop = {"match_id", "date", "result", "home_goals", "away_goals", "complete"}
    odds_cols = {"b365h", "b365d", "b365a"}
    feats = [c for c in df.columns if c not in drop and (with_odds or c not in odds_cols)]
    X_tr, X_te = train[feats].to_numpy(dtype=float), test[feats].to_numpy(dtype=float)
    print(f"features: {len(feats)} (odds={'in' if with_odds else 'out'})")

    # baseline: majority class
    p = np.zeros((len(test), 3))
    p[:, :] = np.bincount(y_tr, minlength=3) / len(y_tr)
    report("majority/prior", y_te, p)

    # baseline: bookmaker odds -> implied probs
    inv = 1.0 / test[["b365h", "b365d", "b365a"]].to_numpy(dtype=float)
    report("bookmaker (B365)", y_te, inv / inv.sum(axis=1, keepdims=True))

    # baseline: Elo + home logistic
    elo_cols = ["elo_home", "elo_away", "form_pts_home", "form_pts_away"]
    lr = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(train[elo_cols], y_tr)
    report("Elo+form logistic", y_te, lr.predict_proba(test[elo_cols]))

    # GBDT on everything
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                        l2_regularization=1.0, random_state=7,
                                        validation_fraction=0.15, early_stopping=True)
    gb.fit(X_tr, y_tr)
    report("HistGradientBoosting", y_te, gb.predict_proba(X_te))

    # MLP (sklearn; torch 2.2 incompatible with numpy 2.4 on this machine)
    from sklearn.neural_network import MLPClassifier
    mlp = make_pipeline(
        SimpleImputer(), StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1.0, max_iter=2000,
                      early_stopping=True, validation_fraction=0.15, random_state=7))
    mlp.fit(X_tr, y_tr)
    report("MLP (sklearn 64-32)", y_te, mlp.predict_proba(X_te))


if __name__ == "__main__":
    main()
