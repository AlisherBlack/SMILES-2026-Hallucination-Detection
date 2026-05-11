"""
Baseline experiment: Logistic Regression on hand-crafted text features.

Computes shallow surface features from the response (length, digits, etc.),
trains LR with `class_weight='balanced'` over the same K-fold splits used by
the probe pipeline (`splitting.split_data`), and reports per-fold metrics.

Run from repo root:
    python -m experiments.lr_handfeatures
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from splitting import split_data  # noqa: E402

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def hand_features(prompt: str, response: str) -> dict:
    r = str(response)
    p = str(prompt)

    response_chars = len(r)
    response_words = len(r.split())
    prompt_chars = len(p)

    return {
        "response_chars": response_chars,
        "response_words": response_words,
        "prompt_chars": prompt_chars,
        # new features
        "response_to_prompt_chars": response_chars / max(1, prompt_chars),
        "has_number": int(any(c.isdigit() for c in r)),
        "ends_with_period": int(r.strip().endswith(".")),
        # existing features
        "n_digits": sum(c.isdigit() for c in r),
        "n_uppercase_words": len(re.findall(r"\b[A-Z][a-zA-Z]+\b", r)),
        "n_punct": sum(c in ".,;:!?" for c in r),
        "n_quotes": r.count('"') + r.count("'"),
        "log_resp_words": float(np.log1p(response_words)),
    }


def build_features(df: pd.DataFrame) -> np.ndarray:
    rows = [hand_features(r["prompt"], r["response"]) for _, r in df.iterrows()]
    return pd.DataFrame(rows).values.astype(np.float32)


def make_lr() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )


def evaluate(X: np.ndarray, y: np.ndarray, splits) -> list[dict]:
    fold_results = []
    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits, start=1):
        idx_fit = (
            np.concatenate([idx_train, idx_val]) if idx_val is not None else idx_train
        )

        dummy = DummyClassifier(strategy="most_frequent").fit(X[idx_fit], y[idx_fit])
        y_dummy = dummy.predict(X[idx_test])
        baseline_acc = accuracy_score(y[idx_test], y_dummy)
        baseline_f1 = f1_score(y[idx_test], y_dummy, zero_division=0)

        lr = make_lr().fit(X[idx_fit], y[idx_fit])
        y_pred = lr.predict(X[idx_test])
        y_prob = lr.predict_proba(X[idx_test])[:, 1]

        acc = accuracy_score(y[idx_test], y_pred)
        f1 = f1_score(y[idx_test], y_pred, zero_division=0)
        auc = roc_auc_score(y[idx_test], y_prob)

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(idx_fit),
            "n_test": len(idx_test),
            "baseline_acc": baseline_acc,
            "baseline_f1": baseline_f1,
            "lr_acc": acc,
            "lr_f1": f1,
            "lr_auc": auc,
        })
        print(
            f"  Fold {fold_idx}: baseline_acc={baseline_acc:.4f}  "
            f"lr_acc={acc:.4f}  lr_f1={f1:.4f}  lr_auc={auc:.4f}"
        )
    return fold_results


def summarize(fold_results: list[dict]) -> dict:
    def stat(key):
        vals = np.array([r[key] for r in fold_results], dtype=float)
        return float(vals.mean()), float(vals.std())

    keys = ["baseline_acc", "baseline_f1", "lr_acc", "lr_f1", "lr_auc"]
    return {k: stat(k) for k in keys}


def main() -> None:
    set_seed(SEED)

    df = pd.read_csv(DATA_FILE)
    y = df["label"].astype(float).astype(int).values
    print(
        f"Loaded {len(df)} samples. Label balance: "
        f"{dict(pd.Series(y).value_counts(normalize=True).round(3))}"
    )

    X = build_features(df)
    print(f"Feature matrix: {X.shape}")

    splits = split_data(y)
    print(f"Splits: {len(splits)} folds (StratifiedKFold)")

    fold_results = evaluate(X, y, splits)
    summary = summarize(fold_results)

    print("\n=== Summary (mean ± std across folds) ===")
    for k, (mean, std) in summary.items():
        print(f"  {k:14s}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
