"""
H7: probe-head sweep on the H6 winning representation.

Features are fixed:
  H4 chunk-layer grid -> fold-local PCA16 -> Mass-Mean score

Only the classifier head changes. The goal is to check whether the current
5.3e head (hidden=64, dropout=0.5, weight_decay=1e-2, 50 epochs) is oversized
for the compact ~17-dim H6 feature space.

Run from repo root:
    python -m experiments.h7_probe_head_sweep
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.h4_chunk_layer_pca import extract_or_load_grid as extract_or_load_h4_grid  # noqa: E402
from splitting import split_data  # noqa: E402

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"
N_COMPONENTS = 16


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tune_threshold_for_accuracy(probs: np.ndarray, y_val: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_acc = 0.5, -1.0
    for t in candidates:
        acc = accuracy_score(y_val, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


class TunedMLPProbe(nn.Module):
    """One-hidden-layer MLP with configurable regularization."""

    def __init__(
        self,
        hidden: int = 64,
        dropout: float = 0.5,
        weight_decay: float = 1e-2,
        epochs: int = 50,
        lr: float = 1e-3,
    ):
        super().__init__()
        self._scaler = StandardScaler()
        self._net: nn.Sequential | None = None
        self._threshold = 0.5
        self._hidden = hidden
        self._dropout = dropout
        self._weight_decay = weight_decay
        self._epochs = epochs
        self._lr = lr

    def fit(self, X, y):
        X_scaled = self._scaler.fit_transform(X)
        torch.manual_seed(SEED)

        layers: list[nn.Module] = [
            nn.Linear(X_scaled.shape[1], self._hidden),
            nn.ReLU(),
        ]
        if self._dropout > 0:
            layers.append(nn.Dropout(self._dropout))
        layers.append(nn.Linear(self._hidden, 1))
        self._net = nn.Sequential(*layers)

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        self._net.train()
        for _ in range(self._epochs):
            optimizer.zero_grad()
            logits = self._net(X_t).squeeze(-1)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        self._net.eval()
        return self

    def predict_proba(self, X):
        X_scaled = self._scaler.transform(X)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(X_scaled).float()).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - probs, probs], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def fit_hyperparameters(self, X_val, y_val):
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = tune_threshold_for_accuracy(probs, y_val)
        return self


class LRProbe:
    """sklearn LogisticRegression sanity check on compact H6 features."""

    def __init__(self, C: float = 1.0):
        self._scaler = StandardScaler()
        self._lr = LogisticRegression(
            penalty="l2",
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
        )
        self._threshold = 0.5

    def fit(self, X, y):
        X_scaled = self._scaler.fit_transform(X)
        self._lr.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        X_scaled = self._scaler.transform(X)
        return self._lr.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self._threshold).astype(int)

    def fit_hyperparameters(self, X_val, y_val):
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = tune_threshold_for_accuracy(probs, y_val)
        return self


class H6FeatureProbe:
    """Fold-local H4-grid PCA16 + Mass-Mean score, then a classifier head."""

    def __init__(self, head_factory: Callable[[], object]):
        self._head_factory = head_factory
        self._grid_scaler = StandardScaler()
        self._pca: PCA | None = None
        self._mm_theta: np.ndarray | None = None
        self._mm_mu = 0.0
        self._mm_sigma = 1.0
        self._head = None

    def _fit_mm(self, X_grid_scaled: np.ndarray, y: np.ndarray) -> np.ndarray:
        pos = X_grid_scaled[y == 1].mean(axis=0)
        neg = X_grid_scaled[y == 0].mean(axis=0)
        theta = pos - neg
        self._mm_theta = theta / (np.linalg.norm(theta) + 1e-12)
        raw = X_grid_scaled @ self._mm_theta
        self._mm_mu = float(raw.mean())
        self._mm_sigma = float(raw.std() + 1e-12)
        return ((raw - self._mm_mu) / self._mm_sigma).reshape(-1, 1)

    def _transform_mm(self, X_grid_scaled: np.ndarray) -> np.ndarray:
        raw = X_grid_scaled @ self._mm_theta
        return ((raw - self._mm_mu) / self._mm_sigma).reshape(-1, 1)

    def _fit_transform_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_scaled = self._grid_scaler.fit_transform(X)
        self._pca = PCA(
            n_components=N_COMPONENTS,
            svd_solver="randomized",
            random_state=SEED,
        )
        X_pca = self._pca.fit_transform(X_scaled)
        X_mm = self._fit_mm(X_scaled, y)
        return np.concatenate([X_pca, X_mm], axis=1).astype(np.float32)

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._grid_scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)
        X_mm = self._transform_mm(X_scaled)
        return np.concatenate([X_pca, X_mm], axis=1).astype(np.float32)

    def fit(self, X, y):
        X_h6 = self._fit_transform_features(X, y)
        self._head = self._head_factory()
        self._head.fit(X_h6, y)
        return self

    def predict_proba(self, X):
        return self._head.predict_proba(self._transform_features(X))

    def predict(self, X):
        return self._head.predict(self._transform_features(X))

    def fit_hyperparameters(self, X_val, y_val):
        if hasattr(self._head, "fit_hyperparameters"):
            self._head.fit_hyperparameters(self._transform_features(X_val), y_val)
        return self


def metrics_for_split(probe, X, y, idx):
    if idx is None:
        return None
    y_true = y[idx]
    y_pred = probe.predict(X[idx])
    y_prob = probe.predict_proba(X[idx])[:, 1]
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": auroc,
    }


def run_probe(probe_factory: Callable[[], object], X, y, splits) -> dict:
    fold_metrics = []
    for idx_train, idx_val, idx_test in splits:
        probe = probe_factory()
        probe.fit(X[idx_train], y[idx_train])
        if idx_val is not None and hasattr(probe, "fit_hyperparameters"):
            probe.fit_hyperparameters(X[idx_val], y[idx_val])
        fold_metrics.append({
            "train": metrics_for_split(probe, X, y, idx_train),
            "val": metrics_for_split(probe, X, y, idx_val),
            "test": metrics_for_split(probe, X, y, idx_test),
        })

    def avg(split, key):
        vals = np.array([fm[split][key] for fm in fold_metrics])
        return float(vals.mean()), float(vals.std())

    return {
        "train": {k: avg("train", k) for k in ["acc", "f1", "auroc"]},
        "val": {k: avg("val", k) for k in ["acc", "f1", "auroc"]},
        "test": {k: avg("test", k) for k in ["acc", "f1", "auroc"]},
    }


def print_summary(results: dict[str, dict]) -> None:
    print(f"\n{'=' * 112}")
    print("  H7 SUMMARY - sorted by test AUROC (mean +/- std across 5 folds)")
    print(f"{'=' * 112}")
    print(
        f"  {'Variant':<38s} | "
        f"{'test AUROC':>16s} | {'gap (tr-te)':>11s} | "
        f"{'train AUROC':>14s} | {'val AUROC':>14s} | "
        f"{'test acc':>14s} | {'test f1':>14s}"
    )
    print("-" * 112)

    rows = []
    for name, r in results.items():
        tr = r["train"]["auroc"]
        va = r["val"]["auroc"]
        te = r["test"]["auroc"]
        te_acc = r["test"]["acc"]
        te_f1 = r["test"]["f1"]
        gap = tr[0] - te[0]
        rows.append((name, tr, va, te, te_acc, te_f1, gap))

    rows.sort(key=lambda row: row[3][0], reverse=True)
    for name, tr, va, te, te_acc, te_f1, gap in rows:
        print(
            f"  {name:<38s} | "
            f"{te[0]:.4f} +/- {te[1]:.3f} | "
            f"{gap:>+10.4f} | "
            f"{tr[0]:.4f} +/- {tr[1]:.3f} | "
            f"{va[0]:.4f} +/- {va[1]:.3f} | "
            f"{te_acc[0]:.4f} +/- {te_acc[1]:.3f} | "
            f"{te_f1[0]:.4f} +/- {te_f1[1]:.3f}"
        )


def main() -> None:
    set_seed(SEED)

    df = pd.read_csv(DATA_FILE)
    y = df["label"].astype(float).astype(int).values
    print(
        f"Loaded {len(df)} samples; balance: "
        f"{dict(pd.Series(y).value_counts(normalize=True).round(3))}"
    )

    X_grid, _ = extract_or_load_h4_grid(df)
    print(f"H4 grid features: {X_grid.shape}")

    splits = split_data(y)
    print(f"Folds: {len(splits)}\n")

    variants: list[tuple[str, Callable[[], object]]] = [
        (
            "h7_current_64_do0.5_wd1e-2_50ep",
            lambda: H6FeatureProbe(
                lambda: TunedMLPProbe(
                    hidden=64, dropout=0.5, weight_decay=1e-2, epochs=50
                )
            ),
        ),
        (
            "h7_mlp8_do0.2_wd1e-2_50ep",
            lambda: H6FeatureProbe(
                lambda: TunedMLPProbe(
                    hidden=8, dropout=0.2, weight_decay=1e-2, epochs=50
                )
            ),
        ),
        (
            "h7_mlp16_do0.2_wd1e-2_50ep",
            lambda: H6FeatureProbe(
                lambda: TunedMLPProbe(
                    hidden=16, dropout=0.2, weight_decay=1e-2, epochs=50
                )
            ),
        ),
        (
            "h7_mlp16_do0_wd1e-2_50ep",
            lambda: H6FeatureProbe(
                lambda: TunedMLPProbe(
                    hidden=16, dropout=0.0, weight_decay=1e-2, epochs=50
                )
            ),
        ),
        (
            "h7_mlp32_do0.2_wd1e-2_50ep",
            lambda: H6FeatureProbe(
                lambda: TunedMLPProbe(
                    hidden=32, dropout=0.2, weight_decay=1e-2, epochs=50
                )
            ),
        ),
        (
            "h7_mlp16_do0.2_wd1e-3_50ep",
            lambda: H6FeatureProbe(
                lambda: TunedMLPProbe(
                    hidden=16, dropout=0.2, weight_decay=1e-3, epochs=50
                )
            ),
        ),
        (
            "h7_mlp16_do0.2_wd1e-2_100ep",
            lambda: H6FeatureProbe(
                lambda: TunedMLPProbe(
                    hidden=16, dropout=0.2, weight_decay=1e-2, epochs=100
                )
            ),
        ),
        (
            "h7_lr_C0.1",
            lambda: H6FeatureProbe(lambda: LRProbe(C=0.1)),
        ),
        (
            "h7_lr_C1.0",
            lambda: H6FeatureProbe(lambda: LRProbe(C=1.0)),
        ),
    ]

    results = {}
    for name, factory in variants:
        print(f"  -> {name}")
        results[name] = run_probe(factory, X_grid, y, splits)

    print_summary(results)


if __name__ == "__main__":
    main()
