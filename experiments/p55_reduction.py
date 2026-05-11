"""
Step 5.5: Dimensionality reduction + hand-feature fusion.

This is a control before H2/H3: can we improve the chosen anti-overfit probe
without changing token position or layer? All variants use default hidden
features (last_seq, last layer, dim=896) and CV=5 via splitting.split_data.

Compared candidates:
  - hidden only -> 5.3e baseline
  - hidden + scaled hand_features -> 5.3e
  - PCA/SVD(hidden) -> LR L2
  - PCA/SVD(hidden) + hand_features -> LR L2
  - PCA/SVD(hidden) -> 5.3e
  - PCA/SVD(hidden) + hand_features -> 5.3e

Dimensionality reducers and hand-feature scalers are fitted inside each
train fold only; val/test are transformed with the fitted fold pipeline.
Primary metric: test AUROC. Accuracy/F1 are shown only for reference.

Run from repo root:
    python -m experiments.p55_reduction
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.lr_handfeatures import build_features as build_hand_features  # noqa: E402
from experiments.p5_probe_regularization import (  # noqa: E402
    LRProbe,
    SmallerMLPProbe,
    extract_or_load_features,
    run_probe,
)
from splitting import split_data  # noqa: E402

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"
REDUCERS = ("pca", "svd")
N_COMPONENTS = (16, 32, 64, 128, 256)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChosenProbe(SmallerMLPProbe):
    """Step 5 winner: 5.3e combo."""

    def __init__(self):
        super().__init__(hidden=64, dropout=0.5, weight_decay=1e-2, epochs=50)


def make_lr(c: float = 0.1) -> LRProbe:
    """Best LR setting from step 5 on default hidden features."""
    return LRProbe(C=c)


class FeaturePipelineProbe:
    """Fold-local feature pipeline wrapped around an existing probe class.

    Input X is the concatenation [hidden, hand]. The wrapper can reduce hidden
    features, append scaled hand features, then delegate fit/predict to a base
    probe. All transformers are fitted only in fit().
    """

    def __init__(
        self,
        base_factory: Callable[[], object],
        hidden_dim: int,
        *,
        reducer: str | None = None,
        n_components: int | None = None,
        include_hand: bool = False,
    ):
        self._base_factory = base_factory
        self._hidden_dim = hidden_dim
        self._reducer_name = reducer
        self._n_components = n_components
        self._include_hand = include_hand
        self._reducer = None
        self._hand_scaler = StandardScaler()
        self._probe = None

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X[:, : self._hidden_dim], X[:, self._hidden_dim :]

    def _make_reducer(self, X_hidden: np.ndarray):
        if self._reducer_name is None:
            return None

        max_components = min(X_hidden.shape[0] - 1, X_hidden.shape[1])
        if self._n_components is None or self._n_components > max_components:
            raise ValueError(
                f"n_components={self._n_components} is invalid for "
                f"train hidden matrix {X_hidden.shape}"
            )

        if self._reducer_name == "pca":
            return PCA(
                n_components=self._n_components,
                svd_solver="randomized",
                random_state=SEED,
            )
        if self._reducer_name == "svd":
            return TruncatedSVD(n_components=self._n_components, random_state=SEED)
        raise ValueError(f"Unknown reducer: {self._reducer_name}")

    def _fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        X_hidden, X_hand = self._split(X)

        self._reducer = self._make_reducer(X_hidden)
        if self._reducer is None:
            parts = [X_hidden]
        else:
            parts = [self._reducer.fit_transform(X_hidden)]

        if self._include_hand:
            parts.append(self._hand_scaler.fit_transform(X_hand))

        return np.concatenate(parts, axis=1).astype(np.float32)

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        X_hidden, X_hand = self._split(X)

        if self._reducer is None:
            parts = [X_hidden]
        else:
            parts = [self._reducer.transform(X_hidden)]

        if self._include_hand:
            parts.append(self._hand_scaler.transform(X_hand))

        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(self, X, y):
        X_pipe = self._fit_transform_features(X)
        self._probe = self._base_factory()
        self._probe.fit(X_pipe, y)
        return self

    def predict_proba(self, X):
        return self._probe.predict_proba(self._transform_features(X))

    def predict(self, X):
        return self._probe.predict(self._transform_features(X))

    def fit_hyperparameters(self, X_val, y_val):
        if hasattr(self._probe, "fit_hyperparameters"):
            self._probe.fit_hyperparameters(self._transform_features(X_val), y_val)
        return self


def summarize_rows(results: dict[str, dict]) -> list[tuple]:
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
    return rows


def print_summary(results: dict[str, dict]) -> None:
    print(f"\n{'=' * 118}")
    print("  STEP 5.5 SUMMARY — sorted by test AUROC (mean ± std across 5 folds)")
    print(f"{'=' * 118}")
    print(
        f"  {'Variant':<38s} | "
        f"{'test AUROC':>16s} | {'gap (tr-te)':>11s} | "
        f"{'train AUROC':>14s} | {'val AUROC':>14s} | "
        f"{'test acc':>14s} | {'test f1':>14s}"
    )
    print("-" * 118)

    for name, tr, va, te, te_acc, te_f1, gap in summarize_rows(results):
        print(
            f"  {name:<38s} | "
            f"{te[0]:.4f} ± {te[1]:.3f}  | "
            f"{gap:>+10.4f} | "
            f"{tr[0]:.4f} ± {tr[1]:.3f}  | "
            f"{va[0]:.4f} ± {va[1]:.3f}  | "
            f"{te_acc[0]:.4f} ± {te_acc[1]:.3f}  | "
            f"{te_f1[0]:.4f} ± {te_f1[1]:.3f}"
        )


def main() -> None:
    set_seed(SEED)

    df = pd.read_csv(DATA_FILE)
    y = df["label"].astype(float).astype(int).values
    print(
        f"Loaded {len(df)} samples; balance: "
        f"{dict(pd.Series(y).value_counts(normalize=True).round(3))}"
    )

    X_hidden = extract_or_load_features(df)
    X_hand = build_hand_features(df)
    X_all = np.concatenate([X_hidden, X_hand], axis=1).astype(np.float32)

    print(f"Hidden features: {X_hidden.shape}")
    print(f"Hand features  : {X_hand.shape}")
    print(f"Combined input : {X_all.shape}")

    splits = split_data(y)
    print(f"Folds: {len(splits)}\n")

    hidden_dim = X_hidden.shape[1]
    variants: list[tuple[str, Callable[[], object]]] = [
        (
            "5.5a_hidden_only_MLP_combo",
            lambda: FeaturePipelineProbe(ChosenProbe, hidden_dim),
        ),
        (
            "5.5b_hidden+hand_MLP_combo",
            lambda: FeaturePipelineProbe(
                ChosenProbe, hidden_dim, include_hand=True
            ),
        ),
    ]

    for reducer in REDUCERS:
        for n_components in N_COMPONENTS:
            variants.extend([
                (
                    f"5.5_{reducer}{n_components}_LR",
                    lambda reducer=reducer, n_components=n_components: (
                        FeaturePipelineProbe(
                            make_lr,
                            hidden_dim,
                            reducer=reducer,
                            n_components=n_components,
                        )
                    ),
                ),
                (
                    f"5.5_{reducer}{n_components}+hand_LR",
                    lambda reducer=reducer, n_components=n_components: (
                        FeaturePipelineProbe(
                            make_lr,
                            hidden_dim,
                            reducer=reducer,
                            n_components=n_components,
                            include_hand=True,
                        )
                    ),
                ),
                (
                    f"5.5_{reducer}{n_components}_MLP_combo",
                    lambda reducer=reducer, n_components=n_components: (
                        FeaturePipelineProbe(
                            ChosenProbe,
                            hidden_dim,
                            reducer=reducer,
                            n_components=n_components,
                        )
                    ),
                ),
                (
                    f"5.5_{reducer}{n_components}+hand_MLP_combo",
                    lambda reducer=reducer, n_components=n_components: (
                        FeaturePipelineProbe(
                            ChosenProbe,
                            hidden_dim,
                            reducer=reducer,
                            n_components=n_components,
                            include_hand=True,
                        )
                    ),
                ),
            ])

    results = {}
    for name, factory in variants:
        print(f"  -> {name}")
        results[name] = run_probe(factory, X_all, y, splits)

    print_summary(results)


if __name__ == "__main__":
    main()
