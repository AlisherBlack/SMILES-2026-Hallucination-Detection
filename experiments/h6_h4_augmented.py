"""
H6: augment the H4 PCA16 representation.

H4 winner:
  response chunks x layers -> fold-local PCA16 -> 5.3e combo probe

This experiment tests whether weak standalone signals add value on top of H4:
  - scaled hand-crafted text features
  - H3 topological/eigen features (top-10 log eigenvalues + pseudo-logdet)
  - Mass-Mean score from PRISM truthfulness direction

All transforms are fold-local. In particular, PCA/scalers/Mass-Mean theta are
fit on the train fold only; val/test are transformed with the fitted pipeline.

Run from repo root:
    python -m experiments.h6_h4_augmented
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.h3_eigen_features import (  # noqa: E402
    extract_or_load_features as extract_or_load_h3_features,
    select_eigen_columns,
)
from experiments.h4_chunk_layer_pca import (  # noqa: E402
    ChosenProbe,
    extract_or_load_grid as extract_or_load_h4_grid,
    run_probe,
    set_seed,
)
from experiments.lr_handfeatures import build_features as build_hand_features  # noqa: E402
from splitting import split_data  # noqa: E402

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"
N_COMPONENTS = 16


class H4AugmentedPipelineProbe:
    """Fold-local H4 PCA + optional hand/topological/MM features."""

    def __init__(
        self,
        base_factory: Callable[[], object],
        grid_dim: int,
        hand_dim: int,
        topo_dim: int,
        *,
        n_components: int = N_COMPONENTS,
        include_hand: bool = False,
        include_topo: bool = False,
        include_mm: bool = False,
    ):
        self._base_factory = base_factory
        self._grid_dim = grid_dim
        self._hand_dim = hand_dim
        self._topo_dim = topo_dim
        self._n_components = n_components
        self._include_hand = include_hand
        self._include_topo = include_topo
        self._include_mm = include_mm

        self._grid_scaler = StandardScaler()
        self._hand_scaler = StandardScaler()
        self._topo_scaler = StandardScaler()
        self._pca: PCA | None = None
        self._mm_theta: np.ndarray | None = None
        self._mm_mu = 0.0
        self._mm_sigma = 1.0
        self._probe = None

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hand_start = self._grid_dim
        topo_start = self._grid_dim + self._hand_dim
        X_grid = X[:, : self._grid_dim]
        X_hand = X[:, hand_start:topo_start]
        X_topo = X[:, topo_start : topo_start + self._topo_dim]
        return X_grid, X_hand, X_topo

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
        X_grid, X_hand, X_topo = self._split(X)

        X_grid_scaled = self._grid_scaler.fit_transform(X_grid)
        self._pca = PCA(
            n_components=self._n_components,
            svd_solver="randomized",
            random_state=SEED,
        )
        parts = [self._pca.fit_transform(X_grid_scaled)]

        if self._include_hand:
            parts.append(self._hand_scaler.fit_transform(X_hand))
        if self._include_topo:
            parts.append(self._topo_scaler.fit_transform(X_topo))
        if self._include_mm:
            parts.append(self._fit_mm(X_grid_scaled, y))

        return np.concatenate(parts, axis=1).astype(np.float32)

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        X_grid, X_hand, X_topo = self._split(X)

        X_grid_scaled = self._grid_scaler.transform(X_grid)
        parts = [self._pca.transform(X_grid_scaled)]

        if self._include_hand:
            parts.append(self._hand_scaler.transform(X_hand))
        if self._include_topo:
            parts.append(self._topo_scaler.transform(X_topo))
        if self._include_mm:
            parts.append(self._transform_mm(X_grid_scaled))

        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(self, X, y):
        X_aug = self._fit_transform_features(X, y)
        self._probe = self._base_factory()
        self._probe.fit(X_aug, y)
        return self

    def predict_proba(self, X):
        return self._probe.predict_proba(self._transform_features(X))

    def predict(self, X):
        return self._probe.predict(self._transform_features(X))

    def fit_hyperparameters(self, X_val, y_val):
        if hasattr(self._probe, "fit_hyperparameters"):
            self._probe.fit_hyperparameters(self._transform_features(X_val), y_val)
        return self


def make_factory(
    grid_dim: int,
    hand_dim: int,
    topo_dim: int,
    *,
    include_hand: bool = False,
    include_topo: bool = False,
    include_mm: bool = False,
):
    return lambda: H4AugmentedPipelineProbe(
        ChosenProbe,
        grid_dim,
        hand_dim,
        topo_dim,
        include_hand=include_hand,
        include_topo=include_topo,
        include_mm=include_mm,
    )


def print_summary(results: dict[str, dict]) -> None:
    print(f"\n{'=' * 112}")
    print("  H6 SUMMARY - sorted by test AUROC (mean +/- std across 5 folds)")
    print(f"{'=' * 112}")
    print(
        f"  {'Variant':<34s} | "
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
            f"  {name:<34s} | "
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
    X_hand = build_hand_features(df)
    _, X_eigen_full, _ = extract_or_load_h3_features(df)
    X_topo = select_eigen_columns(X_eigen_full, 10)
    X_all = np.concatenate([X_grid, X_hand, X_topo], axis=1).astype(np.float32)

    grid_dim = X_grid.shape[1]
    hand_dim = X_hand.shape[1]
    topo_dim = X_topo.shape[1]
    print(f"H4 grid features : {X_grid.shape}")
    print(f"Hand features    : {X_hand.shape}")
    print(f"Topo features    : {X_topo.shape}")
    print(f"Combined input   : {X_all.shape}")

    splits = split_data(y)
    print(f"Folds: {len(splits)}\n")

    variants: list[tuple[str, Callable[[], object]]] = [
        (
            "h6_h4_pca16",
            make_factory(grid_dim, hand_dim, topo_dim),
        ),
        (
            "h6_h4_pca16+hand",
            make_factory(grid_dim, hand_dim, topo_dim, include_hand=True),
        ),
        (
            "h6_h4_pca16+topo",
            make_factory(grid_dim, hand_dim, topo_dim, include_topo=True),
        ),
        (
            "h6_h4_pca16+mm",
            make_factory(grid_dim, hand_dim, topo_dim, include_mm=True),
        ),
        (
            "h6_h4_pca16+hand+topo",
            make_factory(
                grid_dim,
                hand_dim,
                topo_dim,
                include_hand=True,
                include_topo=True,
            ),
        ),
        (
            "h6_h4_pca16+hand+mm",
            make_factory(
                grid_dim,
                hand_dim,
                topo_dim,
                include_hand=True,
                include_mm=True,
            ),
        ),
        (
            "h6_h4_pca16+topo+mm",
            make_factory(
                grid_dim,
                hand_dim,
                topo_dim,
                include_topo=True,
                include_mm=True,
            ),
        ),
        (
            "h6_h4_pca16+hand+topo+mm",
            make_factory(
                grid_dim,
                hand_dim,
                topo_dim,
                include_hand=True,
                include_topo=True,
                include_mm=True,
            ),
        ),
    ]

    results = {}
    for name, factory in variants:
        print(f"  -> {name}")
        results[name] = run_probe(factory, X_all, y, splits)

    print_summary(results)


if __name__ == "__main__":
    main()
