"""
probe.py — Hallucination probe with H4-style feature pipeline.

Pipeline (matches experiments/p7_threshold_tuning.py, FinalH6Model):
    raw H4 grid (14336 dim)
      -> StandardScaler (fit on train fold)
      -> PCA(16)        (fit on train fold)
      -> + Mass-Mean score (theta = mean(X+) - mean(X-) on scaled grid, normalized)
      -> 5.3e MLP head: Linear -> ReLU -> Dropout(0.5) -> Linear, 50 epochs,
                       Adam(lr=1e-3, weight_decay=1e-2), BCE with pos_weight.

Threshold defaults to OOF_TUNED_THRESHOLD (derived in p7 on tokenizer-aware
features); fit_hyperparameters() overrides it via val-set accuracy tuning when
called by evaluate.run_evaluation per-fold.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


SEED = 42
HIDDEN = 64
DROPOUT = 0.5
WEIGHT_DECAY = 1e-2
EPOCHS = 50
LR = 1e-3
PCA_COMPONENTS = 16
OOF_TUNED_THRESHOLD = 0.421707


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._grid_scaler = StandardScaler()
        self._pca: PCA | None = None
        self._mm_theta: np.ndarray | None = None
        self._mm_mu: float = 0.0
        self._mm_sigma: float = 1.0
        self._net: nn.Sequential | None = None
        self._threshold: float = OOF_TUNED_THRESHOLD

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )

    def _fit_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_scaled = self._grid_scaler.fit_transform(X)
        self._pca = PCA(
            n_components=PCA_COMPONENTS,
            svd_solver="randomized",
            random_state=SEED,
        )
        X_pca = self._pca.fit_transform(X_scaled)

        pos = X_scaled[y == 1].mean(axis=0)
        neg = X_scaled[y == 0].mean(axis=0)
        theta = pos - neg
        self._mm_theta = theta / (np.linalg.norm(theta) + 1e-12)

        raw = X_scaled @ self._mm_theta
        self._mm_mu = float(raw.mean())
        self._mm_sigma = float(raw.std() + 1e-12)
        mm = ((raw - self._mm_mu) / self._mm_sigma).reshape(-1, 1)
        return np.concatenate([X_pca, mm], axis=1).astype(np.float32)

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._grid_scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)
        raw = X_scaled @ self._mm_theta
        mm = ((raw - self._mm_mu) / self._mm_sigma).reshape(-1, 1)
        return np.concatenate([X_pca, mm], axis=1).astype(np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Network not built. Call fit() before forward().")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_final = self._fit_features(X, y.astype(int))

        torch.manual_seed(SEED)
        self._build_network(X_final.shape[1])

        X_t = torch.from_numpy(X_final).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        )

        self.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            logits = self(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray,
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_acc = OOF_TUNED_THRESHOLD, -1.0
        for t in candidates:
            acc = accuracy_score(y_val, (probs >= t).astype(int))
            if acc > best_acc:
                best_acc, best_t = acc, float(t)
        self._threshold = best_t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_final = self._transform_features(X)
        X_t = torch.from_numpy(X_final).float()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
