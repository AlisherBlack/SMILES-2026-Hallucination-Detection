"""
Step 5: Probe regularization sweep.

Goal: pick a probe that doesn't overfit before running feature-level
hypotheses (step 6). Current MLP gives train AUROC 0.9999 / test 0.7361 —
comparisons of token/layer/geom features under it are unreliable.

Variants on default features (last_seq, last layer of Qwen2.5-0.5B, dim=896),
CV=5 via splitting.split_data. Comparison metric: **test AUROC** (threshold-
independent). Test acc/F1 are reported for reference but not used to compare.

  5.0  HallucinationProbe                — baseline (MLP 896→256→1, 200 ep)
  5.1  LRProbe (sklearn LR L2)           — C ∈ {0.01, 0.1, 1.0, 10.0}
  5.2  LinearAdamProbe (logreg via Adam) — weight_decay ∈ {0, 1e-2}
  5.3  Regularized MLP — one factor at a time vs baseline:
       a. + dropout 0.5     (hidden 256, wd=0,    200 ep)
       b. + weight_decay 1e-2  (hidden 256, dropout=0, 200 ep)
       c. epochs 200 → 50   (hidden 256, no dropout, no wd)
       d. hidden 256 → 64   (no dropout, no wd, 200 ep)
       e. combo             (hidden 64 + dropout 0.5 + wd 1e-2 + 50 ep)
       f. combo + BatchNorm (e + nn.BatchNorm1d after the first linear)
  5.4  MassMeanProbe (PRISM)             — no training

Features cached to data/cache/last_seq_last_layer.npz for reuse in step 6.

Run from repo root:
    python -m experiments.p5_probe_regularization
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import MAX_LENGTH, get_model_and_tokenizer  # noqa: E402
from probe import HallucinationProbe  # noqa: E402
from splitting import split_data  # noqa: E402

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"
CACHE_DIR = REPO_ROOT / "data" / "cache"
CACHE_FILE = CACHE_DIR / "last_seq_last_layer.npz"
BATCH_SIZE = 4


def tune_threshold_for_accuracy(probs: np.ndarray, y_val: np.ndarray) -> float:
    """Pick threshold maximising val accuracy."""
    candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
    best_t, best_acc = 0.5, -1.0
    for t in candidates:
        acc = accuracy_score(y_val, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Feature extraction (with disk cache)
# --------------------------------------------------------------------------


def extract_or_load_features(df: pd.DataFrame) -> np.ndarray:
    if CACHE_FILE.exists():
        print(f"Loading cached features from {CACHE_FILE}")
        return np.load(CACHE_FILE)["X"].astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache miss — running Qwen forward pass.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    feats: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting"):
        batch = texts[start : start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        hidden = out.hidden_states[-1].float().cpu()
        mask_cpu = mask.cpu()
        for i in range(hidden.size(0)):
            n_real = int(mask_cpu[i].sum().item())
            feats.append(hidden[i, n_real - 1].numpy())

    X = np.stack(feats).astype(np.float32)
    np.savez(CACHE_FILE, X=X)
    print(f"Saved features to {CACHE_FILE} ({X.shape})")
    return X


# --------------------------------------------------------------------------
# Probe variants
# --------------------------------------------------------------------------


class LRProbe:
    """sklearn LogisticRegression with L2 penalty."""

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
        Xs = self._scaler.fit_transform(X)
        self._lr.fit(Xs, y)
        return self

    def predict_proba(self, X):
        Xs = self._scaler.transform(X)
        return self._lr.predict_proba(Xs)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self._threshold).astype(int)

    def fit_hyperparameters(self, X_val, y_val):
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = tune_threshold_for_accuracy(probs, y_val)
        return self


class LinearAdamProbe(nn.Module):
    """Single linear layer + BCE, trained with Adam (logreg via Adam)."""

    def __init__(self, weight_decay: float = 1e-3, epochs: int = 200, lr: float = 1e-3):
        super().__init__()
        self._scaler = StandardScaler()
        self._linear: nn.Linear | None = None
        self._threshold = 0.5
        self._wd = weight_decay
        self._epochs = epochs
        self._lr = lr

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        torch.manual_seed(SEED)
        self._linear = nn.Linear(Xs.shape[1], 1)

        X_t = torch.from_numpy(Xs).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self._linear.parameters(),
            lr=self._lr,
            weight_decay=self._wd,
        )

        self._linear.train()
        for _ in range(self._epochs):
            optimizer.zero_grad()
            logits = self._linear(X_t).squeeze(-1)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
        self._linear.eval()
        return self

    def predict_proba(self, X):
        Xs = self._scaler.transform(X)
        with torch.no_grad():
            logits = self._linear(torch.from_numpy(Xs).float()).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
        return np.stack([1 - probs, probs], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def fit_hyperparameters(self, X_val, y_val):
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = tune_threshold_for_accuracy(probs, y_val)
        return self


class SmallerMLPProbe(nn.Module):
    """MLP 896→hidden→1 with optional dropout / weight decay / batch norm."""

    def __init__(
        self,
        hidden: int = 64,
        dropout: float = 0.5,
        weight_decay: float = 1e-3,
        epochs: int = 50,
        lr: float = 1e-3,
        use_bn: bool = False,
    ):
        super().__init__()
        self._scaler = StandardScaler()
        self._net: nn.Sequential | None = None
        self._threshold = 0.5
        self._hidden = hidden
        self._dropout = dropout
        self._wd = weight_decay
        self._epochs = epochs
        self._lr = lr
        self._use_bn = use_bn

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        torch.manual_seed(SEED)
        layers: list[nn.Module] = [nn.Linear(Xs.shape[1], self._hidden)]
        if self._use_bn:
            layers.append(nn.BatchNorm1d(self._hidden))
        layers += [
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._hidden, 1),
        ]
        self._net = nn.Sequential(*layers)

        X_t = torch.from_numpy(Xs).float()
        y_t = torch.from_numpy(y.astype(np.float32))
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self._lr,
            weight_decay=self._wd,
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
        Xs = self._scaler.transform(X)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(Xs).float()).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
        return np.stack([1 - probs, probs], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def fit_hyperparameters(self, X_val, y_val):
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = tune_threshold_for_accuracy(probs, y_val)
        return self


class MassMeanProbe:
    """Mass-Mean: theta = mean(X+) - mean(X-), score = X · theta / ||theta||."""

    def fit(self, X, y):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        pos = Xs[y == 1].mean(axis=0)
        neg = Xs[y == 0].mean(axis=0)
        theta = pos - neg
        self._theta_norm = theta / (np.linalg.norm(theta) + 1e-12)
        # Calibrate score scale: standardize raw scores on train so sigmoid is meaningful
        raw = Xs @ self._theta_norm
        self._mu = float(raw.mean())
        self._sigma = float(raw.std() + 1e-12)
        self._threshold = 0.5
        return self

    def predict_proba(self, X):
        Xs = self._scaler.transform(X)
        raw = Xs @ self._theta_norm
        z = (raw - self._mu) / self._sigma
        probs = expit(z)
        return np.stack([1 - probs, probs], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def fit_hyperparameters(self, X_val, y_val):
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = tune_threshold_for_accuracy(probs, y_val)
        return self


# --------------------------------------------------------------------------
# Quiet evaluation loop (no per-fold prints)
# --------------------------------------------------------------------------


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


def run_probe(probe_factory, X, y, splits) -> dict:
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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    set_seed(SEED)

    df = pd.read_csv(DATA_FILE)
    y = np.array([int(float(v)) for v in df["label"]])
    print(
        f"Loaded {len(df)} samples; balance: "
        f"{dict(pd.Series(y).value_counts(normalize=True).round(3))}"
    )

    X = extract_or_load_features(df)
    print(f"Feature matrix: {X.shape}")

    splits = split_data(y)
    print(f"Folds: {len(splits)}\n")

    variants: list[tuple[str, callable]] = [
        # ("5.0_baseline_MLP_256_200ep", lambda: HallucinationProbe()),
        # ("5.1a_LR_C0.01", lambda: LRProbe(C=0.01)),
        # ("5.1b_LR_C0.1", lambda: LRProbe(C=0.1)),
        # ("5.1c_LR_C1.0", lambda: LRProbe(C=1.0)),
        # ("5.1d_LR_C10", lambda: LRProbe(C=10.0)),
        # ("5.2a_LinAdam_wd0", lambda: LinearAdamProbe(weight_decay=0.0)),
        # ("5.2b_LinAdam_wd1e-2", lambda: LinearAdamProbe(weight_decay=1e-2)),
        (
            "5.3a_MLP+dropout0.5",
            lambda: SmallerMLPProbe(
                hidden=256, dropout=0.5, weight_decay=0.0, epochs=200
            ),
        ),
        (
            "5.3b_MLP+wd1e-2",
            lambda: SmallerMLPProbe(
                hidden=256, dropout=0.0, weight_decay=1e-2, epochs=200
            ),
        ),
        (
            "5.3c_MLP_50ep",
            lambda: SmallerMLPProbe(
                hidden=256, dropout=0.0, weight_decay=0.0, epochs=50
            ),
        ),
        (
            "5.3d_MLP_hidden64",
            lambda: SmallerMLPProbe(
                hidden=64, dropout=0.0, weight_decay=0.0, epochs=200
            ),
        ),
        (
            "5.3e_MLP_combo",
            lambda: SmallerMLPProbe(
                hidden=64, dropout=0.5, weight_decay=1e-2, epochs=50
            ),
        ),
        (
            "5.3f_MLP_combo+BN",
            lambda: SmallerMLPProbe(
                hidden=64, dropout=0.5, weight_decay=1e-2, epochs=50, use_bn=True
            ),
        ),
        # ("5.4_MassMean", lambda: MassMeanProbe()),
    ]

    results = {}
    for name, factory in variants:
        print(f"  → {name}")
        results[name] = run_probe(factory, X, y, splits)

    # Summary — sort by test AUROC (primary comparison metric).
    print(f"\n{'=' * 110}")
    print("  STEP 5 SUMMARY — sorted by test AUROC (mean ± std across 5 folds)")
    print(f"{'=' * 110}")
    print(
        f"  {'Variant':<28s} | "
        f"{'test AUROC':>16s} | {'gap (tr-te)':>11s} | "
        f"{'train AUROC':>14s} | {'val AUROC':>14s} | "
        f"{'test acc':>14s} | {'test f1':>14s}"
    )
    print("-" * 110)
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
            f"  {name:<28s} | "
            f"{te[0]:.4f} ± {te[1]:.3f}  | "
            f"{gap:>+10.4f} | "
            f"{tr[0]:.4f} ± {tr[1]:.3f}  | "
            f"{va[0]:.4f} ± {va[1]:.3f}  | "
            f"{te_acc[0]:.4f} ± {te_acc[1]:.3f}  | "
            f"{te_f1[0]:.4f} ± {te_f1[1]:.3f}"
        )


if __name__ == "__main__":
    main()
