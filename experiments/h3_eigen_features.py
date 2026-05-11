"""
H3: Eigenvalue geometric features (INSIDE-style).

Hypothesis: hidden states over response tokens carry a geometric signal beyond
the default last-token representation. For each sample we keep the baseline
feature (last_seq token, last layer) and append top-k log-eigenvalues of the
response-token covariance matrix plus a pseudo log-determinant.

Probe is the current feature-pipeline winner: hidden-only 5.3e combo
(SmallerMLPProbe hidden=64, dropout=0.5, weight_decay=1e-2, epochs=50).
Token position and layer are kept at baseline. Comparison metric is test AUROC;
accuracy/F1 are reported only for reference.

Run from repo root:
    python -m experiments.h3_eigen_features
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
from scipy.linalg import eigvalsh
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import MAX_LENGTH, get_model_and_tokenizer  # noqa: E402
from splitting import split_data  # noqa: E402

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"
CACHE_DIR = REPO_ROOT / "data" / "cache"
CACHE_FILE = CACHE_DIR / "h3_last_layer_eigen.npz"
BATCH_SIZE = 4

EIG_LAYER = -1
MAX_EIG_K = 10
EIG_K_VARIANTS = (3, 5, 10)
EPS = 1e-8


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


class ChosenProbe(nn.Module):
    """Step 5/5.5 winner: hidden-only 5.3e combo."""

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
        self._net = nn.Sequential(
            nn.Linear(X_scaled.shape[1], self._hidden),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._hidden, 1),
        )

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


def response_positions_from_offsets(
    prompt: str,
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    offsets_row: torch.Tensor,
    eos_id: int,
) -> list[int]:
    """Find token positions whose char span overlaps the response text."""
    n_real = int(attention_mask_row.sum().item())
    response_start_char = len(str(prompt))
    ids = input_ids_row[:n_real]
    offsets = offsets_row[:n_real].tolist()

    positions = []
    for pos, ((start, end), token_id) in enumerate(zip(offsets, ids.tolist())):
        if token_id == eos_id:
            continue
        if end > response_start_char:
            positions.append(pos)
    return positions


def response_positions_from_prompt_len(
    prompt: str,
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    tokenizer,
    eos_id: int,
) -> list[int]:
    """Fallback when tokenizer offsets are unavailable."""
    n_real = int(attention_mask_row.sum().item())
    prompt_len = len(tokenizer.encode(str(prompt), add_special_tokens=False))
    start = min(prompt_len, max(0, n_real - 1))

    real_ids = input_ids_row[:n_real]
    eos_positions = (real_ids == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        stop = int(eos_positions[-1].item())
    else:
        stop = n_real

    return list(range(start, max(start, stop)))


def fallback_response_positions(
    attention_mask_row: torch.Tensor,
    input_ids_row: torch.Tensor,
    eos_id: int,
) -> list[int]:
    """Always return at least one non-padding, non-EOS-ish position."""
    n_real = int(attention_mask_row.sum().item())
    if n_real <= 1:
        return [0]
    last_pos = n_real - 1
    if int(input_ids_row[last_pos].item()) == eos_id and last_pos > 0:
        return [last_pos - 1]
    return [last_pos]


def eigen_features(response_hidden: np.ndarray, max_k: int = MAX_EIG_K) -> np.ndarray:
    """Top-k log eigenvalues + pseudo log-determinant of token covariance."""
    if response_hidden.shape[0] < 2:
        log_eigs = np.full(max_k, np.log(EPS), dtype=np.float32)
        pseudo_logdet = np.float32(max_k * np.log(EPS))
        return np.concatenate([log_eigs, [pseudo_logdet]]).astype(np.float32)

    z = response_hidden.astype(np.float64)
    z = z - z.mean(axis=0, keepdims=True)

    # Non-zero covariance eigenvalues are eigenvalues of the token Gram matrix.
    gram = (z @ z.T) / max(1, z.shape[0] - 1)
    eigvals = eigvalsh(gram)
    eigvals = np.clip(eigvals, EPS, None)
    eigvals = eigvals[::-1]

    top = eigvals[:max_k]
    if len(top) < max_k:
        top = np.pad(top, (0, max_k - len(top)), constant_values=EPS)

    log_top = np.log(top).astype(np.float32)
    pseudo_logdet = np.log(eigvals).sum().astype(np.float32)
    return np.concatenate([log_top, [pseudo_logdet]]).astype(np.float32)


def tokenize_batch(tokenizer, batch_texts: list[str]):
    try:
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        offsets = encoding.pop("offset_mapping")
        return encoding, offsets
    except Exception:
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        return encoding, None


def extract_or_load_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if CACHE_FILE.exists():
        print(f"Loading cached H3 features from {CACHE_FILE}")
        cached = np.load(CACHE_FILE)
        return (
            cached["X_hidden"].astype(np.float32),
            cached["X_eigen"].astype(np.float32),
            cached["response_token_counts"].astype(np.int32),
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        eos_id = tokenizer.eos_token_id
    print(f"Using EOS id = {eos_id}")

    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    prompts = [str(row["prompt"]) for _, row in df.iterrows()]

    hidden_features: list[np.ndarray] = []
    eigen_feature_rows: list[np.ndarray] = []
    response_token_counts: list[int] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting H3"):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_prompts = prompts[start : start + BATCH_SIZE]
        encoding, offsets = tokenize_batch(tokenizer, batch_texts)

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = outputs.hidden_states[EIG_LAYER].float().cpu()
        ids_cpu = input_ids.cpu()
        mask_cpu = attention_mask.cpu()
        offsets_cpu = offsets.cpu() if offsets is not None else None

        for i in range(hidden.size(0)):
            n_real = int(mask_cpu[i].sum().item())
            last_seq_pos = n_real - 1
            hidden_features.append(hidden[i, last_seq_pos].numpy())

            if offsets_cpu is not None:
                positions = response_positions_from_offsets(
                    batch_prompts[i],
                    ids_cpu[i],
                    mask_cpu[i],
                    offsets_cpu[i],
                    eos_id,
                )
            else:
                positions = response_positions_from_prompt_len(
                    batch_prompts[i],
                    ids_cpu[i],
                    mask_cpu[i],
                    tokenizer,
                    eos_id,
                )

            if not positions:
                positions = fallback_response_positions(mask_cpu[i], ids_cpu[i], eos_id)

            response_hidden = hidden[i, positions].numpy()
            eigen_feature_rows.append(eigen_features(response_hidden))
            response_token_counts.append(len(positions))

    X_hidden = np.stack(hidden_features).astype(np.float32)
    X_eigen = np.stack(eigen_feature_rows).astype(np.float32)
    counts = np.array(response_token_counts, dtype=np.int32)

    np.savez(
        CACHE_FILE,
        X_hidden=X_hidden,
        X_eigen=X_eigen,
        response_token_counts=counts,
    )
    print(f"Saved H3 features to {CACHE_FILE}")
    return X_hidden, X_eigen, counts


def select_eigen_columns(X_eigen: np.ndarray, k: int) -> np.ndarray:
    """Use first k log-eigenvalues plus the final pseudo-logdet column."""
    return np.concatenate([X_eigen[:, :k], X_eigen[:, -1:]], axis=1).astype(np.float32)


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


def print_response_diagnostics(counts: np.ndarray) -> None:
    print("\nResponse-token diagnostics:")
    print(
        f"  mean={counts.mean():.1f}  p50={np.median(counts):.0f}  "
        f"p95={np.percentile(counts, 95):.0f}  "
        f"min={counts.min()}  max={counts.max()}"
    )


def print_summary(results: dict[str, dict]) -> None:
    print(f"\n{'=' * 110}")
    print("  H3 SUMMARY — sorted by test AUROC (mean ± std across 5 folds)")
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


def main() -> None:
    set_seed(SEED)

    df = pd.read_csv(DATA_FILE)
    y = df["label"].astype(float).astype(int).values
    print(
        f"Loaded {len(df)} samples; balance: "
        f"{dict(pd.Series(y).value_counts(normalize=True).round(3))}"
    )

    X_hidden, X_eigen_full, counts = extract_or_load_features(df)
    print(f"Hidden features: {X_hidden.shape}")
    print(f"Eigen features : {X_eigen_full.shape} (top-{MAX_EIG_K} + logdet)")
    print_response_diagnostics(counts)

    splits = split_data(y)
    print(f"\nFolds: {len(splits)}\n")

    variants: list[tuple[str, np.ndarray]] = [
        ("h3a_hidden_only", X_hidden),
    ]
    for k in EIG_K_VARIANTS:
        X_eig_k = select_eigen_columns(X_eigen_full, k)
        variants.append((f"h3_hidden+eig{k}", np.concatenate([X_hidden, X_eig_k], axis=1)))
        variants.append((f"h3_eig{k}_only", X_eig_k))

    results = {}
    for name, X in variants:
        print(f"  -> {name}  dim={X.shape[1]}")
        results[name] = run_probe(ChosenProbe, X.astype(np.float32), y, splits)

    print_summary(results)


if __name__ == "__main__":
    main()
