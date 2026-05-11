"""
H4: response chunks x layers + fold-local PCA.

Hypothesis: the signal is spread across response-localized regions and
middle/late layers, so a compressed layer-chunk grid may beat a single
last_seq vector.

For each sample we extract:
  layers = [13, 15, 17, 24]
  pools  = [response_head, response_mid, response_tail, last_seq]

Each response_* pool is a mean over one third of response tokens. last_seq is
the usual last real token. The flattened grid has
4 layers x 4 pools x 896 = 14336 dimensions before PCA.

PCA is fitted inside each train fold only; val/test are transformed by the
fold-fitted scaler/PCA. This avoids leakage.

Run from repo root:
    python -m experiments.h4_chunk_layer_pca
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
CACHE_FILE = CACHE_DIR / "h4_chunk_layer_grid.npz"
BATCH_SIZE = 4

LAYERS = (13, 15, 17, 24)
POOL_NAMES = ("response_head", "response_mid", "response_tail", "last_seq")
N_COMPONENTS = (16, 32, 64, 128, 256)


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


class PCAPipelineProbe:
    """Fold-local StandardScaler -> PCA -> base probe."""

    def __init__(self, base_factory: Callable[[], object], n_components: int):
        self._base_factory = base_factory
        self._n_components = n_components
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._probe = None

    def _fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        max_components = min(X.shape[0] - 1, X.shape[1])
        if self._n_components > max_components:
            raise ValueError(
                f"n_components={self._n_components} is invalid for {X.shape}"
            )

        X_scaled = self._scaler.fit_transform(X)
        self._pca = PCA(
            n_components=self._n_components,
            svd_solver="randomized",
            random_state=SEED,
        )
        return self._pca.fit_transform(X_scaled).astype(np.float32)

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._pca.transform(X_scaled).astype(np.float32)

    def fit(self, X, y):
        X_pca = self._fit_transform_features(X)
        self._probe = self._base_factory()
        self._probe.fit(X_pca, y)
        return self

    def predict_proba(self, X):
        return self._probe.predict_proba(self._transform_features(X))

    def predict(self, X):
        return self._probe.predict(self._transform_features(X))

    def fit_hyperparameters(self, X_val, y_val):
        if hasattr(self._probe, "fit_hyperparameters"):
            self._probe.fit_hyperparameters(self._transform_features(X_val), y_val)
        return self


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


def response_positions_from_offsets(
    prompt: str,
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    offsets_row: torch.Tensor,
    eos_id: int,
) -> list[int]:
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
    n_real = int(attention_mask_row.sum().item())
    prompt_len = len(tokenizer.encode(str(prompt), add_special_tokens=False))
    start = min(prompt_len, max(0, n_real - 1))

    real_ids = input_ids_row[:n_real]
    eos_positions = (real_ids == eos_id).nonzero(as_tuple=True)[0]
    stop = int(eos_positions[-1].item()) if len(eos_positions) > 0 else n_real
    return list(range(start, max(start, stop)))


def fallback_response_positions(
    attention_mask_row: torch.Tensor,
    input_ids_row: torch.Tensor,
    eos_id: int,
) -> list[int]:
    n_real = int(attention_mask_row.sum().item())
    if n_real <= 1:
        return [0]

    last_pos = n_real - 1
    if int(input_ids_row[last_pos].item()) == eos_id and last_pos > 0:
        return [last_pos - 1]
    return [last_pos]


def split_response_chunks(positions: list[int]) -> dict[str, list[int]]:
    """Split response token positions into head/mid/tail thirds."""
    arr = np.array(positions, dtype=int)
    if arr.size == 0:
        raise ValueError("positions must be non-empty")

    n = len(arr)
    head_end = max(1, int(np.ceil(n / 3)))
    mid_start = min(n - 1, n // 3)
    mid_end = max(mid_start + 1, int(np.ceil(2 * n / 3)))
    mid_end = min(n, mid_end)
    tail_start = max(0, n - head_end)

    return {
        "response_head": arr[:head_end].tolist(),
        "response_mid": arr[mid_start:mid_end].tolist(),
        "response_tail": arr[tail_start:].tolist(),
    }


def extract_or_load_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if CACHE_FILE.exists():
        print(f"Loading cached H4 grid features from {CACHE_FILE}")
        cached = np.load(CACHE_FILE)
        return (
            cached["X_grid"].astype(np.float32),
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
    model.eval()

    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        eos_id = tokenizer.eos_token_id
    print(f"Using EOS id = {eos_id}")

    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    prompts = [str(row["prompt"]) for _, row in df.iterrows()]

    grid_rows: list[np.ndarray] = []
    response_token_counts: list[int] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting H4"):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_prompts = prompts[start : start + BATCH_SIZE]
        encoding, offsets = tokenize_batch(tokenizer, batch_texts)

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        selected_layers = [outputs.hidden_states[layer].float().cpu() for layer in LAYERS]
        ids_cpu = input_ids.cpu()
        mask_cpu = attention_mask.cpu()
        offsets_cpu = offsets.cpu() if offsets is not None else None

        for i in range(input_ids.size(0)):
            n_real = int(mask_cpu[i].sum().item())
            last_seq_pos = n_real - 1

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

            pools = split_response_chunks(positions)
            pools["last_seq"] = [last_seq_pos]
            response_token_counts.append(len(positions))

            sample_parts = []
            for layer_hidden in selected_layers:
                for pool_name in POOL_NAMES:
                    pos = pools[pool_name]
                    pooled = layer_hidden[i, pos].mean(dim=0).numpy()
                    sample_parts.append(pooled)
            grid_rows.append(np.concatenate(sample_parts).astype(np.float32))

    X_grid = np.stack(grid_rows).astype(np.float32)
    counts = np.array(response_token_counts, dtype=np.int32)

    np.savez(
        CACHE_FILE,
        X_grid=X_grid,
        response_token_counts=counts,
        layers=np.array(LAYERS, dtype=np.int32),
        pool_names=np.array(POOL_NAMES),
    )
    print(f"Saved H4 grid features to {CACHE_FILE} ({X_grid.shape})")
    return X_grid, counts


def select_layer_pool(X_grid: np.ndarray, layer: int, pool_name: str) -> np.ndarray:
    hidden_dim = X_grid.shape[1] // (len(LAYERS) * len(POOL_NAMES))
    X = X_grid.reshape(len(X_grid), len(LAYERS), len(POOL_NAMES), hidden_dim)
    layer_idx = LAYERS.index(layer)
    pool_idx = POOL_NAMES.index(pool_name)
    return X[:, layer_idx, pool_idx, :].astype(np.float32)


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
    print(f"\n{'=' * 112}")
    print("  H4 SUMMARY - sorted by test AUROC (mean +/- std across 5 folds)")
    print(f"{'=' * 112}")
    print(
        f"  {'Variant':<32s} | "
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
            f"  {name:<32s} | "
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

    X_grid, counts = extract_or_load_grid(df)
    print(f"Grid features: {X_grid.shape}")
    print(f"Layers       : {list(LAYERS)}")
    print(f"Pools        : {list(POOL_NAMES)}")
    print_response_diagnostics(counts)

    splits = split_data(y)
    print(f"\nFolds: {len(splits)}\n")

    variants: list[tuple[str, np.ndarray, Callable[[], object]]] = [
        (
            "h4_layer24_last_seq",
            select_layer_pool(X_grid, 24, "last_seq"),
            ChosenProbe,
        ),
        (
            "h4_layer13_last_seq",
            select_layer_pool(X_grid, 13, "last_seq"),
            ChosenProbe,
        ),
    ]

    for n_components in N_COMPONENTS:
        variants.append((
            f"h4_grid_pca{n_components}",
            X_grid,
            lambda n_components=n_components: PCAPipelineProbe(
                ChosenProbe, n_components=n_components
            ),
        ))

    results = {}
    for name, X, factory in variants:
        print(f"  -> {name}  dim={X.shape[1]}")
        results[name] = run_probe(factory, X.astype(np.float32), y, splits)

    print_summary(results)


if __name__ == "__main__":
    main()
