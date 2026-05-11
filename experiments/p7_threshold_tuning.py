"""
Step 7: final threshold tuning for accuracy.

Final model is fixed before this script:
  H4 chunk-layer grid -> fold-local PCA16 -> Mass-Mean score -> 5.3e MLP head

This script does not use threshold to choose the model. It only:
  1. builds 5-fold out-of-fold scores for all labelled samples;
  2. picks one global threshold maximizing OOF accuracy;
  3. reports threshold stability by fold and predicted positive rate;
  4. trains the final model on all labelled data;
  5. applies the OOF threshold to test.csv scores.

Outputs:
  - oof_scores.csv
  - test_scores.csv
  - predictions.csv
  - threshold_results.json

Run from repo root:
    python -m experiments.p7_threshold_tuning
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.h4_chunk_layer_pca import (  # noqa: E402
    BATCH_SIZE,
    ChosenProbe,
    POOL_NAMES,
    fallback_response_positions,
    response_positions_from_offsets,
    response_positions_from_prompt_len,
    split_response_chunks,
    tokenize_batch,
)
from model import MAX_LENGTH, get_model_and_tokenizer  # noqa: E402
from splitting import split_data  # noqa: E402

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"
TEST_FILE = REPO_ROOT / "data" / "test.csv"
CACHE_DIR = REPO_ROOT / "data" / "cache"
TRAIN_GRID_CACHE = CACHE_DIR / "h4_chunk_layer_grid.npz"
TEST_GRID_CACHE = CACHE_DIR / "h4_chunk_layer_grid_test.npz"

OOF_SCORES_FILE = REPO_ROOT / "oof_scores.csv"
TEST_SCORES_FILE = REPO_ROOT / "test_scores.csv"
PREDICTIONS_FILE = REPO_ROOT / "predictions.csv"
RESULTS_FILE = REPO_ROOT / "threshold_results.json"

LAYERS = (13, 15, 17, 24)
N_COMPONENTS = 16


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FinalH6Model:
    """H4-grid PCA16 + train-fold Mass-Mean score + 5.3e MLP head."""

    def __init__(self):
        self._grid_scaler = StandardScaler()
        self._pca: PCA | None = None
        self._mm_theta: np.ndarray | None = None
        self._mm_mu = 0.0
        self._mm_sigma = 1.0
        self._probe = ChosenProbe()

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

    def _fit_transform_features(self, X_grid: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_scaled = self._grid_scaler.fit_transform(X_grid)
        self._pca = PCA(
            n_components=N_COMPONENTS,
            svd_solver="randomized",
            random_state=SEED,
        )
        X_pca = self._pca.fit_transform(X_scaled)
        X_mm = self._fit_mm(X_scaled, y)
        return np.concatenate([X_pca, X_mm], axis=1).astype(np.float32)

    def _transform_features(self, X_grid: np.ndarray) -> np.ndarray:
        X_scaled = self._grid_scaler.transform(X_grid)
        X_pca = self._pca.transform(X_scaled)
        X_mm = self._transform_mm(X_scaled)
        return np.concatenate([X_pca, X_mm], axis=1).astype(np.float32)

    def fit(self, X_grid: np.ndarray, y: np.ndarray) -> "FinalH6Model":
        X_final = self._fit_transform_features(X_grid, y)
        self._probe.fit(X_final, y)
        return self

    def predict_proba(self, X_grid: np.ndarray) -> np.ndarray:
        X_final = self._transform_features(X_grid)
        return self._probe.predict_proba(X_final)


def extract_or_load_grid(
    df: pd.DataFrame,
    cache_file: Path,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_file.exists():
        print(f"Loading cached grid features from {cache_file}")
        cached = np.load(cache_file)
        X_cached = cached["X_grid"].astype(np.float32)
        counts_cached = cached["response_token_counts"].astype(np.int32)
        if len(X_cached) == len(df):
            return X_cached, counts_cached
        print(
            f"Cache row-count mismatch: cache={len(X_cached)} df={len(df)}. "
            "Re-extracting."
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

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
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
                    pooled = layer_hidden[i, pools[pool_name]].mean(dim=0).numpy()
                    sample_parts.append(pooled)
            grid_rows.append(np.concatenate(sample_parts).astype(np.float32))

    X_grid = np.stack(grid_rows).astype(np.float32)
    counts = np.array(response_token_counts, dtype=np.int32)
    np.savez(
        cache_file,
        X_grid=X_grid,
        response_token_counts=counts,
        layers=np.array(LAYERS, dtype=np.int32),
        pool_names=np.array(POOL_NAMES),
    )
    print(f"Saved grid features to {cache_file} ({X_grid.shape})")
    return X_grid, counts


def best_threshold_for_accuracy(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate([scores, np.linspace(0.0, 1.0, 1001)]))
    best_threshold, best_acc = 0.5, -1.0
    for threshold in candidates:
        acc = accuracy_score(y, (scores >= threshold).astype(int))
        if acc > best_acc:
            best_acc = float(acc)
            best_threshold = float(threshold)
    return best_threshold, best_acc


def metrics_at_threshold(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, scores)),
        "predicted_positive_rate": float(pred.mean()),
    }


def main() -> None:
    set_seed(SEED)

    df = pd.read_csv(DATA_FILE)
    y = df["label"].astype(float).astype(int).values
    df_test = pd.read_csv(TEST_FILE)
    test_ids = df_test["id"].tolist() if "id" in df_test.columns else df_test.index.tolist()

    print(
        f"Loaded train={len(df)} test={len(df_test)}; train balance: "
        f"{dict(pd.Series(y).value_counts(normalize=True).round(3))}"
    )

    X_grid, _ = extract_or_load_grid(df, TRAIN_GRID_CACHE, "Train H4 grid")
    X_test_grid, _ = extract_or_load_grid(df_test, TEST_GRID_CACHE, "Test H4 grid")
    print(f"Train grid: {X_grid.shape}")
    print(f"Test grid : {X_test_grid.shape}")

    splits = split_data(y)
    print(f"Folds: {len(splits)}")

    oof_scores = np.zeros(len(y), dtype=np.float32)
    oof_folds = np.zeros(len(y), dtype=np.int32)
    fold_diagnostics = []

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits, start=1):
        idx_fit = (
            np.concatenate([idx_train, idx_val]) if idx_val is not None else idx_train
        )
        model = FinalH6Model().fit(X_grid[idx_fit], y[idx_fit])
        scores = model.predict_proba(X_grid[idx_test])[:, 1]
        oof_scores[idx_test] = scores.astype(np.float32)
        oof_folds[idx_test] = fold_idx

        fold_threshold, fold_best_acc = best_threshold_for_accuracy(scores, y[idx_test])
        fold_metrics_at_global_05 = metrics_at_threshold(scores, y[idx_test], 0.5)
        fold_metrics_at_best = metrics_at_threshold(
            scores,
            y[idx_test],
            fold_threshold,
        )
        fold_diagnostics.append({
            "fold": fold_idx,
            "n_fit": int(len(idx_fit)),
            "n_test": int(len(idx_test)),
            "best_threshold": fold_threshold,
            "best_accuracy": fold_best_acc,
            "positive_rate_at_best_threshold": fold_metrics_at_best[
                "predicted_positive_rate"
            ],
            "auroc": fold_metrics_at_global_05["auroc"],
            "positive_rate_at_0_5": fold_metrics_at_global_05[
                "predicted_positive_rate"
            ],
        })
        print(
            f"Fold {fold_idx}: AUROC={fold_metrics_at_global_05['auroc']:.4f} "
            f"best_t={fold_threshold:.4f} best_acc={fold_best_acc:.4f}"
        )

    threshold, best_oof_acc = best_threshold_for_accuracy(oof_scores, y)
    oof_metrics = metrics_at_threshold(oof_scores, y, threshold)
    oof_metrics_at_05 = metrics_at_threshold(oof_scores, y, 0.5)

    print("\nFinal OOF threshold tuning")
    print(f"  threshold     : {threshold:.6f}")
    print(f"  OOF accuracy  : {oof_metrics['accuracy']:.4f}")
    print(f"  OOF F1        : {oof_metrics['f1']:.4f}")
    print(f"  OOF AUROC     : {oof_metrics['auroc']:.4f}")
    print(f"  OOF pos rate  : {oof_metrics['predicted_positive_rate']:.4f}")
    print(f"  OOF acc @ 0.5 : {oof_metrics_at_05['accuracy']:.4f}")

    oof_pred = (oof_scores >= threshold).astype(int)
    pd.DataFrame({
        "id": np.arange(len(y)),
        "fold": oof_folds,
        "label": y,
        "score": oof_scores,
        "pred": oof_pred,
    }).to_csv(OOF_SCORES_FILE, index=False)

    final_model = FinalH6Model().fit(X_grid, y)
    test_scores = final_model.predict_proba(X_test_grid)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)

    pd.DataFrame({
        "id": test_ids,
        "score": test_scores,
        "label": test_pred,
    }).to_csv(TEST_SCORES_FILE, index=False)
    pd.DataFrame({
        "id": test_ids,
        "label": test_pred,
    }).to_csv(PREDICTIONS_FILE, index=False)

    results = {
        "model": "h6_h4_pca16+mm_current_5.3e_head",
        "threshold": threshold,
        "oof_metrics": oof_metrics,
        "oof_metrics_at_0_5": oof_metrics_at_05,
        "fold_diagnostics": fold_diagnostics,
        "test_predicted_positive_rate": float(test_pred.mean()),
        "n_train": int(len(y)),
        "n_test": int(len(test_pred)),
        "outputs": {
            "oof_scores": str(OOF_SCORES_FILE.relative_to(REPO_ROOT)),
            "test_scores": str(TEST_SCORES_FILE.relative_to(REPO_ROOT)),
            "predictions": str(PREDICTIONS_FILE.relative_to(REPO_ROOT)),
        },
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {OOF_SCORES_FILE}")
    print(f"Saved {TEST_SCORES_FILE}")
    print(f"Saved {PREDICTIONS_FILE}")
    print(f"Saved {RESULTS_FILE}")


if __name__ == "__main__":
    main()
