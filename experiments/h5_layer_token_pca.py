"""
H5: all layers x sampled response tokens + fold-local PCA.

This generalizes H4. Instead of four selected layers and chunk means, we use
all hidden-state layers and fixed response token positions:

  layers = 0..24  (embedding + 24 transformer layers)
  tokens = [first, q25, q50, q75, last_response, last_seq]

The flattened grid has 25 x 6 x 896 = 134400 dimensions before PCA.
PCA is fitted inside each train fold only; val/test are transformed by the
fold-fitted scaler/PCA. This avoids leakage.

Run from repo root:
    python -m experiments.h5_layer_token_pca
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.h4_chunk_layer_pca import (  # noqa: E402
    BATCH_SIZE,
    CACHE_DIR,
    ChosenProbe,
    PCAPipelineProbe,
    fallback_response_positions,
    print_response_diagnostics,
    response_positions_from_offsets,
    response_positions_from_prompt_len,
    run_probe,
    set_seed,
    tokenize_batch,
)
from model import get_model_and_tokenizer  # noqa: E402
from splitting import split_data  # noqa: E402

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"
CACHE_FILE = CACHE_DIR / "h5_layer_token_grid.npz"

LAYERS = tuple(range(25))
TOKEN_NAMES = (
    "response_first",
    "response_q25",
    "response_q50",
    "response_q75",
    "last_response",
    "last_seq",
)
N_COMPONENTS = (16, 32, 64)


def sampled_token_positions(positions: list[int], last_seq_pos: int) -> dict[str, int]:
    """Pick fixed response quantile positions plus last_seq."""
    if not positions:
        raise ValueError("positions must be non-empty")

    n = len(positions)

    def at(frac: float) -> int:
        idx = int(round((n - 1) * frac))
        return positions[min(n - 1, max(0, idx))]

    return {
        "response_first": positions[0],
        "response_q25": at(0.25),
        "response_q50": at(0.50),
        "response_q75": at(0.75),
        "last_response": positions[-1],
        "last_seq": last_seq_pos,
    }


def extract_or_load_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if CACHE_FILE.exists():
        print(f"Loading cached H5 token-grid features from {CACHE_FILE}")
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

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting H5"):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_prompts = prompts[start : start + BATCH_SIZE]
        encoding, offsets = tokenize_batch(tokenizer, batch_texts)

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if len(outputs.hidden_states) != len(LAYERS):
            raise ValueError(
                f"Expected {len(LAYERS)} hidden-state layers, "
                f"got {len(outputs.hidden_states)}"
            )

        all_layers = [h.float().cpu() for h in outputs.hidden_states]
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

            token_pos = sampled_token_positions(positions, last_seq_pos)
            response_token_counts.append(len(positions))

            sample_parts = []
            for layer_hidden in all_layers:
                for token_name in TOKEN_NAMES:
                    sample_parts.append(layer_hidden[i, token_pos[token_name]].numpy())
            grid_rows.append(np.concatenate(sample_parts).astype(np.float32))

    X_grid = np.stack(grid_rows).astype(np.float32)
    counts = np.array(response_token_counts, dtype=np.int32)

    np.savez(
        CACHE_FILE,
        X_grid=X_grid,
        response_token_counts=counts,
        layers=np.array(LAYERS, dtype=np.int32),
        token_names=np.array(TOKEN_NAMES),
    )
    print(f"Saved H5 token-grid features to {CACHE_FILE} ({X_grid.shape})")
    return X_grid, counts


def select_layer_token(X_grid: np.ndarray, layer: int, token_name: str) -> np.ndarray:
    hidden_dim = X_grid.shape[1] // (len(LAYERS) * len(TOKEN_NAMES))
    X = X_grid.reshape(len(X_grid), len(LAYERS), len(TOKEN_NAMES), hidden_dim)
    layer_idx = LAYERS.index(layer)
    token_idx = TOKEN_NAMES.index(token_name)
    return X[:, layer_idx, token_idx, :].astype(np.float32)


def print_summary(results: dict[str, dict]) -> None:
    print(f"\n{'=' * 112}")
    print("  H5 SUMMARY - sorted by test AUROC (mean +/- std across 5 folds)")
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
    print(f"Token-grid features: {X_grid.shape}")
    print(f"Layers             : {list(LAYERS)}")
    print(f"Tokens             : {list(TOKEN_NAMES)}")
    print_response_diagnostics(counts)

    splits = split_data(y)
    print(f"\nFolds: {len(splits)}\n")

    variants: list[tuple[str, np.ndarray, Callable[[], object]]] = [
        (
            "h5_layer24_last_seq",
            select_layer_token(X_grid, 24, "last_seq"),
            ChosenProbe,
        ),
        (
            "h5_layer13_last_seq",
            select_layer_token(X_grid, 13, "last_seq"),
            ChosenProbe,
        ),
    ]

    for n_components in N_COMPONENTS:
        variants.append((
            f"h5_grid_pca{n_components}",
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
