"""
H1: Token position experiment.

Compares three token positions for hidden-state pooling on the LAST layer:
  - last_seq        — last real token of the full input (current default)
  - last_response   — token just before the final <|endoftext|> in the response
  - end_of_question — last token of the prompt (right before response begins)

Probe is the regularized winner from step 5: SmallerMLPProbe with
hidden=64, dropout=0.5, weight_decay=1e-2, 50 epochs (5.3e_combo).
Layer is kept at baseline (last). Only the token position changes.
CV=5 via splitting.split_data.

Run from repo root:
    python -m experiments.h1_token_position
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluate import run_evaluation  # noqa: E402
from experiments.p5_probe_regularization import SmallerMLPProbe  # noqa: E402
from model import MAX_LENGTH, get_model_and_tokenizer  # noqa: E402
from splitting import split_data  # noqa: E402


class ChosenProbe(SmallerMLPProbe):
    """Step 5 winner: 5.3e combo. Parameterless ctor for evaluate.run_evaluation."""

    def __init__(self):
        super().__init__(hidden=64, dropout=0.5, weight_decay=1e-2, epochs=50)

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"
BATCH_SIZE = 4
STRATEGIES = ["last_seq", "last_response", "end_of_question"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_positions(
    prompt: str,
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    tokenizer,
    eos_id: int,
) -> dict[str, int]:
    """Compute three token positions for one sample (1-D inputs)."""
    n_real = int(attention_mask_row.sum().item())
    last_seq = n_real - 1

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    end_of_question = max(0, min(len(prompt_ids), last_seq + 1) - 1)

    real_ids = input_ids_row[:n_real]
    eos_positions = (real_ids == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        last_response = max(0, int(eos_positions[-1].item()) - 1)
    else:
        last_response = last_seq

    return {
        "last_seq": last_seq,
        "last_response": last_response,
        "end_of_question": end_of_question,
    }


def extract_features(model, tokenizer, df: pd.DataFrame, device: torch.device):
    """Forward pass; collect last-layer hidden states at the three positions."""
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        eos_id = tokenizer.eos_token_id
    print(f"  Using <|endoftext|> id = {eos_id}")

    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    prompts = list(df["prompt"])

    feats = {k: [] for k in STRATEGIES}
    pos_diag = {k: [] for k in STRATEGIES}

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting"):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_prompts = prompts[start : start + BATCH_SIZE]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_last = out.hidden_states[-1].float().cpu()
        ids_cpu = input_ids.cpu()
        mask_cpu = attention_mask.cpu()

        for i in range(hidden_last.size(0)):
            positions = find_positions(
                batch_prompts[i], ids_cpu[i], mask_cpu[i], tokenizer, eos_id
            )
            for key, pos in positions.items():
                feats[key].append(hidden_last[i, pos].numpy())
                pos_diag[key].append(pos)

    X = {k: np.stack(v).astype(np.float32) for k, v in feats.items()}
    return X, pos_diag


def position_diagnostics(pos_diag: dict[str, list[int]]) -> None:
    print("\nPosition diagnostics (token index per strategy):")
    for key, positions in pos_diag.items():
        arr = np.array(positions)
        print(
            f"  {key:18s} mean={arr.mean():6.1f}  "
            f"p50={np.median(arr):4.0f}  p95={np.percentile(arr, 95):4.0f}  "
            f"min={arr.min():4d}  max={arr.max():4d}"
        )


def summarize_strategy(fold_results: list[dict]) -> dict[str, float]:
    return {
        "test_acc_mean": float(np.mean([r["test_accuracy"] for r in fold_results])),
        "test_acc_std": float(np.std([r["test_accuracy"] for r in fold_results])),
        "test_f1_mean": float(np.mean([r["test_f1"] for r in fold_results])),
        "test_f1_std": float(np.std([r["test_f1"] for r in fold_results])),
        "test_auroc_mean": float(np.mean([r["test_auroc"] for r in fold_results])),
        "test_auroc_std": float(np.std([r["test_auroc"] for r in fold_results])),
    }


def main() -> None:
    set_seed(SEED)

    df = pd.read_csv(DATA_FILE)
    y = np.array([int(float(v)) for v in df["label"]])
    print(f"Loaded {len(df)} samples")

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

    X_dict, pos_diag = extract_features(model, tokenizer, df, device)
    position_diagnostics(pos_diag)

    splits = split_data(y)
    print(f"\nFolds: {len(splits)}")

    summary: dict[str, dict[str, float]] = {}
    for name in STRATEGIES:
        print(f"\n{'=' * 60}\n  Strategy: {name}\n{'=' * 60}")
        fold_results = run_evaluation(splits, X_dict[name], y, ChosenProbe)
        summary[name] = summarize_strategy(fold_results)

    print("\n" + "=" * 70)
    print(f"  H1 SUMMARY (test mean ± std across 5 folds)")
    print("=" * 70)
    print(f"  {'Strategy':<18s} {'Acc':>16s} {'F1':>16s} {'AUROC':>16s}")
    for name in STRATEGIES:
        s = summary[name]
        print(
            f"  {name:<18s} "
            f"{s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f}  "
            f"{s['test_f1_mean']:.4f} ± {s['test_f1_std']:.4f}  "
            f"{s['test_auroc_mean']:.4f} ± {s['test_auroc_std']:.4f}"
        )


if __name__ == "__main__":
    main()
