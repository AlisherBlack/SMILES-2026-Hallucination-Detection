"""
H2: Layer scan with variance-ratio screening + per-layer probe + top-K concat.

Stage 1: extract hidden states at last_seq for all 25 layers (1 embedding +
         24 transformer), cache to data/cache/last_seq_all_layers.npz.
Stage 2: variance ratio R = V_θ / V_T per layer (PRISM Eq.6). Cheap screening.
Stage 3: full ChosenProbe per layer (CV=5). Pearson(R, AUROC) sanity check.
Stage 4: concat top-K layers (K=2,3,5) by R; train probe on concat.

Probe = ChosenProbe (5.3e combo). Token = last_seq (winner of H1).
Comparison metric: test AUROC (threshold-independent).

Run from repo root:
    python -m experiments.h2_layer_scan
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

from experiments.p5_probe_regularization import (  # noqa: E402
    SmallerMLPProbe,
    run_probe,
)
from model import MAX_LENGTH, get_model_and_tokenizer  # noqa: E402
from splitting import split_data  # noqa: E402

SEED = 42
DATA_FILE = REPO_ROOT / "data" / "dataset.csv"
CACHE_DIR = REPO_ROOT / "data" / "cache"
CACHE_FILE = CACHE_DIR / "last_seq_all_layers.npz"
BATCH_SIZE = 4
TOP_K_LIST = [2, 3, 5]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChosenProbe(SmallerMLPProbe):
    """Step 5 winner: 5.3e combo. Parameterless ctor for run_probe."""

    def __init__(self):
        super().__init__(hidden=64, dropout=0.5, weight_decay=1e-2, epochs=50)


# --------------------------------------------------------------------------
# Stage 1: hidden states at last_seq across all layers
# --------------------------------------------------------------------------


def extract_or_load_all_layers(df: pd.DataFrame) -> np.ndarray:
    """Returns (n_samples, n_layers, hidden_dim) at last_seq position."""
    if CACHE_FILE.exists():
        print(f"Loading cached features from {CACHE_FILE}")
        return np.load(CACHE_FILE)["X"].astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Cache miss — running Qwen forward pass for all layers.")

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

        last_pos = mask.sum(dim=1) - 1  # (batch,)
        idx = torch.arange(ids.size(0), device=ids.device)
        # For each layer, gather last_seq position → (batch, hidden_dim)
        per_layer = [
            h[idx, last_pos].float().cpu() for h in out.hidden_states
        ]
        # (batch, n_layers, hidden_dim)
        stacked = torch.stack(per_layer, dim=1)
        feats.append(stacked.numpy())

    X = np.concatenate(feats, axis=0).astype(np.float32)
    np.savez(CACHE_FILE, X=X)
    print(f"Saved features to {CACHE_FILE} ({X.shape})")
    return X


# --------------------------------------------------------------------------
# Stage 2: variance ratio (PRISM Eq.6)
# --------------------------------------------------------------------------


def variance_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """R = V_θ / V_T. Fraction of total variance along truthfulness direction.

    θ = mean(X⁺) − mean(X⁻).
    V_θ = (1/n) Σᵢ (xcᵢ · θ̂)²  where  xcᵢ = xᵢ − mean(X), θ̂ = θ / ‖θ‖.
    V_T = (1/n) Σᵢ ‖xcᵢ‖² = trace of empirical covariance.
    Ratio is invariant to the 1/n factor; we drop it.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    theta = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    theta_norm_sq = float(theta @ theta) + 1e-12
    proj = Xc @ theta  # (n,)
    V_theta = float(proj @ proj) / theta_norm_sq
    V_T = float((Xc * Xc).sum())
    return V_theta / (V_T + 1e-12)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    set_seed(SEED)

    df = pd.read_csv(DATA_FILE)
    y = np.array([int(float(v)) for v in df["label"]])
    print(f"Loaded {len(df)} samples\n")

    X_all = extract_or_load_all_layers(df)
    n_samples, n_layers, hidden_dim = X_all.shape
    print(f"X shape: {X_all.shape} (n_layers includes embedding @ index 0)")

    splits = split_data(y)
    print(f"Folds: {len(splits)}\n")

    # ─── Stage 2: variance ratio screening ────────────────────────────────
    print("=" * 60)
    print("Stage 2: variance ratio R = V_θ / V_T (full data, screening)")
    print("=" * 60)
    R_per_layer = np.array([variance_ratio(X_all[:, L, :], y) for L in range(n_layers)])

    # ─── Stage 3: full probe per layer ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 3: ChosenProbe per layer (CV=5)")
    print("=" * 60)
    rows = []
    for L in tqdm(range(n_layers), desc="Per-layer probe"):
        result = run_probe(ChosenProbe, X_all[:, L, :], y, splits)
        rows.append(
            {
                "layer": L,
                "R": float(R_per_layer[L]),
                "auroc_mean": result["test"]["auroc"][0],
                "auroc_std": result["test"]["auroc"][1],
                "acc_mean": result["test"]["acc"][0],
                "f1_mean": result["test"]["f1"][0],
                "train_auroc": result["train"]["auroc"][0],
                "gap": result["train"]["auroc"][0] - result["test"]["auroc"][0],
            }
        )
    df_layers = pd.DataFrame(rows)

    # Pearson(R, AUROC) — PRISM screening sanity check
    r_corr = float(np.corrcoef(df_layers["R"], df_layers["auroc_mean"])[0, 1])

    print(f"\n  {'Layer':>5s}  {'R':>7s}  {'AUROC':>16s}  {'gap':>7s}  {'acc':>7s}")
    print("  " + "-" * 50)
    for _, r in df_layers.iterrows():
        marker = " ←" if r["layer"] == n_layers - 1 else ""
        print(
            f"  {int(r['layer']):>5d}  {r['R']:>7.4f}  "
            f"{r['auroc_mean']:.4f} ± {r['auroc_std']:.3f}  "
            f"{r['gap']:>+7.4f}  {r['acc_mean']:>7.4f}{marker}"
        )

    print(f"\n  Pearson(R, AUROC) over {n_layers} layers = {r_corr:.4f}")
    print(f"  (PRISM reports ≈0.77; high corr ⇒ R is a fast proxy for AUROC.)")

    top_by_R = df_layers.sort_values("R", ascending=False).head(5)
    top_by_AUROC = df_layers.sort_values("auroc_mean", ascending=False).head(5)
    print(f"\n  Top-5 layers by R     : {top_by_R['layer'].tolist()}")
    print(f"  Top-5 layers by AUROC : {top_by_AUROC['layer'].tolist()}")

    default_layer = n_layers - 1
    default_auroc = float(df_layers.loc[default_layer, "auroc_mean"])
    best_layer = int(top_by_AUROC.iloc[0]["layer"])
    best_auroc = float(top_by_AUROC.iloc[0]["auroc_mean"])
    print(f"\n  Default (layer {default_layer}) AUROC = {default_auroc:.4f}")
    print(f"  Best single layer = {best_layer}, AUROC = {best_auroc:.4f}")
    print(f"  Δ vs default = {best_auroc - default_auroc:+.4f}")

    # ─── Stage 4: concat top-K layers by R ────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 4: concat top-K layers by R")
    print("=" * 60)
    R_argsort = np.argsort(-R_per_layer)
    concat_rows = []
    for K in TOP_K_LIST:
        top_K = R_argsort[:K]
        X_concat = X_all[:, top_K, :].reshape(n_samples, -1)
        result = run_probe(ChosenProbe, X_concat, y, splits)
        concat_rows.append(
            {
                "K": K,
                "layers": sorted(top_K.tolist()),
                "dim": X_concat.shape[1],
                "auroc_mean": result["test"]["auroc"][0],
                "auroc_std": result["test"]["auroc"][1],
                "acc_mean": result["test"]["acc"][0],
                "f1_mean": result["test"]["f1"][0],
                "gap": result["train"]["auroc"][0] - result["test"]["auroc"][0],
            }
        )

    print(f"\n  {'K':>3s}  {'layers':<24s}  {'dim':>5s}  {'AUROC':>16s}  {'gap':>7s}  {'acc':>7s}")
    print("  " + "-" * 70)
    for r in concat_rows:
        layers_str = "[" + ", ".join(map(str, r["layers"])) + "]"
        print(
            f"  {r['K']:>3d}  {layers_str:<24s}  {r['dim']:>5d}  "
            f"{r['auroc_mean']:.4f} ± {r['auroc_std']:.3f}  "
            f"{r['gap']:>+7.4f}  {r['acc_mean']:>7.4f}"
        )

    # ─── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("H2 SUMMARY (test AUROC)")
    print("=" * 60)
    print(f"  Default (layer {default_layer})   = {default_auroc:.4f}")
    print(f"  Best single layer ({best_layer:>2d}) = {best_auroc:.4f}  (Δ {best_auroc - default_auroc:+.4f})")
    for r in concat_rows:
        print(
            f"  Top-{r['K']} concat              = {r['auroc_mean']:.4f}  "
            f"(Δ {r['auroc_mean'] - default_auroc:+.4f})"
        )


if __name__ == "__main__":
    main()
