# Experiments

5-fold StratifiedKFold (seed=42); inside each outer fold, 15% of train → val.
Metrics are averaged over 5 folds. The primary comparison metric is test AUROC
(threshold-independent). The final threshold is tuned only for the winning
configuration (step 7).

## Baselines

| Model | acc | F1 | AUROC |
|-------|-----|-----|-------|
| Majority class | 0.7010 | 0.8242 | — |
| LR on hand-features (lengths/digits/punct/quotes, balanced) | 0.5355 | 0.5621 | 0.6639 |
| Probe baseline (last token / last layer, MLP 896→256→1, 200 ep) | 0.7039 | 0.8097 | 0.7361 |

Baseline probe: train AUROC 0.9999 vs test 0.7361 → severe overfit (586 train samples × 896 features × 256 hidden).

---

## Step 5 — Probe regularisation

`experiments/p5_probe_regularization.py`. Default features (last_seq, last layer).
13 variants, compared by test AUROC and train↔test gap.

**Winner: 5.3e combo** — `SmallerMLPProbe(hidden=64, dropout=0.5, weight_decay=1e-2, epochs=50)`. test AUROC 0.7203 ± 0.037, gap +0.132. Not the highest AUROC (5.3d hidden64 is even higher: 0.7447), but the smallest gap among MLPs → stable basis for feature comparison in step 6.

Key observations:
- Single-factor regularisers do not close the gap. Dropout 0.5 alone **hurts** AUROC (0.7205 < 0.7361 baseline) — Adam + dropout on 689 samples is noisy.
- 200 → 50 epochs cuts train AUROC sharply (1.0 → 0.91); epochs are a real lever.
- sklearn LR (lbfgs) > torch Linear+Adam for the linear model: 0.69–0.72 vs 0.68.
- MassMean standalone: AUROC 0.66, gap 0.02 — linear floor.
- BatchNorm (5.3f) did not help: full-batch training gives BN nothing to regularise.

## Step 5.5 — Dim reduction + hand-fusion (negative)

`experiments/p55_reduction.py`. PCA/SVD `{16,32,64,128,256}` × {LR L2, 5.3e MLP} × {±hand-features} on the last-token vector. Winner unchanged: **hidden-only 5.3e** = 0.7203. Best reduced ≤ 0.7098; hand-features (`hidden+hand` 0.7177) do not help. Nothing to compress on a single last-token vector.

---

## Step 6 — Feature hypotheses

### H1 — Token position (negative)

| Strategy | test AUROC |
|----------|------------|
| **last_seq** (default) | **0.7203 ± 0.037** |
| last_response | 0.6888 ± 0.038 |
| end_of_question | 0.4958 ± 0.037 (≈ random) |

Orgad's hypothesis of +10–18 AUC from changing the position is not confirmed: base Qwen2.5-0.5B *reads* an externally generated response, so the hidden state at end_of_question cannot structurally know about the hallucination. `last_seq` is fixed.

### H2 — Layer scan (small positive)

The script extracts `last_seq` for all 25 layers and trains a 5.3e probe on each.

| Layer | test AUROC | gap |
|-------|------------|-----|
| **13** | **0.7351 ± 0.029** | +0.175 |
| 15 | 0.7330 ± 0.039 | +0.166 |
| 17 | 0.7310 ± 0.042 | +0.158 |
| 24 (default) | 0.7203 ± 0.037 | +0.132 |

Variance ratio R (PRISM proxy) does not work: Pearson(R, AUROC) = 0.0475. Top-K by R `[24, 23, 0, 21, 19]` ≠ Top-K by AUROC `[13, 15, 17, 16, 14]`. Concat of top-K did not beat single layer 13.

### H3 — Eigenvalue features (negative)

INSIDE-style top-k log-eigenvalues + pseudo-logdet over response tokens. Standalone AUROC 0.66; fusion with hidden-only baseline: `hidden+eig10` = 0.7191 vs `hidden_only` 0.7203. Does not help.

### H4 — Response chunks × layers + PCA (winner)

Grid `layers [13, 15, 17, 24]` × `pools [response_head, response_mid, response_tail, last_seq]`, mean over pool positions, flatten = 14336 → fold-local PCA → 5.3e probe.

| Variant | test AUROC | gap | test acc |
|---------|------------|-----|----------|
| **h4_grid_pca16** | **0.7812 ± 0.016** | **+0.025** | 0.7373 |
| h4_grid_pca32 | 0.7724 ± 0.025 | +0.072 | 0.7330 |
| h4_grid_pca64 | 0.7582 ± 0.027 | +0.155 | 0.7301 |
| h4_grid_pca128 | 0.7561 ± 0.033 | +0.213 | 0.7242 |
| h4_layer13_last_seq | 0.7351 ± 0.029 | +0.175 | 0.7140 |

PCA16 is the first strong gain: +0.06 AUROC over the last-layer baseline and +0.05 over single-layer H2. Gap is cut by ~5×. Increasing the number of components immediately inflates the gap.

### H5 — All layers × sampled response tokens (negative)

25 layers × 6 quantile positions (`first/q25/q50/q75/last_response/last_seq`) → flatten 134400 → PCA. Best `pca32` = 0.7552 AUROC, worse than H4 `pca16` (0.7812). Pointwise positions are noisier than chunk-mean.

### H6 — H4 augmentation (winner)

H4 PCA16 + fold-local extra features: scaled hand-features, H3 eigen, Mass-Mean score (PRISM truthfulness direction, train-fold-fit).

| Variant | test AUROC | gap | test acc | test F1 |
|---------|------------|-----|----------|---------|
| **h6_h4_pca16+mm** | **0.7902 ± 0.023** | **+0.028** | **0.7518** | **0.8396** |
| h6_h4_pca16 (H4 baseline) | 0.7812 ± 0.016 | +0.025 | 0.7373 | 0.8271 |
| h6_h4_pca16+topo+mm | 0.7783 ± 0.038 | +0.041 | 0.7532 | 0.8389 |
| h6_h4_pca16+hand+mm | 0.7721 ± 0.034 | +0.035 | 0.7460 | 0.8281 |

Only Mass-Mean yields a positive fusion. PCA strips the high-variance directions; MM brings back an explicit class direction. Hand/topo features do not help on top of H4.

### H7 — Probe head sweep (kept H6)

Features fixed (H6 winner). Vary the classifier head: hidden size, dropout, wd, epochs, plus sklearn LR.

| Variant | test AUROC | gap |
|---------|------------|-----|
| h7_lr_C1.0 | **0.7938 ± 0.016** | +0.106 |
| h7_lr_C0.1 | 0.7906 ± 0.018 | +0.044 |
| **h7_current_5.3e** | 0.7902 ± 0.023 | **+0.028** |
| h7_mlp16_do0.2_wd1e-2 | 0.7699 ± 0.019 | +0.026 |
| h7_mlp8 | 0.6597 ± 0.041 | +0.030 |

LR C=1.0 yields +0.0036 AUROC at the cost of gap +0.106 vs +0.028. Smaller MLPs underfit. We keep the 5.3e MLP — near-best AUROC, best gap, stable across folds.

---

## Step 7 — Threshold tuning

`experiments/p7_threshold_tuning.py`. Model fixed: H4 grid → PCA16 → ⊕ Mass-Mean → 5.3e MLP. Procedure:

1. 5-fold OOF scores for all 689 labelled samples.
2. One global threshold maximising OOF accuracy.
3. Final model trained on all 689 samples; the same threshold is applied to `test.csv`.

Per-fold thresholds: 0.369, 0.395, 0.423, 0.422, 0.425 — stable after fold 1.

| Metric | Value |
|--------|-------|
| threshold | **0.421707** |
| OOF accuracy | **0.7750** |
| OOF F1 | 0.8482 |
| OOF AUROC | 0.7832 |
| OOF predicted positive rate | 0.7808 |
| OOF accuracy @ 0.5 | 0.6749 |

The threshold is critical: default 0.5 yields acc 0.6749 (below majority 0.7010); the OOF-tuned 0.421707 lifts it to 0.7750.
