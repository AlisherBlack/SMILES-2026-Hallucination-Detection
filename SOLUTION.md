# SOLUTION

## Pipeline

```
H4 chunk-layer grid (14336-d)
  └─ StandardScaler          (fold-local)
  └─ PCA(16)                 (fold-local)
  └─ ⊕ Mass-Mean score       (fold-local θ, normalised by train μ/σ)
  └─ MLP head: 17 → 64 → 1   (ReLU, Dropout 0.5, wd=1e-2, Adam, 50 ep)
  └─ threshold = 0.421707    (OOF-tuned for accuracy)
```

H4 grid = layers `[13, 15, 17, 24]` × pools `[response_head, response_mid, response_tail, last_seq]`, mean over pool positions, flattened → 14336.

OOF on 5-fold CV (seed 42): **accuracy 0.7750**, F1 0.8482, AUROC 0.7832. Majority baseline accuracy 0.7010.

## Reproduction

```bash
pip install -r requirements.txt
python solution.py                            # → results.json + predictions.csv
```

`python solution.py` yields `test AUROC 0.7885` / `test accuracy 0.7649` / `test F1 0.8460` (per-fold val-tuned threshold).

Artefacts produced by the run:
- `results.json` — per-fold metrics → [results.json](https://disk.yandex.ru/d/wVasAariVgt5vg)
- `predictions.csv` — final labels for `data/test.csv` → [predictions.csv](https://disk.yandex.ru/d/tHXS3lBCHD6JAg)

Threshold 0.421707 is hard-coded in `probe.py` as the default from the p7 OOF calibration. `fit_hyperparameters()` overrides it per-fold inside `evaluate.run_evaluation` (val-tuned); for the final fit in `solution.py`, `fit_hyperparameters` is not called, so 0.421707 is used.

The plan that drove the final result is in `./docs/plan.md`.

Experiment scripts live in `./experiments`. Example:
```bash
python -m experiments.h6_h4_augmented
```

## What was modified

| File | Change |
|------|--------|
| `aggregation.py` | H4 grid (14336-d) instead of last-token last-layer + helpers for locating the response via offsets / prompt-len / fallback |
| `probe.py` | StandardScaler → PCA(16) → ⊕ Mass-Mean → MLP(64, dropout 0.5, wd 1e-2, 50 ep), threshold 0.421707 |
| `splitting.py` | 5-fold StratifiedKFold (seed 42), 15% of train+val → val |
| `solution.py` | Had to be touched: the `aggregate(...)` contract was extended with a third argument `response_positions`. In both extraction loops the tokenizer is now called with `return_offsets_mapping=True`; for each sample response positions are computed (via offsets, otherwise via `tokenizer.encode(prompt)` length, otherwise a fallback to the last token) and passed to `aggregate`. Without this, `aggregate(hidden_states, attention_mask)` could not separate prompt from response. |

`evaluate.py` and `model.py` were not modified.

## What contributed

| Change | test AUROC | Notes |
|--------|-----------|-------|
| Baseline: last token / last layer + MLP 256, 200 ep | 0.7361 | train AUROC 1.0000, gap +0.264 |
| → regularised head (5.3e: 64 hidden, dropout 0.5, wd 1e-2, 50 ep) | 0.7203 | gap +0.13, stable basis for feature comparison |
| → middle layer 13 instead of 24 (H2) | 0.7351 | +0.015 |
| → H4 grid (4 layers × 4 chunks) + fold-local PCA(16) | **0.7812** | **+0.046**, gap +0.025 |
| → + Mass-Mean score (H6) | **0.7902** | +0.009, best acc/F1 |
| → OOF threshold 0.421707 (vs 0.5) | — | OOF accuracy 0.7750 vs 0.6749 |

Main architectural change: response-localised layer × chunk grid → PCA. Main submission-pipeline change: OOF threshold tuning.

## What did not work

- **H1 — token position.** `end_of_question` ≈ random (AUROC 0.50), `last_response` worse than `last_seq` (0.69 vs 0.72). Reason: base Qwen *reads* an externally generated response — the hidden state at the end of the question cannot yet "know" about the hallucination. Orgad's hypothesis does not transfer from instruct models.
- **H3 — eigenvalue / topological features.** Weak standalone (≤0.66 AUROC); fusion with hidden-only / with H4 yields no gain.
- **H5 — 25 layers × 6 quantile positions + PCA.** Best `pca32` = 0.7552, worse than H4 `pca16` = 0.7812. Pointwise positions are noisier than chunk-mean.
- **5.5 — PCA/SVD on one last-token vector + hand-features.** Winner remained hidden-only 0.7203. Nothing to compress; hand-features (lengths, punctuation, digits) overlap with the hidden state.
- **H2 variance-ratio screening (PRISM).** Pearson(R, AUROC) = 0.0475 — the proxy does not work on our dataset. Layers in the final grid were chosen by actual AUROC, not by R.
- **H7 — sklearn LR / smaller MLP on the final features.** LR C=1.0 gives +0.0036 AUROC but gap +0.106 vs +0.028 — unstable. Smaller MLPs (hidden=8/16) underfit. Kept 5.3e MLP.
- **BatchNorm in probe head.** Full-batch training gives BN nothing to regularise; StandardScaler already normalises the input.
- **Dropout 0.5 as the sole regulariser.** On 689 samples it hurts AUROC (0.7205 vs 0.7361). It only works in combination with smaller hidden size and fewer epochs.

Per-fold tables — `docs/experiments.md`; plan — `docs/plan.md`.
