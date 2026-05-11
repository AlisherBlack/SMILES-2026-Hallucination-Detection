Main metric: test accuracy, ideally above majority 0.701.
Since accuracy depends on the threshold, during experiments we use a secondary
metric — test AUROC — and aim for a small train↔test gap. At the very end we
tune the threshold for accuracy.


Across all experiments

Experiment metric. To compare variants we use test AUROC —
it is threshold-independent. Test accuracy/F1 are shown in the tables at
threshold=0.5 (or naively tuned) for reference. **Final threshold tuning is
done once, on the winner** (step 7). Before step 7 we do not look at thresholds.

---

1. [done] Setup: `random_state=42` in splitting.py, `torch.manual_seed(42)` in probe.fit(),
   fixed seed (random/numpy/torch) in experiments/.

2. [done] EDA, see docs/eda_results.md.

3. [done] Baselines (CV=5 StratifiedKFold, see docs/experiments.md):
   - Majority: acc 0.7010, f1 0.8242
   - LR hand-features: acc 0.5355, AUROC 0.6639
   - Probe baseline (default agg, last-token last-layer): test acc 0.7039, AUROC 0.7361

4. [done] Paper review: docs/papers_short_review.md.

5. [done] Probe regularisation. Beat the overfit (train AUROC=0.9999).
   Winner: 5.3e combo (`SmallerMLPProbe(hidden=64, dropout=0.5,
   weight_decay=1e-2, epochs=50)`). test AUROC 0.7203 ± 0.037, gap +0.13.
   Not the highest AUROC (5.3d 0.7447), but the smallest gap among MLPs. See
   docs/experiments.md step 5.
   All variants on default features (last_seq, last layer, dim=896), CV=5.
   Compared by test AUROC and train↔test gap.

   - 5.0. `HallucinationProbe` baseline (MLP 896→256→1, 200 ep, no reg).
   - 5.1. sklearn LR L2: `C ∈ {0.01, 0.1, 1.0, 10.0}`, `class_weight='balanced'`,
     lbfgs. Exact baseline from Orgad.
   - 5.2. torch Linear via Adam (logreg via Adam): same training loop as the
     MLP but without the hidden layer; `weight_decay ∈ {0, 1e-2}`. Isolates
     "depth" (vs 5.0) and optimiser (vs 5.1).
   - 5.3. Regularisation of the baseline MLP — one factor at a time over 5.0:
     - 5.3a. + dropout 0.5 (hidden 256, wd=0, 200 ep).
     - 5.3b. + weight_decay 1e-2 (hidden 256, dropout=0, 200 ep).
     - 5.3c. 200 ep → 50 ep (hidden 256, no dropout, no wd).
     - 5.3d. hidden 256 → 64 (no dropout, no wd, 200 ep).
     - 5.3e. combo: hidden 64 + dropout 0.5 + wd 1e-2 + 50 ep.
   - 5.4. Mass-Mean (PRISM): `θ = mean(X⁺) − mean(X⁻)`. No training.
   Script: `experiments/h45_probe_regularization.py`.
   Winner — probe with the best test AUROC and gap ≤ 0.15.
   - 5.5. Dim reduction + hand-feature fusion on the default aggregation.
   Before the next step we check whether we can improve the chosen anti-overfit
   probe without changing token/layer. Separate control: if there is a gain,
   we run H1/H2/H3 with the improved feature pipeline.
Candidates:
     - hidden only → 5.3e baseline
     - hidden + scaled hand_features
     - PCA/SVD(hidden) → LR L2
     - PCA/SVD(hidden) + hand_features → LR L2
     - PCA/SVD(hidden) → 5.3e
     - PCA/SVD(hidden) + hand_features → 5.3e
  Result: the best variant remained 5.5a_hidden_only_MLP_combo (`test AUROC
0.7203 ± 0.037`, gap +0.1324). Hand-features and PCA/SVD did not improve the
hidden-only pipeline: `hidden+hand` 0.7177, the best reduced variant ≈ 0.7098.
For the following experiments we fix hidden-only 5.3e combo.


6. [done] Feature hypotheses — on top of the step 5 winner.
   Each hypothesis isolates one factor; everything else as in the baseline.
   - 6.1. H1 — token position (Orgad). [done, negative]
     last_seq (default) is best: AUROC 0.7203. last_response 0.6888,
     end_of_question 0.4958 (≈ random). Reason: base Qwen reads an externally
     generated response → end-of-question does not know about the
     hallucination. `last_seq` is fixed. See docs/experiments.md.
   - 6.2. H2 — layer scan (SAPLMA + INSIDE + Orgad). [done]
     Screened via variance ratio R = Vθ/V_T (PRISM, Eq.6); top-3 → full probe
     training.
     Best single layer: 13 (`test AUROC 0.7351`, Δ +0.0149 vs layer 24).
     Concat of top-K by R worse than layer 13. Variance ratio as a proxy did
     not work: Pearson(R, AUROC)=0.0475.
     File: `experiments/h2_layer_scan.py`.
   - 6.3. H3 — eigenvalue geometric features (INSIDE). [done, negative]
     Top-k log-eigenvalues of the hidden-state covariance over response tokens
     + pseudo-logdet. Standalone eigen features give a weak signal (up to
     0.6570 AUROC); fusion does not improve the hidden-only baseline:
     `hidden_only` 0.7203, `hidden+eig10` 0.7191. Not taken to the final
     pipeline. File: `experiments/h3_eigen_features.py`.

   - 6.4. H4 — response chunks × layers + PCA. [done, positive]
     Instead of picking one position by hand we take a response-localised grid:
     layers `[13, 15, 17, 24]` × pools `[response_head, response_mid,
     response_tail, last_seq]`, flatten to 14336 features and run fold-local
     PCA before the 5.3e combo probe. Motivation: Orgad's heatmap layer×token +
     SAPLMA/INSIDE middle-layer signal. Not a repeat of 5.5: PCA receives new
     information from different layers and parts of the response, rather than
     compressing one last-token vector.
     Winner: h4_grid_pca16 — `test AUROC 0.7812 ± 0.016`, gap +0.0255,
     `test acc 0.7373 ± 0.034`. Best result so far.
     File: `experiments/h4_chunk_layer_pca.py`.

   - 6.5. H5 — all layers × sampled response tokens + PCA. [done, negative]
     Generalisation of H4: instead of 4 layers and chunk-mean, all 25
     hidden-state layers (`0..24`) and fixed response positions: `first`,
     `q25`, `q50`, `q75`, `last_response`, `last_seq`. Flatten:
     `25 × 6 × 896 = 134400` features, then fold-local PCA (`16/32/64`) before
     the 5.3e combo probe. Goal — check whether a finer Orgad-style
     layer×token grid yields a gain over H4.
     Result: no gain over H4. Best variant h5_grid_pca32: `test AUROC
     0.7552 ± 0.037`, gap +0.0762. Better than H2 but worse than H4
     `pca16` 0.7812. Pointwise token positions are less stable than
     chunk-mean. File: `experiments/h5_layer_token_pca.py`.

   - 6.6. H6 — H4 augmentation: hand/topological/MM features. [done, positive]
     Take the best H4 representation (`chunk-layer grid → fold-local PCA16`)
     and add fold-local extra features: scaled hand-features, H3
     eigen/topological features, Mass-Mean score (PRISM truthfulness
     direction, fit on the train fold only). Check whether weak standalone
     signals add value on top of H4 for AUROC/accuracy.
     Winner: h6_h4_pca16+mm — `test AUROC 0.7902 ± 0.023`,
     gap +0.0279, `test acc 0.7518 ± 0.020`, `test f1 0.8396 ± 0.007`.
     Hand/topological features do not add value on top of H4+MM.
     File: `experiments/h6_h4_augmented.py`.

   - 6.7. H7 — probe head sweep on H6 features. [done, keep H6]
     Features fixed as the H6 winner: `H4 chunk-layer grid → fold-local PCA16
     → + Mass-Mean score`. Vary only the MLP/head: hidden size, dropout,
     weight decay, epochs, plus a sklearn LR sanity check. Hypothesis: after
     PCA16+MM the input is ~17-dim, so the current `hidden=64/dropout=0.5`
     may be excessive.
     LR C=1.0 gives slightly higher AUROC (`0.7938 ± 0.016`) but a much worse
     gap (+0.1056 vs +0.0279). LR C=0.1 is almost equal to H6 in AUROC
     (`0.7906`) but also worse in gap. Result: keep h6_h4_pca16+mm with the
     current 5.3e MLP head as the most stable trade-off.
     File: `experiments/h7_probe_head_sweep.py`.


7. [done] Threshold tuning for accuracy + final solution.
   Final model chosen before threshold tuning:
   H6/H7 keep-H6 = `H4 chunk-layer grid → fold-local PCA16 →
   Mass-Mean score → 5.3e MLP head`.

   Script: `experiments/p7_threshold_tuning.py`.
   - 5-fold OOF scores for all labelled samples.
   - One global threshold over OOF scores, maximising accuracy.
   - Stability diagnostics: per-fold thresholds, OOF accuracy/F1/AUROC,
     predicted positive rate.
   - Final model trained on all labelled data.
   - OOF threshold applied to `test.csv` scores.
   - Outputs: `oof_scores.csv`, `test_scores.csv`, `predictions.csv`,
     `threshold_results.json`.

   The threshold is not used for model selection, only for final
   binarisation.

   Result: OOF threshold 0.421707, OOF accuracy 0.7750, OOF F1 0.8482,
   OOF AUROC 0.7832, predicted positive rate 0.7808. Accuracy at
   threshold 0.5 was 0.6749, so the final threshold matters substantially for
   the ranking metric. Saved files: `oof_scores.csv`, `test_scores.csv`,
   `predictions.csv`, `threshold_results.json`.

8. [done] SOLUTION.md: reproduction instructions, final solution and rationale,
   what was tried and why it did not work.
