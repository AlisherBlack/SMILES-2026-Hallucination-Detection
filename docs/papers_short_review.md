# Short paper digest

> ⚠ **This is a pre-experiment forecast.** The document was written before any
> runs and records what was planned from the literature. The actual results
> often diverged from the forecast (e.g. Orgad's +10–18 AUC token-position gain
> turned out negative for us — base Qwen2.5 reads an external response; PRISM's
> variance-ratio Pearson with AUROC came out 0.0475 instead of the ~0.77 from
> their paper). Final numbers and confirmed/rejected hypotheses are in
> `docs/experiments.md`. The hypothesis numbering here (H1–H4) does not match
> the final one (H1–H7 in `plan.md`).

Mapped to hypotheses from `docs/plan.md` (H1–H5).



## 1. SAPLMA — Azaria & Mitchell, 2023

https://arxiv.org/abs/2304.13734

Canonical supervised baseline on hidden states (ancestor of our method). Method:
last token of statement → MLP probe.

What we take:
- Layer scan — the main result (Tables 1–2): on LLaMA2-7B / OPT-6.7B the optimum
  is not the last layer but mid-to-late layers. For Qwen2.5-0.5B (24 layers)
  proportional candidates: layers 12 (middle), 15, 18, 21. → H2
- Probe architecture 256→128→64, ReLU, Adam, few epochs (~5). Compared to our
  MLP 256 × 200 ep — our train AUROC is 0.9999, classical overfit. → H3
- Threshold tuning on val — already in `evaluate.py`.

What we skip: 3-seed averaging, dataset generation methodology.



## 2. INSIDE — Chen et al., ICLR 2024

https://arxiv.org/abs/2402.03744

⚠ Crucial: INSIDE is unsupervised, K=10 answers per question. We are
single-response. EigenScore in its original form is inapplicable.

What we take:
- Middle-layer confirmation (Figure 3b: optimum ~ layer 17/32 on LLaMA-7B).
  Consistent with SAPLMA. → H2
- Last-token pooling > mean pooling in their experiments (Section 4.3). An
  argument to test last/mean honestly rather than swap the default unchecked. → H1
- Eigenvalue features within a single answer (adapting EigenScore from K samples
  to response tokens): top-k log-eigenvalues of the hidden-state covariance
  across response tokens + log-determinant. +5–10 scalars in
  `extract_geometric_features`. → H4

What we skip: K-sampling, the EigenScore metric, optional feature clipping.


## 3. Orgad et al., ICLR 2025 — most relevant

https://arxiv.org/abs/2410.02707

Setting identical to ours: white-box, single-response, supervised probing.

What we take (top priority):
- Token position — the biggest lever (+10–18 AUC, Table 1):
  - End-of-question token (last token of the prompt before `<|im_start|>assistant`):
    easy to find, strong "model can / cannot answer" signal. → H1
  - Last response token (not last sequence token): find the boundary before
    `<|endoftext|>`/`<|im_end|>`, take the token before it. Our current default
    uses the last real token of the whole sequence — this is different. → H1
  - Exact answer token — potentially the best but requires extraction. Include
    only if EDA shows responses are long with the answer in the middle.
- LogReg L2 as probe (Appendix A.2): cheaper, regularised; in their experiments
  it beats the MLP. On 689 samples it is the #1 candidate against overfit. → H3
- Heatmap layer × token — methodology for choosing the optimum. → H1 × H2

What we skip: K=30 resampling, exact answer via an instruct LLM, p(True) prompting.

---

## 4. PRISM — Zhang et al., 2025

https://arxiv.org/abs/2411.04847

⚠ Crucial: PRISM is about cross-domain generalisation. Our domain is
homogeneous (EDA: `unique_prompts_train = 689`, shared format,
`cosine(centroid_train, centroid_test)` ~0.95+). The PRISM prompt itself is not
useful for us. We take only the methodology.

What we take:
- Variance ratio R = Vθ/V_T (Eq.6) as a cheap screening metric. For each
  (layer, token) pair we measure the fraction of variance along
  `θ = mean(X⁺) − mean(X⁻)`. Pearson corr with final AUROC ≈ 0.77 (Section
  5.5.1). Across 25 layers × N token strategies — seconds, no probe training.
  Speeds up the H1×H2 sweep. → H2
- Mass-mean probe (Marks & Tegmark 2023): predict = sigmoid(X · θ / ‖θ‖).
  No training. If close to MLP, the signal is linear and the MLP overfits.
  Third mandatory baseline in `experiments/`. → H3

What we skip: prompt wrapping (requires a second forward + chases a
cross-domain effect we do not have), leave-one-domain-out.

---

## Convergence across papers

| Aspect | SAPLMA | INSIDE | Orgad | PRISM | Result |
|---|---|---|---|---|---|
| Token | last statement | last middle layer | exact answer / end-of-q | last wrapped | H1: end-of-q + last response |
| Layer | sweep, mid-to-late | middle | middle-to-late | last/middle | H2: middle/late + variance-ratio screening |
| Probe | 3-layer MLP 256-128-64 | n/a | LR L2 | MM | H3: LR L2 + MM |
| Geom features | — | EigenScore | — | variance ratio | H4: eigenvalue features |

---

## Hypothesis priorities (by expected gain)

See `docs/plan.md` step 5 for details. Briefly:

1. H1 (token position) — Orgad, +10–18 AUC. Do first.
2. H2 (layer scan) — +5–10 AUC. Screen via variance ratio (PRISM).
3. H3 (probe LR L2 + MM) — stability ±2–5 AUC, protection against overfit.
4. H4 (eigenvalue features) — +1–3 AUC. Optional, only if H1–H3 do not close the day.
