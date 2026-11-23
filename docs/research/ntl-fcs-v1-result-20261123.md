# FCS-v1 — Frozen pretrained DistilGPT2 causal-surprisal results

**2026-11-23 MYT; authentic scientific job completed SUCCESS.** This is a *historical OOF* research experiment, NOT Kaggle Public BA nor an independent full-18 external transfer test.

## Ground truth and execution

- [Completed GitHub Actions run #45107714630](https://github.com/kolihhan/name-kaggle/actions/runs/45107714630) (2026-11-23 03:44–04:05 UTC).
- The [contract](ntl-fcs-v1-contract-20261123.md) was frozen at commit `608b2ada9e496d8a630fabea2a45cb250aa89c7b` before the job produced any score.
- Script: [`scripts/ntl_fcs_v1_20261123.py`](../../scripts/ntl_fcs_v1_20261123.py); pretrained actual `distilbert/distilgpt2` @ `f241065e938b44ac52db2c5de82c8bd2fafc76d0`, FP32, Transformers 4.57.1, PyTorch 2.6.0+cpu, all genuine teacher-forced next-token log probability normalized by full vocabulary.
- Dataset: exactly 2,699 official TRAIN responses, 150 prompts, 18 classes, frozen 5-fold prompt-group heldout, TRAIN-only fold-local 18-way LogisticRegression readouts (55 summary features), no source/protocol access, no public/test labels, no Kaggle submission or externally fitted Winner artifacts.
- Token cap: first 127 BPE pieces. **1,381 / 2,699 passages truncated**. Therefore the experiment measures **prefix likelihood**, not entire response evidence.
- Model forward cost: **535.7 s** for genuine text and **534.1 s** for token-permuted negative control; total science wall-clock **1,226.2 s (20m26s)** on GitHub CPU (excludes Python dependency install).
- Strong Sparse single-system control reproduced `0.3670892` BA on identical OOF rows and prompt splits (not combined with LM).

## Scores

| Arm | Same-row 18-class OOF BA |
|---|---:|
| A: BOS-only wordpiece marginal + unchanged summary/readout | **0.1748198** |
| B: actual pretrained contextual next-token likelihood summaries | **0.1952026** |
| C: class-label-independent shuffled wordpiece order + pretrained contextual likelihood | **0.1718568** |
| D: independently fitted Strong Sparse 4-view LinearSVC | **0.3670892** |

- **B−A: +0.0203828**, 1,000 prompt-bootstrap 95% CI **[+0.0051852,+0.0363173]**; 238 rescues, 183 regressions, 1,499 changed predictions, **5/5 fold wins**.
- **B−C: +0.0233458**, 1,000 prompt-bootstrap 95% CI **[+0.0066789,+0.0388889]**; 287 rescues, 224 regressions, 1,803 changes, **5/5 fold wins**.
- **B−D: −0.1718867**, 95% CI **[−0.1937764,−0.1503623]**; 232 rescues, 696 regressions. Therefore **NOT competitively better** than even the existing local single-model sparse reference.
- Maximum per-true-class recall regression B versus A: 0.086667, below 0.15 safety threshold.
- Fold B BA: 0.1777778 / 0.1962963 / 0.2148148 / 0.1944444 / 0.1927203.
- Predeclared **mechanism gate PASS**; competitive-signal gate **FAIL**.
- All CLI result assertions and unit sanity checks passed, Actions status `SUCCESS`.

## What the evidence means

**Supported (conditional on historical dev):** A real pretrained LM's context-conditioned token surprise profile adds a small, reproducible signal relative to its own BOS-only wordpiece probabilities and to an order-shuffled-input control. A and B use the same underlying LM and frozen summary feature schema. The measured deltas are paired over the same source texts, and do not use the 0.715 public Champion.

**Not established:** That the contextual signal is target-generator-specific, stable across generation protocols, unique to the true source model, or powerful enough to improve Kaggle hidden BA. The shuffled-wordpiece control creates unnatural text; its effect alone could not establish useful context identity; the additional BOS-only A comparison is essential. The 2,699 official training texts and their folds are long **spent** for adaptive research. The positive bootstrap CI is conditional on this historical prompt population.

**Interpretation:** The move from hand-built 99-shape-token Markov to native pretrained probabilities repaired the previous **B<A** failure direction, but a global 55-statistic signature from one small reference LM still loses >0.17 BA to ordinary discriminative sparse features. The current ceiling is in identity specificity and representation sufficiency, not a lack of evidence that context ever matters.

## Next scientific decision

- **Keep/continue the broader pretrained conditional-evidence *hypothesis*** because the mechanisms earned a real 18-class, 5/5 fold paired signal.
- **Do not promote this exact 55-summary-feature DistilGPT2 classifier** to a competition candidate, claim it improves 0.715, or attempt heuristic blending/hard routing.
- **Next highest-value work:** test source-conditioned full-token likelihood or source-specific contextual token-choice heads using a genuinely independent full-18, source/prompt/protocol-isolated validation panel; freeze code and conditions first. Existing [conditional-denoising18-v1 contract](../../artifacts/candidates/conditional-denoising18-v1/CONTRACT.md) already designs source-specific masked-token output residual adapters and controls—inspect implementation/feasibility instead of inventing yet another almost-equivalent model. Historical [GLM real-likelihood work](../../artifacts/research/ntl-glm-frozen-likelihood-20261114/RESULT.md) is source-specific but only a limited binary-sibling mechanism test, not global 18-way transfer.
- Next experiment must account for **long-passage truncation (51.2%)** and report per-model distribution, runtime, and errors on unseen-source 18-class data. Avoid a Public leaderboard-driven parameter sweep.

**Competition account champion remains 0.715 Public BA; no new submission.**
