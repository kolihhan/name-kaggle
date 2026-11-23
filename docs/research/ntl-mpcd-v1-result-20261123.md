# MPCD-v1 — Actual 18-Class Prompt-Disjoint CPU Result

Date: 2026-11-23 MYT. Scientific code and all pass/fail thresholds frozen in [contract](ntl-mpcd-v1-contract-20261123.md) **before** viewing scores.

## Execution and reproducibility

- Authentic completed [GitHub Actions #45104027619](https://github.com/kolihhan/name-kaggle/actions/runs/45104027619). Status `SUCCESS`; the self-test and mathematical gradient checks passed.
- Exact code: [`scripts/ntl_mpcd_v1_20261123.py`](../../scripts/ntl_mpcd_v1_20261123.py), source training 2,699 official rows / 18 classes / 150 prompts; frozen permanent `GroupKFold` 5-fold manifests; no same-prompt leakage across fold; no unseen data.
- Features: 45,000 raw char TF-IDF features → train-only TruncatedSVD(128) + 155 existing style features, total 283-dimensional vector. Mean SVD explained variance ≈ 0.303; no pretrained encoder.
- 3 identically initialized multinomial linear classifiers trained separately per fold with the same C=1.5 L2 penalty, L-BFGS-B max 65. All optimizers reported convergence in 49–63 iterations, and true conditional-loss analytic gradient numerically verified.
- C0 = per-response CE; C1 = C0 + matched *real* prompt-panel **column CE**; C2 = same objective but panel grouping shuffled **within each class** while preserving one model per prompt and group sizes. No Prompt inference or Hungarian assignment on held-out samples.
- CPU wallclock scientific fitting/preprocessing **34.4 s**. No GPU, Kaggle official code submission, 0.715 winner artifact use or hidden test.

## Real results

| Arm | All 2699 same-row OOF Balanced Accuracy |
|---|---:|
| C0 row CE | **0.3007407** |
| C1 genuine prompt matched CE | **0.3014815** |
| C2 fake prompt matched CE negative control | **0.3040741** |

- **C1 − C0:** +0.0007408 BA, 1000x prompt-bootstrap 95% CI **[-0.0055556,+0.0062963]**; 36 rescues, 34 regressions, 177 altered predictions; won **2 / 5** folds.
- **C2 − C0:** +0.0033334 BA; bootstrap CI [-0.0040741,+0.0103796]; 46 rescues, 37 regressions, 185 predictions altered.
- **C1 − C2:** **−0.0025926 BA**; prompt-bootstrap CI [-0.0085185,+0.0033333]; 28 rescues vs 35 regressions.
- Fold BA pairs C0/C1/C2:
  - Fold0: .3092593 / .3111111 / .3222222
  - Fold1: .3074074 / .2981481 / .3092593
  - Fold2: .2962963 / .2962963 / .3000000
  - Fold3: .2925926 / .2833333 / .2870370
  - Fold4: .2981481 / .3185185 / .3018519
- No class drop >0.15 under C1, but this veto alone is insufficient for quality promotion. Historical independent Phase5b stronger sparse model OOF ≈ **0.36709** on the frozen official folds (not same-run readout); all current linear arms are weaker than it.

## Interpretation and scientific decision

**FACT:** Optimizer worked, no CV leakage detected, genuine pairing produced negligible BA improvement and could not outperform the shuffled-panel control. The predeclared C1 ≥0.02, C1–C2 ≥0.01, positive bootstrap lower bounds, ≥4/5 wins, and class safeguard **collectively fail**. Gate status **FAIL**.

**INFERENCE:** In these 128-dimensional compressed lexical + 155-style features, using a prompt-conditional training objective adds little identity-discriminative information beyond ordinary CE. A generic regularization explanation fits the result as well or better than true prompt-specific learning. Negative result is *not* proof every prompt-conditional representation, neural model or data-generating hypothesis is impossible.

**UNKNOWN:** Hidden-test transfer, source/hosting shifts, errors of independent pretrained encoders, interaction with genuine strong generative likelihood heads, and incremental benefit on an unused, legally sourced full 18-class benchmark are all unmeasured. SVD explained variance only describes raw input variance—not identity-information retention.

**DECISION:** Retire exactly MPCD-v1 linear/corpus/objective configuration; do **not** retune column-weight/SVD dimensionality/C, pick a special class or seed, compare its 0.3015 OOF as if same-row with external 0.715 Public, or submit it. The project has now observed *three* related prompt-removal/identity-learning mechanisms with weak/nontransfer results: ModernBERT-LDA prompt residual external transfer FAIL (−0.0358 mean source BA); cross-prompt-prototype neural v2 source transfer FAIL (+0.002685 CI crosses zero); MPCD-v1 row-vs-column conditional 18-way DEV FAIL (+0.000741 vs negative control +0.003333).

**Next differentiated mechanistic bet (not executed automatically):** Improve the **evidence representation and provenance**, not rearrange prompts or tune class-loss parameters. Specifically test a frozen token-level **generator conditional likelihood / sibling log-odds** or content-matched response residual from *independent pre-trained* model family evidence, compare against source/channel fingerprints on truly independent, complete 18-class matched data before any new heavyweight training. Prior current model has at least a dozen expensive adapted experts and specialized GLM code; do not claim novelty for mechanisms already there. Required disjoint truth provenance is presently **BLOCKED** per [independent validation readiness](../../artifacts/research/ntl-18class-independent-validation-gate-20261111.md).

**Long-run target:** surpass real official 0.715 BA, but no fresh official score resulted from this CPU diagnostic.
