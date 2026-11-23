# Generative Token-Transition Fingerprint v1 — real five-fold result

Date: 2026-11-23 MYT. This result applies the [unchanged, pre-scoring mechanism contract](ntl-generative-transition-v1-contract-20261123.md). Branch: `research/ntl-generative-transition-fingerprint-v1-20261123`.

## Execution evidence

- Real [GitHub Actions #45106442160](https://github.com/kolihhan/name-kaggle/actions/runs/45106442160), terminal `SUCCESS` at 2026-11-23 03:34 UTC, scientific runtime 165 s.
- Code: [`scripts/ntl_generator_transition_fingerprint_v1_20261123.py`](../../scripts/ntl_generator_transition_fingerprint_v1_20261123.py). Causal Markov models use only official **train** 2,699 labels and the permanent five prompt-disjoint groups (30 prompts per heldout fold). No hidden test, no Kaggle competition submission, no pretrained weights, no 0.715 source or predictions, no ensemble, no GPU.
- Model-specific abstract token vocabulary sizes 97–101 by fold, including function words, anonymized content-word lengths/case, punctuation, numbers, and explicit UNK. The language model is a **synthetic observable Markov proxy**, NOT the actual GLM/GPT/Claude conditional token likelihood.
- Compared four models on identical labels and folds. Strong Sparse control reproduced known frozen Phase 5b 0.36709 all-row BA. The generative LM arms use identical fold-local priors and locked smoothing; shuffled-control class-specific unigrams and bigrams' global prior remain unchanged.
- Unit sanity for probability normalization, arithmetic cancellation of common language-model residual, label-independent deterministic within-response shuffle all passed.

## Real scores

| Model | Historical full-18 five-fold OOF BA |
|---|---:|
| A: class-conditional unigram token LM | **0.2444743** |
| B: class-conditional observed-order bigram LM | **0.1892916** |
| C: same conditional LM fit on within-passage shuffled-order training tokens | **0.1537534** |
| D: original direct Strong Sparse 4-view+style LinearSVC | **0.3670892** |

All five B−C fold BA deltas are positive. Fixed 1000-draw prompt-bootstrap results:

| Paired effect | OOF BA change | 95% bootstrap interval | B rescues / regressions |
|---|---:|---|---:|
| **B−A** | **−0.0551827** | **[−0.0718519, −0.0370370]** | 255 / 404 |
| **B−C** | **+0.0355382** | **[+0.0188433, +0.0522457]** | 392 / 296 |
| **B−D** | **−0.1777977** | **[−0.1985327, −0.1562887]** | 240 / 720 |

All metrics use same heldout official *train* rows, not Public scoring; the raw Kaggle hidden label distribution and true private final metric are unknown.

## Scientific interpretation — avoid a false success

**FACT:** Genuine-order conditional evidence clearly improves the counterpart trained on **synthetically random text order**. This comparison shows that real order matters to this proxy under a fixed token marginal distribution.

**BUT:** This is a **weak, distribution-mismatched negative control**: the randomized-order model is deliberately forced to learn unnatural word transitions and then tested on natural sequences. Hence B>C alone does **not** prove useful generator-identity-specific conditional information *beyond unigram counts*. The A unigram control is critical here and decisively beats B by 5.52 percentage points. Thus H1's value-add claim **FAILS**.

**FACT:** The actual Strong Sparse single-model reference is substantially better, 0.3671 vs 0.1893. No competitive stand-alone candidate was produced, and class-specific recalls are generally weak.

**INFERENCE:** Unigram/word-shape frequency signatures survive more reliably than sparse model-specific next-style-token transition probabilities in this 99-token abstraction. Possible explanations include poor context support from only ~120 training responses/model/fold, over-generalized content-word abstraction, weak smoothing/calibration assumptions, tokenization mismatch, and remaining source/protocol variation. These are **hypotheses**, not proven causes. Do not tune bin counts, smoothing or context length against this outcome; the fixed candidate did not meet its predeclared conditions.

## Gate and next work

- Predeclared mechanism PASS required B−A >=+.01, B−C >=+.01, positive bootstraps for both, ≥3/5 B>C wins, per-class safety. Observed B−A negative with negative CI: **GATE FAIL**.
- **RETIRED:** exact class-conditional 99-style-token Markov surrogate as a stand-alone 18-way attribution model. Do not make a submission, blend with 0.715, or alter this candidate post hoc to rescue BA.
- **NOT retired:** meaningful *true* generator-specific conditional token evidence. Independent prior in-repo GLM full vs Flash genuine next-token log-likelihood yields 89.33% **binary** balanced accuracy on old paired train and only +5/252 rescues under a constrained external observational filter; see [real GLM frozen-likelihood result](../../artifacts/research/ntl-glm-frozen-likelihood-20261114/RESULT.md). This does not establish 18-way hidden correctness.
- For a scientifically differentiated next major experiment, require **pretrained contextual token probability** with explicit source-conditioned adapters or response-generation log-odds, compare with a matched context-destroyed/lexical-only control on independently sourced **full-18** records with verified versions, prompt/protocol/source separation. The older frozen [conditional-denoising18-v1 contract](../../artifacts/candidates/conditional-denoising18-v1/CONTRACT.md) already specifies a larger such model; review its current readiness and data provenance before inventing another variant. That contract is not thereby authorized to skip its gates.
- Keep official 0.715 Public Champion as a *competition metric reference*, not a component of model or source code. The public score is not estimable by a direct conversion of these OOF figures.

## Evaluation confidence

Strong validity for historical same-row B/A/C/D comparison and exact heldout folds, limited validity for future hidden test because the 2,699 official rows have been adaptively reused. CI describes variation over previously seen Prompt groups, not new-source generalization. No independent complete 18-class source-disjoint benchmark has yet been certified in repository policy.
