# NTL — SemEval-inspired prompt-controlled cross-source fingerprint falsifier

2026-11-24. Status: PREREGISTERED; not scored, no Kaggle competition submission.

## Why this direction

- **AISPACE SemEval-2024 Task 8 Subtask B (1st)**: observed encoder-only attribution performance, class-weighted CE, soft-vote among independent systems. Original source: https://aclanthology.org/2024.semeval-1.212/. Their task has six classes including human-written text and **does not imply** any specific 18-way NTL performance gain.
- **MLab SemEval-2024**: DeBERTa outperformed RoBERTa in its SemEval Subtask B setup: https://aclanthology.org/2024.semeval-1.210/. Do not default to retraining 14B, LoRA or an encoder without transfer evidence.
- **NTL public-source research**: jerrythomas/name-that-llm reports 0.365 -> 0.425 grouped CV after prompt-centering on a separate simpler baseline; this is **third-party self-reported local CV**, not our 0.715 Public BA. https://github.com/jerrythomas/name-that-llm.
- **External data diversification is not enough**. Unlike SemEval, our 2699 official TRAIN responses span 18 model labels and 150 prompt groups, but external publisher C1/C2 cover only 12 model labels and originate from the same publisher; unverified served-model aliases remain a limitation.

## Global bottleneck / competing hypotheses

H1: source-conditioned generation protocol changes surface text fingerprints, so naive pooling of external LLM generations into training data risks shortcut learning and Public LB regression.

H0: prompt-controlled source-conditioned language fingerprints remain stable across official TRAIN and C1/C2 even when domains differ. If they remain stable, a properly provenance-gated small encoder or augmentation experiment could be reopened, **not automatically promoted**.

Test **untrained, label-fixed source-transfer descriptives** *before* considering new training and without any scorer inference.

## Frozen data and comparisons

- Official competition historical TRAIN: exactly 2,699 rows and 18 source labels, downloaded ephemerally with authorized Kaggle token, never published. No adaptive fold tuning. This historical set has been repeatedly analyzed, hence this is a mechanism diagnostic, NOT a new holdout.
- External original publisher Kaggle datasets: `beroli/ntl-benchmark-c1-generated-panels` and `beroli/ntl-benchmark-c2-generated-panels`. Original `passages.csv` SHA256 pins C1 `8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177` (360 records) and C2 `bf25ef1e953d5c37b177f27b841a04098ae9b35f45fc43fa11036d91e8831e02` (470). Both 12/18 exact label coverage, same publisher, **not independently verified source-model identities**.
- Use only the same 12 exact labels in all three sources.
- Features fixed ahead: whitespace-independent regex token word count; mean word length; punctuation per 100 words; digit characters per 100 words; fraction of fixed common function words. All computed identically from raw text, and all aggregated only.
- Remove each prompt's *other* labels' mean for each feature (leave-one-out prompt centering) to control content effects. Require >=6 unique model labels per prompt block; all 12 labels must remain supported after filtering.
- Average within each label and compare **cross-label rank ordering** (Spearman rho, 12-point model fingerprint) between official vs C1, official vs C2, and C1 vs C2.
- Preregister descriptive stability indicator: a feature is stable if rho>=0.35 in BOTH independent comparisons official↔C1 and official↔C2. Gate = >=3 of fixed 5 metrics stable. This gate is descriptive only; no confidence intervals, p-value, or independent data claim. The C1/C2 publisher overlap makes the two checks correlated.
- Report official per-class min/max frequencies and implied inverse-frequency weight ratio. If <1.01, SemEval class-balanced CE is not justified by official TRAIN class imbalance alone. Do not make a supervised loss change.

## Stop / pivot

- Dataset SHA, label support, schema drift or duplicate model-prompt blocks -> hard fail, do not reinterpret.
- If gate fails: park naive 12-class external-pool transfer; pursue generation-channel matching or verified served-model metadata, not more prompt/feature tweaking on this same cohort.
- If gate passes: it only weakly supports surface fingerprint stability, **not** an encoder or 0.715 improvement. Next step is a separately frozen, budget-bounded genuine labelled error diagnostic with a comparable strong scorer; only then consider training or a Kaggle bet.
- No new model fit, no GPU, no paid API, no competition submission, no official test inference, no raw text export, no Public LB tuning, no writes to original 0.715 files.

## Expected artifacts

GitHub Actions terminal log `NTL_SEMEVAL_SOURCE_TRANSFER_REAL_AGGREGATES`: factual support sizes, prompt counts, raw class distribution and precommitted 5 feature correlations. No raw text, prompts, row IDs, predictions or source records.
