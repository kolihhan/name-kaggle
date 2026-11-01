# Domain separability diagnostic contract

Date: 2026-11-01

## Question

Can a deliberately simple classifier distinguish official competition responses from the frozen external responses used by Phase 7 or Phase 15, after controlling for model identity?

This is an **audit-only transfer diagnostic**, not a competition-model experiment. It does not score model identity, select a Kaggle system, tune external-source membership, or authorize a submission.

## Why model identity must be controlled

Phase 7 external data covers 8 model labels and Phase 15 external data covers 11. A pooled official-vs-external classifier could win trivially by learning which model labels are absent from an external corpus. Therefore domain separability is measured **within each shared model label independently**, then macro-averaged across labels.

## Frozen corpora

- Official: unchanged `data/train.csv`, 2,699 rows / 150 prompt groups / 18 labels.
- Phase 7 external control: exact frozen `data/external/gpt_gemini_generations.parquet`, 2,503 rows / 8 covered labels.
- Phase 15 external: exact provider-terms-clean 26,025-row / 11-label subset reconstructed under the Phase 15 contract from the pinned public pool.

No new external rows are admitted.

## Frozen holdout

Use salt `ntl-domain-separability-v1`.

For each `(comparison, model_label, domain, prompt_id)` independently:

1. compute `sha256(f"ntl-domain-separability-v1|{comparison}|{model_label}|{domain}|{prompt_id}".encode()).hexdigest()`;
2. convert the first 16 hex digits to an integer;
3. hold out the prompt group when the integer modulo 5 equals 0;
4. train on the other prompt groups.

All rows sharing a prompt ID inside one domain/label remain together. No seed search or split replacement is allowed.

If a label/domain lacks at least one training and one held-out prompt group, the audit fails rather than changing the split.

## Frozen classifiers

Run both classifiers independently for every shared model label.

### A. Style/format only

Use deterministic numeric features only, including:

- character and whitespace-token length;
- line/newline structure;
- punctuation rates;
- digit, uppercase, non-ASCII rates;
- backticks / code fences;
- Markdown headings / list markers;
- braces / parentheses;
- URL count;
- average token length.

Fit `StandardScaler` on training rows, then `LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20261101)`.

### B. Lexical sparse

Use word TF-IDF only:

- ngrams 1–2;
- lowercase true;
- `min_df=2`;
- `max_features=20,000`;
- sublinear TF.

Fit the same fixed balanced logistic regression configuration.

No hyperparameter search, threshold tuning, source filtering, per-label tuning, or feature selection is allowed after results are visible.

## Metrics

For each shared model label and each classifier report:

- balanced accuracy for `official` vs `external`;
- official recall;
- external recall;
- train/test row counts and prompt-group counts.

Across labels report macro mean / sample SD / min / max balanced accuracy.

Also report simple style-feature distribution summaries and the largest absolute standardized mean differences as descriptive signals only.

## Interpretation

- A score near `0.5` means the tested feature family does not easily separate the two corpora under this holdout.
- Materially higher scores mean corpus/source differences are learnable even after controlling model identity.
- Compare Phase 15 against the Phase 7 known-transfer control. Absolute domain separability alone is not proof of transfer failure because Phase 7 already transferred reasonably to Public LB.
- This audit cannot prove which features the Phase 15 model used, cannot observe the hidden test distribution directly, and cannot establish prompt leakage.

## Prohibited consequences

Do not use this audit to:

- remove or downweight Phase 15 sources post hoc;
- tune a competition classifier;
- select thresholds from the Public leaderboard;
- claim hidden-distribution identity;
- authorize another Kaggle submission.
