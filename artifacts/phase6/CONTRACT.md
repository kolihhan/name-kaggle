# Phase 6 — Strong sparse external-data A/B

Date: 2026-10-31

## Question

Does the already-audited 2,503-row GPT/Gemini external source still improve OOF once the official-only sparse baseline is strong (`0.36709`)?

## Fixed validation

- Official train rows: 2,699.
- Permanent 5-fold `GroupKFold(prompt_id)` mapping.
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Metric: balanced accuracy.
- Validation rows are official rows only.
- External rows never enter the metric denominator.
- Condition A must reproduce Phase 5b `0.3670892` within absolute tolerance `0.001` or the run fails.

## External source identity

Expected audited snapshot:

- rows: 2,503;
- labels: 8 exact competition labels;
- exact label counts are enforced in code;
- no blank text;
- no duplicate normalized text;
- no exact normalized overlap with official train.

The source is `data/external/gpt_gemini_generations.parquet`, derived from the Phase 2 audited GPT/Gemini snapshot.

## Conditions

### A — official only

For each fold, fit the unchanged Phase 5b strong-sparse model on the four official training folds.

### B — official + external

For each fold, fit the same Phase 5b model on the same four official training folds plus all 2,503 audited external rows.

Each condition independently fits its TF-IDF vocabularies and style scaler inside its own training data. No validation text is used to fit transforms.

## Fixed model

Identical to Phase 5b:

- char TF-IDF 1–5, max 180k;
- word TF-IDF 1–2, max 150k;
- boundary `char_wb` 3–5, max 100k;
- punctuation-aware token TF-IDF 1–2, max 100k;
- deterministic style branch × 0.2;
- `LinearSVC(C=1.5, class_weight="balanced", dual=True, max_iter=15000)`.

## Required output

- A and B aggregate OOF balanced accuracy;
- all five fold scores;
- aggregate delta B-A;
- per-model recall delta;
- mean recall delta for the 8 external-covered labels;
- mean recall delta for the 10 uncovered labels;
- OOF predictions for both conditions.

## Decision

If B improves honest official OOF, external data is promoted for the canonical sparse path. If B does not improve, keep Phase 5b official-only as the CPU anchor and stop treating the Phase 2 uplift as transferable to the stronger feature union.
