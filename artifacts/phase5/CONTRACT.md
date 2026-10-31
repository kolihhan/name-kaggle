# Phase 5 — Official-only style multiview baseline

Date: 2026-10-31

## Question

Can a cheap, explicit authorship-style model close the visible gap between the Phase 1 char TF-IDF baseline and the stronger public prompt-grouped baselines before we spend more GPU compute?

## Fixed validation contract

- Official rows only: 2,699.
- Permanent 5-fold `GroupKFold(prompt_id)` mapping.
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Metric: balanced accuracy.
- All TF-IDF vocabularies, numeric scalers, and classifiers fit inside each training fold only.
- No external data.

## Fixed views

1. Character TF-IDF: char 1–5 grams, case and punctuation preserved, `min_df=2`, `max_features=200000`.
2. Word TF-IDF: whitespace-token 1–2 grams, case and punctuation preserved, `min_df=2`, `max_features=150000`.
3. Numeric style/typography features: length, lexical diversity, word/sentence length statistics, character classes, punctuation/quote/dash/markdown rates, and function-word frequencies.

Each view uses `LogisticRegression(C=4.0, class_weight="balanced", solver="liblinear")`.

## Blend

Equal-weight geometric mean of the three fold-local probability matrices. No blend weights are tuned on OOF.

## Required output

Record char, word, style, and blend OOF balanced accuracy; all five fold scores; per-model recall; and OOF predictions for official rows.

## Decision rule

Treat this as a fixed first style baseline. Compare it against Phase 1 `0.31930` and the public ~`0.374–0.376` grouped-CV reference range. Do not tune on the same result before first inspecting which view and which labels account for the gap.
