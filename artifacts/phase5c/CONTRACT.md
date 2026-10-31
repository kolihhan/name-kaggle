# Phase 5c — Reference RichStyle reproduction

Date: 2026-10-31

## Question

Does the remaining gap between this repo's Phase 5b OOF (`0.36709`) and the public `multi_c150_s020` grouped-CV result come from the deterministic style-feature definition?

## Fixed validation

- 2,699 official rows only.
- Permanent 5-fold `GroupKFold(prompt_id)` mapping.
- Fold fingerprint `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Balanced accuracy on official OOF rows only.
- No external data.

## Fixed model

All sparse branches and classifier settings are identical to Phase 5b:

- char TF-IDF 1–5, max 180k;
- word TF-IDF 1–2, max 150k;
- boundary char_wb 3–5, max 100k;
- punctuation-aware word TF-IDF 1–2, max 100k;
- style branch weight 0.2;
- LinearSVC C=1.5, balanced classes, max_iter=15000.

The only changed component is the style extractor: use the narrower 108-feature RichStyle definition reproduced from the public `arshjohari1/Kaggle-LLM-Challenge` implementation instead of this repo's broader 155-feature Phase 5 extractor.

## Decision

This is a reproduction check, not a hyperparameter search. If it closes the gap, adopt the narrower style definition. If not, keep Phase 5b as the local CPU anchor and stop sparse micro-tuning.
