# Phase 5b — Official-only strong sparse anchor

Date: 2026-10-31

## Question

Does the remaining gap after Phase 5 come mainly from its separate LogisticRegression views and equal probability blend rather than from missing sparse/style signal?

## Validation

- 2,699 official rows only.
- Permanent 5-fold `GroupKFold(prompt_id)` mapping.
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Balanced accuracy on official OOF rows only.
- Every vectorizer and scaler is fitted inside its training fold only.
- External rows: 0.

## Fixed configuration

One feature union, selected before seeing this run:

- char TF-IDF 1–5, case/punctuation preserved, max 180k;
- word TF-IDF 1–2, whitespace tokens, case/punctuation preserved, max 150k;
- boundary `char_wb` TF-IDF 3–5, max 100k;
- punctuation-aware token TF-IDF 1–2, max 100k;
- Phase 5 deterministic style features scaled fold-locally and multiplied by 0.2;
- `LinearSVC(C=1.5, class_weight="balanced", dual=True, max_iter=15000)`.

This is a single prior-art-backed anchor, not a grid search.

## Decision

Compare against Phase 5 blend `0.34821` and the independent public grouped-CV reference range around `0.374–0.376`. If it closes the gap, promote it as the canonical CPU anchor. If it does not, inspect the specific gap before adding more configuration search.
