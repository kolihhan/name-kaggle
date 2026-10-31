# Phase 10 — Opus 4.8 nine-class specialist contract

Date: 2026-10-31

## Question

Can the newly admitted pre-competition Claude Opus 4.8 ARC response corpus improve the promoted Phase 7 system when added as exactly one new covered label, without changing the frozen validation, sparse model family, or routing rule?

This is a **single permanent-fold shot**. No result-dependent tuning is allowed after this contract is committed.

## Frozen validation

Use the existing permanent validation unchanged:

- official rows: **2,699**
- official prompt groups: **150**
- labels: **18**
- validation: 5-fold `GroupKFold(prompt_id)`
- fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`
- metric: balanced accuracy
- scored rows: official validation rows only

The promoted comparison baseline is Phase 7:

- balanced accuracy: **0.3700521998508576**
- `claude-opus-4-8` recall: **0.20666666666666667**

## Frozen anchor and features

Reproduce the canonical Phase 5b anchor exactly:

- char TF-IDF 1–5, `lowercase=False`, `min_df=2`, max 180,000
- word TF-IDF 1–2, whitespace tokenization, `lowercase=False`, `min_df=2`, max 150,000
- boundary `char_wb` TF-IDF 3–5, max 100,000
- punctuation-aware word TF-IDF 1–2, max 100,000
- deterministic style features, weight `0.2`
- `LinearSVC(C=1.5, class_weight=balanced, dual=True, max_iter=15000)`

No feature, classifier, class-weight, or C changes are permitted.

## Frozen external data

### Existing Phase 7 corpus

Use exactly the Phase 7 frozen GPT/Gemini snapshot:

- **2,503** rows
- the same eight labels already covered by Phase 7
- no newly added rows from later versions of the public dataset

### New Opus 4.8 corpus

Use exactly the audited assistant responses reconstructed from:

- dataset: `arcprize/arc_agi_v2_public_eval`
- immutable revision: `026789c1c12a4c34580a32e84dcaf5630d7e8f31`
- license: MIT
- high task files: 120
- medium task files: 120
- max task files: 90
- retained assistant responses: **868**

Extraction is fixed:

- include only `metadata.choices[*].message.content` where `message.role == "assistant"`;
- require `metadata.model == "claude-opus-4-8"` for every used attempt;
- exclude all user messages, ARC prompts, structured ARC `answer` fields, and aggregate `results.json` files;
- use all 868 admitted responses exactly once; no sampling, weighting, truncation, setting-specific selection, or deduplication tuning.

The admission audit already established 868/868 non-empty unique normalized responses, zero official answer-text overlap, zero overlap with the admitted GPT/Gemini external corpus, and no missing assistant responses.

## Frozen specialist

The Phase 10 specialist labels are exactly the Phase 7 eight plus `claude-opus-4-8`:

1. `claude-opus-4-8`
2. `gemini-3.1-pro-preview`
3. `gemini-3.6-flash`
4. `gemini-3.8-flash`
5. `gpt-5.5`
6. `gpt-5.6-luna`
7. `gpt-5.6-sol`
8. `gpt-5.6-terra`
9. `gpt-6-astra`

For each fold:

1. take only the four official training folds;
2. keep only official rows from these nine labels;
3. append all **2,503** frozen Phase 7 external rows;
4. append all **868** admitted ARC Opus 4.8 assistant responses labeled `claude-opus-4-8`;
5. fit the exact frozen strong-sparse feature construction inside this specialist training set;
6. fit the exact frozen balanced `LinearSVC(C=1.5)` as a nine-class classifier;
7. predict all official validation rows.

## Frozen gate

Use the same structural gate as Phase 7, with the covered set expanded from eight to nine labels:

- if the Phase 5b anchor prediction is one of the nine covered labels, replace it with the nine-class specialist prediction;
- otherwise keep the Phase 5b anchor prediction unchanged.

No confidence threshold, score blending, calibration, learned router, class-specific rule, fallback, or post-hoc exception is allowed.

The remaining nine uncovered labels therefore must have per-class recall exactly identical to the Phase 5b anchor and Phase 7. Any drift is a contract failure.

## Promotion rule

Promote Phase 10 only if all are true:

1. OOF balanced accuracy is **strictly greater than 0.3700521998508576** (Phase 7);
2. `claude-opus-4-8` recall is **strictly greater than 0.20666666666666667** (Phase 7);
3. all remaining nine uncovered-label recalls are exactly preserved;
4. the run uses the exact frozen data and configuration above.

Otherwise reject this mechanism as tested. Do not tune sample counts, settings, weights, gate thresholds, C, features, class weights, or label membership on the same permanent folds.

## Kaggle rule

This contract authorizes only the local permanent-fold experiment. It does **not** authorize a second Kaggle submission. A Kaggle submission requires a separate decision after the frozen OOF result is known.
