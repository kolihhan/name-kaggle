# Phase 7 — Gated external specialist contract

Date: 2026-10-31

## Question

Can the audited GPT/Gemini corpus improve the canonical Phase 5b strong-sparse anchor when it is isolated in an 8-class specialist and prevented from changing the anchor's covered-vs-uncovered decision?

## Frozen validation

- Official rows: **2,699**
- Official prompts: **150**
- Labels: **18**
- Validation: permanent 5-fold `GroupKFold(prompt_id)`
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`
- Metric: balanced accuracy
- Scored rows: official validation rows only

## Frozen anchor

Reproduce Phase 5b exactly:

- char TF-IDF 1–5, `lowercase=False`, `min_df=2`, max 180,000
- word TF-IDF 1–2, whitespace tokens, `lowercase=False`, `min_df=2`, max 150,000
- boundary `char_wb` TF-IDF 3–5, max 100,000
- punctuation-aware TF-IDF 1–2, max 100,000
- deterministic style features, weight `0.2`
- `LinearSVC(C=1.5, class_weight=balanced, dual=True, max_iter=15000)`
- no external rows

Canonical reference OOF balanced accuracy: **0.36709**.

## Frozen external source

Use exactly `data/external/gpt_gemini_generations.parquet`:

- **2,503 rows**
- **430 prompt IDs**
- raw identity gate already passed
- prompt bank reported as 1,000 new prompts
- zero exact normalized-text overlap with official train in the Phase 2 audit

Covered labels are exactly:

1. `gemini-3.1-pro-preview`
2. `gemini-3.6-flash`
3. `gemini-3.8-flash`
4. `gpt-5.5`
5. `gpt-5.6-luna`
6. `gpt-5.6-sol`
7. `gpt-5.6-terra`
8. `gpt-6-astra`

The other 10 competition labels are uncovered.

## Frozen specialist

For each fold:

1. take only the four official training folds;
2. keep only official rows from the 8 covered labels;
3. append all 2,503 audited external rows;
4. fit the exact Phase 5b strong-sparse feature construction inside this specialist training set;
5. fit the exact Phase 5b balanced `LinearSVC(C=1.5)` as an 8-class classifier;
6. predict all official validation rows.

## Frozen gate

For every official validation row:

- if the Phase 5b anchor prediction is one of the 10 uncovered labels, keep it unchanged;
- if the Phase 5b anchor prediction is one of the 8 covered labels, replace it with the specialist prediction.

No confidence threshold, score blending, probability calibration, learned router, class-specific rule, or fallback is allowed.

This gate cannot change covered-vs-uncovered membership. Therefore all 10 uncovered per-class recalls must be exactly identical to the anchor; any drift is a contract failure.

## Promotion rule

Promote Phase 7 only if both are true:

1. ensemble OOF balanced accuracy is strictly greater than the reproduced Phase 5b anchor;
2. all 10 uncovered per-class recalls are exactly preserved.

If Phase 7 does not beat the anchor, reject this mechanism. Do not tune gate thresholds, specialist sample counts, class weights, SVC C, covered labels, or other parameters on the same permanent folds.
