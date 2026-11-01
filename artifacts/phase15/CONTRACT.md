# Phase 15 — provider-terms-clean public sparse contract

Date: 2026-11-01

## Question

Does the exact promoted Phase 14 four-view sparse recipe still beat the last submission-eligible promoted system after external passages blocked or unresolved under the repository's conservative provider-output-terms standard are removed?

## Why this is a new authorized shot

Phase 15 is not a post-hoc attempt to rescue Phase 14's score. The only change is an independently justified compliance filter frozen from provider-terms evidence after detecting an inconsistency between Phase 13 and Phase 14 admission standards.

The filter is documented in `artifacts/phase15/TERMS_AUDIT.md` and was applied in an audit-only workflow before this contract. No Phase 15 permanent-fold result has been observed.

## Frozen validation

- Official rows: **2,699**
- Official prompts: **150**
- Labels: **18**
- Validation: permanent 5-fold `GroupKFold(prompt_id)`
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`
- Metric: balanced accuracy
- Scored rows: official validation rows only

No external row may be scored.

## Baselines

Submission-eligible reference:

- Phase 7 OOF balanced accuracy: **0.3700521998508576**
- Phase 7 verified Kaggle Public LB: **0.369**

Terms-blocked diagnostic reference only:

- Phase 14 OOF balanced accuracy: **0.4178100919711658**

Phase 14 is not the promotion threshold because its full 43,781-row external pool does not satisfy the same provider-output-terms standard used by Phase 13.

## Frozen external data

Start from the exact Phase 14 conservative subset rebuilt from:

- Kaggle dataset: `nishantdahal/ntl-handwriting-blend`
- file: `pool_v4.parquet`
- raw rows: **75,783**
- raw SHA-256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`
- Phase 14 conservative subset rows before terms filtering: **43,781**

Apply exactly the frozen exclusion from `TERMS_AUDIT.md`:

```text
gemini-3.1-pro-preview
gemini-3.6-flash
gemini-3.7-flash
gemini-3.8-flash
glm-5.3
glm-5.3-flash
kimi-k3
```

Audit-only workflow `41270419945` then established:

- blocked rows removed: **17,756**
- retained external rows: **26,025**
- retained external prompt IDs: **2,801**
- retained external labels: **11**
- blocked labels remaining: **0**
- null text/model: **0**

Frozen retained counts:

| Label | Rows |
|---|---:|
| claude-fable-5 | 1,193 |
| claude-opus-4-6 | 1,352 |
| claude-opus-4-8 | 1,364 |
| claude-opus-5 | 1,240 |
| claude-sonnet-5 | 1,350 |
| gpt-5.5 | 3,525 |
| gpt-5.6-luna | 4,326 |
| gpt-5.6-sol | 3,879 |
| gpt-5.6-terra | 3,875 |
| gpt-6-astra | 3,044 |
| qwen3.8-max | 877 |

Official competition rows for all 18 labels remain untouched.

## Frozen model recipe

Use the exact Phase 14 feature/model recipe with **no parameter changes**.

For each permanent fold, fit on the four official training folds plus all 26,025 terms-clean external rows.

Feature union:

1. char TF-IDF 3–5, max 250,000, weight `1.0`;
2. word TF-IDF 1–3, max 180,000, weight `0.7`;
3. shape-char TF-IDF 2–6, max 60,000, weight `0.5`;
4. small/function-word TF-IDF 1–3, max 50,000, weight `0.5`.

Shared vectorizer configuration:

- `lowercase=False`
- `sublinear_tf=True`
- `min_df=2`
- `dtype=float32`
- word token pattern `(?u)\b\w+\b|[^\w\s]`

Classifier:

`LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")`

No threshold, class weights, view weights, `C`, feature caps, source subsets, label subsets, resampling, or post-hoc routing may be tuned after seeing the result.

## Required invariants

The one-shot run must assert:

1. raw pool SHA and raw row count match Phase 14;
2. Phase 14 conservative subset rebuilds to exactly 43,781 rows;
3. terms-clean subset is exactly 26,025 rows / 2,801 prompt IDs with the frozen 11-label counts above;
4. no blocked provider label remains in external training data;
5. every external row is training-only;
6. OOF has 2,699 unique official IDs and no null predictions;
7. permanent fold fingerprint is unchanged.

## Promotion rule

Promote Phase 15 and authorize a second Kaggle submission only if all are true:

1. final OOF balanced accuracy is greater than Phase 7 `0.3700521998508576` by more than numerical equality (`atol=1e-12`, `rtol=0`);
2. all required invariants pass;
3. no provider-terms-blocked external label appears in training.

Phase 15 is allowed to score below Phase 14; Phase 14 is a diagnostic upper reference, not an eligibility gate.

If Phase 15 fails the Phase 7 promotion gate, do not submit it.

## One-shot rule

This contract authorizes **one** Phase 15 permanent-fold evaluation after implementation tests are green.

After the result is visible, do not tune this recipe on the same permanent folds. A future shot requires genuinely new evidence or a separately justified mechanism frozen before evaluation.
