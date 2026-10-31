# Phase 11 — global Fable 5 binary override contract

Date: 2026-10-31

## Question

Can an independently admitted pre-competition Claude Fable 5 response corpus improve the promoted Phase 7 system through a fixed binary Fable-vs-rest detector, without tuning a multiclass gate against the permanent folds?

## Frozen validation

- Official rows: **2,699**
- Official prompts: **150**
- Labels: **18**
- Validation: permanent 5-fold `GroupKFold(prompt_id)`
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`
- Metric: balanced accuracy
- Scored rows: official validation rows only

No external row may be scored.

## Frozen baseline

Reproduce **Phase 7** exactly in the same run:

1. reproduce the canonical Phase 5b strong-sparse anchor;
2. reproduce the fixed Phase 7 eight-class GPT/Gemini specialist using exactly `data/external/gpt_gemini_generations.parquet`;
3. apply the original Phase 7 gate unchanged.

Reference Phase 7 OOF balanced accuracy: **0.3700521998508576**.

Reference Phase 7 `claude-fable-5` recall: **0.14093959731543623**.

The Phase 7 baseline implementation and predictions are not tunable in Phase 11.

## Frozen Fable 5 external source

Source: `TheFusionCube/Fable-5-CoT-Traces`

Immutable revision:

`3624e3a8aab36a89dc3b89aebef0e034a64ea04e`

License: **MIT**.

Use exactly the response-only corpus admitted by `artifacts/phase11/fusioncube_fable5_audit.json`:

- raw rows seen: **468**;
- source-defined `category == decoy` rows excluded: **115**;
- singleton model-usage dict rows attributable only to `claude-fable-5`: **13**, retained;
- mixed `claude-fable-5` + `claude-opus-4-8` provenance rows: **9**, excluded;
- retained Fable responses: **344**;
- non-empty: **344**;
- normalized-unique: **344**;
- normalized official answer-text overlap: **0**;
- normalized overlap with already admitted Phase 7 GPT/Gemini text: **0**;
- classifier text: `response` only; prompt text is excluded.

Any drift from these counts is a contract failure.

## Frozen detector

For each permanent fold:

1. take only the four official training folds;
2. use **all official training rows from all 18 labels**;
3. map official `claude-fable-5` rows to binary positive `claude-fable-5`;
4. map every other official label to binary negative `__other__`;
5. append all **344** admitted external Fable responses as positive `claude-fable-5` rows;
6. fit the exact Phase 5b strong-sparse feature construction on this binary detector training set:
   - char TF-IDF 1–5, `lowercase=False`, `min_df=2`, max 180,000;
   - word TF-IDF 1–2, whitespace tokenization, `lowercase=False`, `min_df=2`, max 150,000;
   - boundary `char_wb` TF-IDF 3–5, max 100,000;
   - punctuation-aware word TF-IDF 1–2, max 100,000;
   - deterministic style features, weight `0.2`;
7. fit the exact balanced `LinearSVC(C=1.5, class_weight=balanced, dual=True, max_iter=15000)`;
8. call the classifier's native `predict` on every official validation row.

There is no tuned confidence threshold, probability calibration, sample weighting, external-data weighting, resampling, or class-specific threshold.

## Frozen override

For each official validation row:

- if the binary detector predicts `claude-fable-5`, output `claude-fable-5`;
- otherwise keep the Phase 7 prediction unchanged.

Consequences of this fixed rule:

- Phase 11 may add Fable predictions globally;
- it may never change a Phase 7 Fable prediction away from Fable;
- it may never replace a Phase 7 non-Fable prediction with another non-Fable label.

These invariants must be asserted in code.

## Promotion rule

Promote Phase 11 only if all are true:

1. final OOF balanced accuracy is greater than the Phase 7 reference by more than numerical equality (`atol=1e-12`, `rtol=0`);
2. final `claude-fable-5` recall is strictly greater than **0.14093959731543623**;
3. no Phase 7 Fable prediction is removed;
4. every changed prediction changes **to** `claude-fable-5` and no other label-to-label rewrite occurs.

If any condition fails, reject Phase 11.

## One-shot rule

This contract authorizes **one** permanent-fold evaluation after implementation tests are green.

After the result is visible, do not tune on the same folds:

- the native SVC decision threshold;
- `C` or class weights;
- Fable external sample subsets;
- decoy/mixed-provenance inclusion;
- positive/negative weights or resampling;
- feature ranges or feature caps;
- override exceptions;
- per-class routing rules.

A rejected result requires genuinely new evidence or a separately justified mechanism frozen before another permanent-fold evaluation.

## Kaggle rule

Do not submit Phase 11 to Kaggle unless it passes the promotion rule first.
