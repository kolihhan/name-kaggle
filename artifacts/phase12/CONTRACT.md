# Phase 12 — HealthBench Opus 5 / Sonnet 5 paired specialist contract

Date: 2026-10-31

## Question

Can the newly admitted pre-competition HealthBench-Psych response corpus improve the promoted Phase 7 system by distinguishing only `claude-opus-5` from `claude-sonnet-5`, while making every other class structurally impossible to change?

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
- `claude-opus-5` recall: **0.6133333333333333**
- `claude-sonnet-5` recall: **0.38**
- pair mean recall: **0.49666666666666665**

## Frozen Phase 7 baseline

Reproduce Phase 7 exactly. Do not alter its anchor, eight-class specialist, external snapshot, features, classifier, or gate.

The Phase 7 predictions are the baseline predictions that Phase 12 may modify only under the pair gate below.

## Frozen HealthBench external data

Use only the admitted response-only corpus recorded in `artifacts/phase12/healthbench_anthropic_audit.json`:

- dataset: `mindbench-ai/healthbench-psych`
- source tag: `v1.0.0`
- immutable revision: `7a116ea7b4fefe0c72eabb8398ec390c6375f823`
- source last modified: 2026-08-28, before competition launch
- license: MIT
- `claude-opus-5`: 610 raw rows, **600 retained non-empty `response_text` rows**
- `claude-sonnet-5`: 610 raw rows, **610 retained `response_text` rows**
- total admitted rows: **1,210**

Extraction is fixed:

- classifier text is only `response_text`;
- empty Opus 5 refusal rows are excluded mechanically;
- `correction`, prompt text, benchmark reference information, and all non-target model rows are never classifier text;
- use all 600 admitted Opus 5 responses and all 610 admitted Sonnet 5 responses exactly once;
- no sampling, weighting, truncation, prompt-specific filtering, or deduplication tuning.

The admission audit established:

- 1,210/1,210 normalized-unique retained responses;
- zero within-label normalized duplicate groups;
- zero Opus5/Sonnet5 cross-label response overlap;
- zero official-train response-text overlap;
- zero overlap with the already admitted Phase 7 GPT/Gemini external corpus;
- `training_authorized=true`.

## Frozen paired specialist

The specialist has exactly two labels:

1. `claude-opus-5`
2. `claude-sonnet-5`

For each permanent fold:

1. take only the four official training folds;
2. retain only official rows whose true label is Opus 5 or Sonnet 5;
3. append all 600 admitted HealthBench Opus 5 responses;
4. append all 610 admitted HealthBench Sonnet 5 responses;
5. build the exact canonical Phase 5b strong-sparse feature construction inside this training set;
6. fit the exact frozen `LinearSVC(C=1.5, class_weight=balanced, dual=True, max_iter=15000)`;
7. predict every official validation row, but use these predictions only through the gate below.

No feature, classifier, class-weight, C, threshold, calibration, or external-data weight changes are permitted.

## Frozen pair gate

For each official validation row:

- if the Phase 7 baseline prediction is `claude-opus-5` or `claude-sonnet-5`, replace it with the paired specialist prediction;
- otherwise keep the Phase 7 baseline prediction exactly unchanged.

The paired specialist itself may emit only the two target labels.

This structure implies a hard invariant: the per-class recall of the other **16 labels must be exactly identical to Phase 7**, because no prediction outside the Opus5/Sonnet5 pair can change, and any changed prediction remains inside that pair.

No confidence threshold, margin rule, abstention, blending, learned router, class-specific exception, or post-hoc override is allowed.

## Promotion rule

Promote Phase 12 only if all are true:

1. OOF balanced accuracy is greater than Phase 7 by more than floating-point noise (`delta > 1e-12`);
2. mean recall across Opus 5 and Sonnet 5 is greater than **0.49666666666666665** by more than `1e-12`;
3. all other 16 per-model recalls are exactly preserved;
4. every changed prediction is a swap within `{claude-opus-5, claude-sonnet-5}`;
5. the run uses the exact frozen data and configuration above.

Otherwise reject this mechanism as tested. Do not tune C, thresholds, feature weights, sample counts, prompt subsets, external weights, or the gate using the same permanent folds.

## Kaggle rule

This contract authorizes only this one local permanent-fold experiment. It does **not** authorize another Kaggle submission. A second Kaggle submission requires a separate decision after the frozen OOF result is known.
