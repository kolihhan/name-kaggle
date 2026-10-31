# Phase 9 — precompetition five-class gated specialist contract

Date frozen: 2026-10-31

## Question

Does the conservative 2,921-row pre-competition external candidate improve the promoted Phase 7 OOF system when it is used only as a five-class specialist for labels that Phase 7 did not previously cover?

## Baseline

Reproduce Phase 7 in the same run:

1. canonical official-only strong sparse anchor;
2. canonical Phase 7 eight-class gated specialist using `data/external/gpt_gemini_generations.parquet`;
3. require the reproduced Phase 7 balanced accuracy to match the promoted result (~0.370052) up to implementation-level floating precision.

Do not compare against Phase 5b alone.

## Validation statement

Official validation remains the frozen five-fold `GroupKFold(prompt_id)` OOF and only official validation rows are scored.

The Phase 9 external sources/prompt banks were independently evidenced as public before the competition opened on 2026-10-14 and have zero exact/normalized answer-text overlap with official training answers. Official training data does not expose prompt text, so residual accidental prompt identity cannot be directly observed or mathematically excluded. Therefore this experiment must **not** be described as provably prompt-leakage-free.

External rows are training-only and never scored as validation.

## Frozen external candidate

Exactly 2,921 unique responses from these four pre-competition sources:

- `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`
- `greghavens/kimi-k3-coding-and-debugging-traces`
- `Zek-Takai/glm53-flash-harvest`
- `zlab-princeton/SWEeper-Bench-traces`

Exact class counts:

- `claude-opus-4-6`: 1,934
- `glm-5.3`: 200
- `glm-5.3-flash`: 284
- `kimi-k3`: 317
- `qwen3.8-max`: 186

No other external rows are permitted.

## Frozen specialist

Five classes only:

- `claude-opus-4-6`
- `glm-5.3`
- `glm-5.3-flash`
- `kimi-k3`
- `qwen3.8-max`

For each fold, train on:

- official fold-training rows whose label is one of the five classes;
- all 2,921 frozen external rows.

Use the same sparse feature builder and `LinearSVC` configuration as the canonical anchor/Phase 7 specialist. No new hyperparameters.

## Frozen gate

For each official OOF row:

- if the **Phase 7 final prediction** is one of the five Phase 9 labels, replace it with the Phase 9 specialist prediction;
- otherwise leave the Phase 7 prediction unchanged.

This structurally guarantees that recall for the other 13 true labels cannot decrease: rows already correctly predicted as one of those 13 labels are never touched, while rows from those labels that Phase 7 mispredicted into the five-class set remain inside the five-class set.

## Promotion rule

Promote Phase 9 only if all are true:

1. reproduced Phase 7 baseline is aligned with the canonical promoted Phase 7 result;
2. final official OOF balanced accuracy is strictly greater than the reproduced Phase 7 balanced accuracy;
3. recall for every one of the 13 labels outside the Phase 9 specialist set is exactly unchanged (absolute tolerance `1e-12`);
4. Phase 7 predictions outside the five-class gate are never changed;
5. no post-hoc threshold, class weight, C, label-set, feature, row-selection, or gate sweep is performed.

If it fails, record the negative result and stop this exact route. Do not tune on the permanent folds.
