# Phase 6 — Multiview external-data ablation contract

Date: 2026-10-31

## Question

Does the already-audited 2,503-row GPT/Gemini external source still improve the stronger Phase 5 multiview CPU anchor?

Phase 5 official-only OOF balanced accuracy is frozen at **0.36003977131493914** and is reused from committed evidence rather than refit in this phase.

## Validation contract

- Official rows: **2,699**.
- Permanent split: deterministic **5-fold `GroupKFold(prompt_id)`**.
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Metric: balanced accuracy.
- Validation rows: official competition rows only.
- External rows never enter the OOF denominator.
- Phase 5 model configuration is reused unchanged through `build_multiview_model()`.
- No hyperparameter tuning in this phase.

## External snapshot contract

The Phase 2 audited snapshot must have:

- **2,503 rows**;
- **430 prompt IDs**;
- **8 exact competition labels**;
- no unknown competition labels;
- no null `text/model/prompt_id` rows;
- no duplicate exact text rows;
- minimum class count **140**.

Audited raw upstream identity:

- repository: `RohanExploit/kaggle-name-that-llm-top20`;
- path: `data/extra_gpt_gemini/ntl_generations.csv`;
- raw SHA-256: `8e7a66eb4dda074ab80d47f11ce3002f4dccc19fe363ed252c6cfb5ef834b19e`.

The eight represented labels are:

- `gemini-3.1-pro-preview`
- `gemini-3.6-flash`
- `gemini-3.8-flash`
- `gpt-5.5`
- `gpt-5.6-luna`
- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `gpt-6-astra`

## Fixed conditions

### A — official-only reference

Reuse the committed Phase 5 result: **0.36003977131493914**.

### B — all external

For each fold, fit the unchanged Phase 5 multiview model on:

- the four official training folds;
- all **2,503 external rows**.

Score only the held-out official fold.

### C — balanced external

Deterministically sample the minimum class count, **140 rows per represented label**, seed 42:

- 8 labels × 140 = **1,120 external rows**.

Again score only held-out official rows.

## Required outputs

- aggregate OOF balanced accuracy for B and C;
- all five fold scores;
- per-model recall;
- per-model recall delta versus Phase 5;
- mean recall delta on the 8 augmented labels;
- mean recall delta on the 10 untouched labels;
- two OOF CSVs for B and C.

## Decision rule

- If B or C improves the Phase 5 anchor, keep the best variant as the new CPU training recipe.
- If neither improves, keep Phase 5 official-only as the canonical CPU anchor and stop spending time on this external source for sparse models.
- Do not infer encoder transfer from this result; Phase 3 already showed representation-specific mismatch.
