# Phase 3 — Frozen ModernBERT result

Date: 2026-10-31

## Result

Using the permanent 5-fold `GroupKFold(prompt_id)` validation contract and scoring only the 2,699 official rows:

| Condition | OOF balanced accuracy | Delta |
|---|---:|---:|
| Official only | **0.30224** | — |
| Official + 2,503 promoted external rows | **0.29041** | **-0.01183** |

The external source therefore **fails the frozen-encoder admission gate**.

## Strongest diagnostic

The external source covers exactly eight competition labels. Recall decreased for **all 8 / 8 covered labels** when those external rows were added:

- `gemini-3.1-pro-preview`: -0.1200
- `gemini-3.6-flash`: -0.1000
- `gemini-3.8-flash`: -0.1267
- `gpt-5.5`: -0.0333
- `gpt-5.6-luna`: -0.1467
- `gpt-5.6-sol`: -0.0933
- `gpt-5.6-terra`: -0.1000
- `gpt-6-astra`: -0.1867

This is stronger evidence of representation/domain mismatch than a small aggregate fluctuation: the source hurts every class it directly supplies under this frozen encoder representation.

## Decision

**Do not promote these 2,503 rows into the first trainable-encoder experiment.**

The Phase 2 sparse result remains valid for sparse character features, but it does not transfer to frozen ModernBERT. Keep the external snapshot for later targeted experiments; do not treat it as generally beneficial training data.

Next experiment: trainable ModernBERT on **official data only** under the same frozen folds. External-data reintroduction requires a separate ablation after the official-only trainable baseline exists.

## Important limitation

The frozen encoder itself scores 0.30224, below the Phase 1 sparse char baseline (~0.31930). This does **not** reject trainable ModernBERT: the Phase 3 contract intentionally tests frozen representation transfer, not end-to-end fine-tuning quality.
