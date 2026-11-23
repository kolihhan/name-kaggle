# ATLAS A0 paired-error attribution audit — frozen contract (2026-11-23)

## Question

The completed ATLAS A0 readout changed 21/300 labels against the reproduced 53-feature C_global model: 6 rescues and 15 regressions. Before considering any new model, determine what the existing fold-held-out C_global model relied on in those two discordant groups, and whether ATLAS margins are already strongly represented by its five paired global features.

## Frozen protocol

- Reuse the exact official 2,699-row TRAIN and permanent five prompt-disjoint folds.
- Reuse the exact pinned 800 + 1,899 public GLM token-cache parts; verify source SHA256s before use.
- Recompute the same 300 GLM-5.3 / GLM-5.3-Flash labelled rows and the exact five-fold StandardScaler + LogisticRegression(C=1, class_weight=balanced, seed=20261122) 53-feature C_global.
- Recompute exact ATLAS A0 argmax-prod at T={1,2,4,8}; no new LLM forward calls.
- Gate exact reproduction of C_global BA 0.94333333, ATLAS BA 0.91333333, 6 rescues, and 15 regressions.
- For every held-out row, decompose the C_global logit into standardized feature contributions. Aggregate only the two paired-error groups; also report margin association, ATLAS-margin correlation with the five existing pair features, and coefficient sign stability across folds.
- No response text, row IDs, raw token arrays, or per-row prediction records may be persisted. The workflow prints aggregate JSON only and deletes recovered source material.

## Interpretation limits

- This is a diagnostic on repeatedly consumed official TRAIN rows, not independent validation or the hidden 18-class benchmark.
- Feature contributions explain the fitted fold model's arithmetic; they do not prove causal feature importance.
- The 6 rescue cases are too few to establish independent ATLAS signal by themselves.
- No new model fitting, feature blending, temperature/threshold tuning, champion change, or Kaggle submission is authorized by this audit.
- Only if the audit identifies a credible nonredundant signal should a separate, predeclared nested prompt-grouped experiment be considered.

## Execution

Workflow: [ntl-atlas-paired-error-audit-20261123.yml](../../.github/workflows/ntl-atlas-paired-error-audit-20261123.yml)  
Script: [ntl_atlas_paired_error_audit_20261123.py](../../scripts/ntl_atlas_paired_error_audit_20261123.py)

No paid APIs, model inference, QLoRA, GPU, or Kaggle submission.
