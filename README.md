# Name That LLM — Kaggle

Working repository for the Kaggle competition **Name That LLM**: identify which of 18 language models generated each passage.

Competition: https://www.kaggle.com/competitions/name-that-llm

## Current state

Phase 1 establishes an honest, reusable validation contract before stronger modeling:

- official training data audited;
- deterministic 5-fold `GroupKFold` by `prompt_id`;
- zero prompt overlap between train and validation folds;
- baseline metric: balanced accuracy;
- best current classical baseline: character TF-IDF (1–5 grams) + LinearSVC, OOF balanced accuracy **0.3193**;
- hierarchical lab→model classification did not materially improve the baseline.

The frozen fold assignment is stored in `artifacts/phase1/folds.csv`. All future experiments should reuse it so scores remain comparable.

## Repository layout

```text
data/                    Official competition files
artifacts/phase1/        Audit, frozen folds, baseline results, error profile
```

## Next experiment

Audit public external generations for provenance and label coverage, then test encoder / LoRA approaches against the same frozen official folds. Do not tune against random row splits.

## Data attribution

The competition Data page lists the official dataset under **CC BY-SA 4.0**. See `data/README.md` for source and attribution.
