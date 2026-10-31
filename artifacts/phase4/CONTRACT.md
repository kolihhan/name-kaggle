# Phase 4 — Official-only trainable ModernBERT contract

Date: 2026-10-31

## Question

What honest grouped-CV performance can a trainable ModernBERT baseline achieve using only the official competition training data?

Phase 3 showed that the 2,503-row GPT/Gemini external snapshot hurts frozen ModernBERT representation transfer, including recall declines for all 8 labels it directly covers. Therefore this first trainable-encoder baseline intentionally uses **zero external rows**.

## Validation contract

- Official rows: **2,699**.
- Labels: **18**.
- Prompts: **150**.
- Split: permanent deterministic **5-fold `GroupKFold(prompt_id)`**.
- Fold fingerprint must equal:
  `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Official `train.csv` SHA-256 must equal:
  `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- Metric: **balanced accuracy**.
- OOF denominator: official rows only.
- External rows: **0**.

If the official data hash or fold fingerprint differs, abort rather than silently changing the experiment.

## Model/config

- Encoder: `answerdotai/ModernBERT-base`.
- Training: full end-to-end fine-tuning.
- Sequence length: **256**.
- Epochs: **2**, fixed; no per-fold best-epoch selection.
- Train batch size: **16**.
- Eval batch size: **32**.
- Optimizer: AdamW.
- Learning rate: **2e-5**.
- Weight decay: **0.01**.
- Warmup ratio: **0.10**.
- Gradient clipping: **1.0**.
- Base seed: **20261031**, offset by fold.
- GPU execution: private Kaggle kernel, NVIDIA T4 requested.

## Required outputs

- `trainable_modernbert_official_cv.json`
  - aggregate OOF balanced accuracy;
  - all five fold scores;
  - per-model recall;
  - full data/model/validation config.
- `trainable_modernbert_official_oof.csv`
  - official row id;
  - prompt id;
  - true model;
  - fold id;
  - OOF prediction.

## Interpretation

This is the first trainable encoder baseline, not a hyperparameter sweep.

- Compare first against the honest sparse baseline (~0.31930).
- Do not reintroduce the Phase 2 external source in this run.
- If the trainable baseline is credible, the next experiment may compare LoRA/full-finetune settings or targeted external-data variants under the same folds.
- If it is weak, diagnose training/representation behavior before expanding the search space.
