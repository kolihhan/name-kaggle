# Phase 3 — Frozen ModernBERT A/B contract

Date: 2026-10-31

## Question

Does the promoted 2,503-row external GPT/Gemini source still help when the text representation changes from sparse character TF-IDF to a frozen pretrained encoder?

This is a representation-level admission test before any GPU fine-tuning or LoRA work. It is not intended to be the final competition model.

## Fixed validation contract

- Official train rows: 2,699.
- Official validation folds: the permanent 5-fold `GroupKFold(prompt_id)` mapping from Phase 1.
- Validation rows: official competition rows only.
- External rows never enter the metric denominator.
- Metric: balanced accuracy.
- The official-only and official+external conditions use identical official validation indices in every fold.

## Fixed encoder configuration

- Model: `answerdotai/ModernBERT-base`.
- Encoder weights: frozen.
- Max sequence length: 256 tokens.
- Representation: attention-mask-aware mean of `last_hidden_state`, then L2 normalization.
- Downstream classifier: `LinearSVC(class_weight="balanced", C=1.0)`.
- Each text is encoded once; the same embeddings are reused across all five folds.

## A/B conditions

### A — official only

For each fold, fit the classifier only on the four official training folds.

### B — official + promoted external

For each fold, fit the same classifier on the same four official training folds plus all 2,503 promoted external rows.

The external source is the Phase 2 snapshot already admitted by the raw-data gate and sparse frozen-CV ablation.

## Required output

`artifacts/phase3/frozen_modernbert_ablation.json` must record:

- aggregate OOF balanced accuracy for A and B;
- all five fold scores for A and B;
- per-model official OOF recall for A and B;
- per-model recall delta, B minus A;
- aggregate delta, B minus A;
- encoder/data/classifier configuration.

## Interpretation rule

- If B improves honest official OOF, external data is promoted to the first trainable-encoder experiment.
- If B does not improve, do not assume the external source transfers to trainable encoders; inspect per-model effects before spending GPU compute.
- Frozen ModernBERT performance by itself does not decide whether fine-tuned ModernBERT can work; this experiment isolates representation transfer and external-data effect first.
