# Phase 6 — Strong sparse + audited external transfer

Date: 2026-10-31

## Question

Does the already-audited 2,503-row GPT/Gemini external source still improve the stronger Phase 5b sparse attribution model, or was its Phase 2 uplift specific to the weaker char-only baseline?

## Fixed validation contract

- Official train: 2,699 rows / 150 prompts / 18 labels.
- Permanent 5-fold `GroupKFold(prompt_id)` mapping.
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Metric: balanced accuracy.
- Score **only untouched official validation rows**.
- No external row may enter validation.

## Fixed external source

Use `data/external/gpt_gemini_generations.parquet`, already audited in Phase 2:

- 2,503 rows;
- 430 independent prompts;
- 8 exact competition labels;
- 0 duplicate normalized texts;
- 0 exact normalized overlap with official train;
- maximum audited word 1–2 TF-IDF similarity to official train: 0.16553;
- raw identity gate passed.

Represented labels:

- `gemini-3.1-pro-preview`
- `gemini-3.6-flash`
- `gemini-3.8-flash`
- `gpt-5.5`
- `gpt-5.6-luna`
- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `gpt-6-astra`

Use **all 2,503 rows**. Do not rebalance or tune the external subset in this run.

## Fixed model

Keep the Phase 5b anchor unchanged:

- char TF-IDF 1–5;
- word TF-IDF 1–2;
- boundary-char TF-IDF 3–5;
- punctuation-aware TF-IDF;
- deterministic style features with weight `0.2`;
- balanced `LinearSVC(C=1.5, dual=True, max_iter=15000)`.

For each fold, fit every vocabulary/scaler/classifier on that fold's four official training folds **plus all 2,503 external rows**. Transform the official validation fold only after training-side fitting.

## Anchor

Official-only Phase 5b OOF balanced accuracy: **0.36708923688789463**.

## Required output

- OOF balanced accuracy on 2,699 official rows;
- all five fold scores;
- per-model recall;
- mean recall for the 8 externally covered labels;
- mean recall for the 10 uncovered labels;
- deltas versus the Phase 5b anchor;
- full official OOF predictions.

## Decision rule

Promote the external source for the strong-sparse path only if it beats **0.36709** under this unchanged validation contract. Do not infer anything about encoder transfer from this result; Phase 3 already showed that representation-specific transfer can reverse sign.
