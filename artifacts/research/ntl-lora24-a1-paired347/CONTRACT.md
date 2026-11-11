# Frozen independent LoRA24 A1 — paired 347-row sibling evidence

Status: **PREREGISTERED before candidate inference.** Date: 2026-11-11 MYT.

## Source & objective

Evaluate the *existing*, own, frozen Stage-1 A1 `Qwen3-8B-Base` LoRA24 checkpoint, not the Stage-2 `0.520` Kaggle candidate. No training or checkpoint changes. The scientific question is whether this independent encoder's correct predictions *complement* own official Public-BA `0.679` champion on the exact 347 previously scored fresh external rows, especially provider sibling mistakes.

Checkpoint: Kaggle dataset `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`; Qwen-3 8B Base model `qwen-lm/qwen-3/transformers/8b-base/1`; runtime `kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2`. Use the existing `load_exact_stage1_checkpoint`, `predict_heldout`, checkpoint manifest verification and frozen `max_length_before_eos` in repository source, no new checkpoint or fitting. Prior evidence [original Gate A](../../candidates/two-stage-lora24-v1/RESULT.md).

Two exact external datasets and source-file SHA (read-only):
- `nimithashreea/ntl-proxy-candidate-generations-20261110`, `generated_proxy_candidates.csv`, 240 rows, SHA256 `25d21134cecbddd1b556ec4c3b81fb58dd22cb3102ec7bad967d4191364e167e`, CC BY 4.0 with Seyit prompt-bank attribution.
- `toanplam/ntl-fresh-generation-pilot-nov08`, `passages.jsonl`, 107 rows, SHA256 `89a9d0cec7d434dc30bfa6ad303233821eaead8800e396fab6039a7dfeafdf17`, CC0-1.0.
- 12 true labels, 18 possible predicted labels. 29 source-local prompt groups, 347 unique model+prompt responses.
- Original champion predictions: *already frozen*, private Kaggle kernel `kolihhan/ntl-0679-fresh-transfer-diagnostic`, output `ntl_0679_transfer_predictions.parquet`. Obtain it only to a temporary GitHub runner for **paired comparison**, never print/commit row-level labels/text. Exact champion diagnostic [RESULT](../ntl-0679-transfer-diagnostic/RESULT_20261111.md).

## Predeclared metrics & decision

1. A1 standalone conditional balanced accuracy over these **12 observed true labels**, preserving **all 18 predicted classes**, source-wise and combined.
2. Paired rescues: champion wrong & A1 correct. Regressions: champion correct & A1 wrong. `net=rescue-regression`. Report both by source, with provider family/Gemini/GPT slices and error overlaps. Do **not** blindly blend or override.
3. Honest quality screen: evidence is interesting only if `rescue>regression` in **both** independently generated sources, and net evidence covers material Gemini/GPT sibling errors. If not, reject A1 as a competitive source-transfer verifier; still report upper-bound oracle rescue counts without implying realizable model gain.
4. Reuse 347 rows for **diagnostics only**; this set's champion errors have already been inspected, so no claim of unbiased model selection or fresh external transfer, even if positive. No label-specific thresholds, blend optimization, fixed exception rules, per-family calibrations or Public LB submission.
5. Never compare 12-label conditional BA directly with official 18-label Public BA 0.679. No claim that A1 is an original new architecture (it's an existing own checkpoint); this is evidence of complementarity.
6. Fail closed if Kaggle model assets/checkpoints cannot be loaded exactly, file hashes/counts differ, GPU resources unavailable, or private outputs mismatch. Preserve source provenance.

## Resource/safety

One Kaggle **private inference-only** Notebook, no training/backprop/optimizer, no competition submission, offline T4. No raw competition data, public external text, private model weights or credentials committed to public GitHub. GitHub runner may read explicit CC-licensed public datasets transiently only to verify metadata; training texts stay private inside Kaggle and output consists solely of prediction IDs/labels and aggregate counts. Existing runner workflow must not repeat model runs for score-seeking.

The user requested new substantive research to progress beyond 0.679, not new paid training. This is the lowest-risk independent pretrained signal available with verified weights and a complete inference pipeline.
