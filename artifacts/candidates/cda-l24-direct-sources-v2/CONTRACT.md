# CDA-L24 direct-original-source v2 — one locked Kaggle submission

Freeze: 2026-11-08 before viewing any completed C0/C1 GPU transfer predictions.

## Hypothesis and training

Test whether class-conditional source-adversarial gradients improve an independently task-adapted Qwen3-8B LoRA24 model's exact-generator identity representation. This is an application of existing domain-adversarial learning, not a new GRL algorithm. Our own LoRA24 Public=0.520; reproduced third-party Best=0.668, not a paired baseline.

Use exactly two first-party external source datasets, `arxitekton/ntl-claude-gen-v1` and `trnhip/ntl-api-gen`, as continued Stage-A training. These sources are NOT proven free of competition-host distribution conditioning. Their provenance limitations must be reported. External holdout is the untouched-by-training `kilupy/ntl-18-model-corpus` using 80 stable SHA-selected original prompts per independently written 5-Claude/13-other prompt block, selected before observing any candidate scores. This checks transfer across original prompt banks with 18 target labels, **not** matched 18-model prompts or complete protocol-version invariance. Reject exact-answer overlap with original A1 fit/selected train. Same selected rows for both arms.

C0 vs C1 uses SAME A1 checkpoint, 24-layer LoRA, class/source training rows, optimizer/epochs/seed/head. Both train class-conditional source head; C0 reverses feature gradient with lambda=0, C1 with frozen lambda=0.10. One external-stage epoch, no model/data hyperparameter sweep and no additional generator calls.

**Observed data-construction limitation and preregistered engineering fix before any successful GPU run:** arx+trn together have no examples of `gemini-3.7-flash` but A1 knows all 18. Do not invent example rows. Continue training identity loss only over 17 represented classes, mask missing label logit in *training CE only*, preserve existing 18-way head and all 18 logits in evaluation and final official Stage B. Fix is applied identically to C0/C1. If any OTHER official label goes missing, fail. Official full final Stage B uses all 18 labels.

## Evaluation and stopping

Primary diagnostic: on exactly the preselected original Kilupy heldout rows, compare C0/C1 balanced accuracy over all 18 true model labels; publish per-label recall and provider macro recall. Paired bootstrap is by original prompt ID (5,000 seeded draws), no new slicing or tuning. Report score whether positive or negative. Retired v1 arxitekton heldout and spent official grouped folds are NOT independent evidence.

The user specifically requested a real **one-shot Kaggle competition submission**, not another loop of selective CPU tuning. Therefore this project runs ONE frozen C1 final-fit/inference submission even if the external-transfer diagnostic does not beat C0, provided both actual checkpoint outputs are complete, identity/provenance passes, code is executable and hidden inference is feasible. In that case label it **diagnostic submission / not scientifically promoted**; do not claim improvement, reweight, retune or resubmit from Public leaderboard feedback. The grouped official OOF coordinate remains historically required by repo EVALUATION_POLICY; **if not executed, clearly report its absence** instead of implying full policy compliance. Repeated OOF is spent adaptive evidence, not a substitute for independent transfer.

Final inference: Kaggle Code Competition, offline Qwen3-8B model source plus private C1 checkpoint, original 2-epoch official full final fit, plain 18-way head, submission.parquet exact `id/model`; test labels never train or tune. Use exact bytes, private weights and complete manifest SHA256; do not upload official rows, raw external prompts, private weights or private predictions to public GitHub. One official submission only, verify accepted ID and terminal Public score, compare 0.668 as historical reproduction.

If C1/C0 kernels ERROR, record the actual traceback and fix only provable engineering bugs with another explicitly numbered version. Do not run a different research direction, source/weight search, or falsely claim a score. Preserve both GPU kernels for audit after a result exists. No training performed by GitHub runner (only private Kaggle GPU kernels).