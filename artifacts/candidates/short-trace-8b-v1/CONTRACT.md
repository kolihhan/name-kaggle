# Short-TRACE 8B v1 — frozen representation-control experiment

Date: 2026-11-07. **Status: FROZEN PRE-SCORE; not promoted.**

## Research question

On model-author attribution of short passages (unlike the long-form TRACE paper), do **within-passage adjacent token-rank transitions** carry source-transfer information beyond the **marginal rank histogram**, or do they merely reflect rank frequencies, length and source artifacts? This is a test of a representation mechanism, not evidence of novelty of transition fingerprints. Reference: [TRACE, EMNLP 2026](https://github.com/anudeex/TRACE).

## Current evidence and exclusions

The independent LoRA24 Public 0.520 and reproduced V47 0.666 differ in many components, without a causal ablation. The upstream V47 already blends pretrained/multimodel and family specialists; the archived 18-class OOF shows Gemini sibling errors vs GLM provider errors. Separate source analyses show surface/style-only features underperform lexical features. The fixed sibling-output-length direction failed across external sources (SWEeper vs Zakir). **No part of this experiment may fit a prompt-length rule, provider gate, or ensemble weight.** Prior DCT-v3 uses surrogate **hidden states**; Short-TRACE uses **causal next-token rank transitions**, a distinct signal.

## One fixed evaluator and three arms

Evaluator: **Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4** (Kaggle model `qwen-lm/qwen-3/transformers/8b-base/1`), complete model frozen in inference/eval mode. 4-bit NF4 quantization, fp16 compute. No instruction-tuned model, fine-tuning, generation or target-model API calls. Input is the original raw response as a stand-alone sequence, tokenized with this frozen evaluator and maximum 384 tokens; skip the first token because it has no context. For every remaining observed token, compute its **zero-based rank** under Qwen3-8B's causal next-token vocabulary logits, `sum(logit > target_logit)`; report counts of ties and token truncation. This is an evaluator-model behavior, NOT the original author's true decoder trajectory.

Fixed rank bins: `[0], [1], [2-3], [4-7], [8-15], [16-31], [32-63], [64-127], [128+]` (9 bins). Feature histograms and adjacency matrices are normalized separately to sum to one (unless empty). Use a single deterministic text-hash-seeded shuffle of rank **bin labels within each passage** for the negative control; it exactly preserves the histogram and sample length while destroying true token adjacency. Random shuffle uses seed `20261107`, not true label/source/prompt. All three arms consume the same precomputed 8B rank sequence.

- **H (baseline):** histogram (9 dimensions).
- **S (shuffled control):** histogram concatenated with one shuffled 9x9 transition matrix (90 dims).
- **T (treatment):** histogram concatenated with true ordered 9x9 transition matrix (90 dims).

Single fixed readout in every arm: `StandardScaler` fitted **only on training rows**, then `LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=20261107)`. No grid search, per-class threshold, post-hoc weights or ensemble. The H/S/T comparison is not identical dimension but T vs S is matched width, classifier and histogram content. Report H vs T and S vs T separately.

## Corpus and locked sampling

Public source: Kaggle `nishantdahal/ntl-handwriting-blend` file `pool_v4.parquet`, SHA-256 `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`. Apply exact existing Phase 14 conservative subset and the **already frozen** DCT source-disjoint split, which holds out `alperen`, `hailey`, `willrice`. These are **previously selected, not new fresh holdouts**, so transfer evidence is exploratory, not an independent promotion set. Prevent train/holdout overlap in both `prompt_id` and `prompt_key`.

Small pilot budget: restrict to the frozen eligible-label set (17 labels); sample **20 train and 12 heldout distinct prompt IDs per label**, taking rows sorted by SHA256(`"short-trace-8b-v1" + label + prompt_id + source + text`). Sampling happens before loading/using evaluator predictions; stop if any label has fewer unique prompt IDs. The pilot totals **340 train + 204 heldout rows**; all feature columns and labels remain external to official competition data. Do not use official train/test, hidden IDs or Public LB for this pilot.

## Primary mechanism test and stop rule

- Report exactly the heldout **17-label macro balanced accuracy** for H, S, T, per-source macro across available labels, provider/sibling error and rows rescued/regressed versus H and S.
- Paired 1,000-resample **prompt-group** bootstrap of `BA(T)-BA(S)` on the same heldout rows, seed `20261107`, with labels absent in a resample omitted from that resample's macro BA.
- **Advance to larger 8B validation only if** T exceeds both H and S by >=0.03 absolute heldout BA **and** bootstrap lower 95% for `T-S` >0, with no source losing >0.05 absolute BA to S. This gate is intentionally demanding; failed/uncertain = do not upgrade 14B or submit.
- With only 12 heldout rows per label, **a PASS licenses larger validation, not 14B or Kaggle submission**. Before 14B, full-scale 8B analysis needs precommitted additional transfer set and official prompt-grouped OOF, preserving fold isolation. Only after 8B passes an independent transfer check can 14B upgrade be considered under a new frozen contract.

## Execution and data security

GPU run uses the existing Kaggle private, internet-disabled script-kernel path with public source dataset, offline Qwen3-8B model and pinned offline runtime wheels. No official competition data or new external training corpus publication. Only aggregate metrics and code may be published. Preserve original dataset hashes. If quota, resource, model availability, or strict data check fails, mark **BLOCKED**, not a scientific FAIL. A smoke or CPU unit test is engineering evidence only; do not claim it validates 8B fingerprints.

**No new competition submission is authorized by this pilot.**
