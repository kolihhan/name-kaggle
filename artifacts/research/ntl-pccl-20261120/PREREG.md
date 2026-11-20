# PC-CL v0 — preregistration BEFORE the first real QLoRA result, 2026-11-20

**Status:** Training path enabled by user's explicit 2026-11-20 answers (Q1–Q12) to pursue original research *and* Kaggle, allow training, matched controls, data quality, early-stop pilot and 2-experiment reflection. Prior 2026-11-13 global `NO NEW TRAINING` assumption is superseded **for this bounded PC-CL pilot only**. This does not authorize arbitrary Kaggle submission, unbounded GPU or data redistribution.

## Motivating evidence / competing hypotheses

- Real incumbent: our authentic Kaggle Public **0.702** (`#58514818`, attributed public source), vs 2026-11-20 top-10 Public cutoff **0.761**. Own trained LoRA24 A1 had **24 rescues, 66 regressions** vs older 0.679 on 347 previously examined external rows: no verified 0.702 complementarity.
- Historical **source-adversarial ModernBERT trained on a different 18-label source-disjoint cohort** was **0.214444 CE vs 0.054444 GRL** on 2026-11-17 [real GHA #44116912517](https://github.com/kolihhan/name-kaggle/actions/runs/44116912517): +source-invariance by gradient reversal was strongly detrimental, **not a reason to repeat GRL**.
- Historical [prompt-column train-time CPU test #44277551557](https://github.com/kolihhan/name-kaggle/actions/runs/44277551557): *matched* 0.282598, row CE 0.278153, shuffled 0.290375 on repeatedly reused official grouped folds; matched-row +0.00444 CI includes zero, matched-shuffled **−0.00778**. Not a result supporting correct prompt matching; this is a **strong negative background prior**.
- H1 (generator-stable style hypothesis): QLoRA with prompt-controlled **positive pairs from same generator on different prompt** and **hard negatives from different generator on the same prompt** learns a generator-specific latent signal that transfers to unseen sources/prompts better than matched-data CE.
- H0 (representation/source artifact or generic regularization): correct prompt pair contrastive harms, is neutral, or shifts apparent signal but no source transfer; benefits require shuffled contrastive control to rule out generic regularization. Old CPU matched-shuffle negative raises the H0 prior.
- Alternative explanation: different source/model route/revision artifacts make label classification less meaningful; public pool source_protocol and served ID metadata are **publisher statements** rather than provider-authenticated labels. This pilot cannot claim absolute 0.702 uplift.

## Frozen equal-budget two-arm pilot

**Data prep code:** [frozen current source/prompt and rectangle sampler](../../scripts/ntl_pccl_18source_prepare_20261120.py). Live Kaggle pool `crtproduction/name-that-llm-open-pool` downloaded with authenticated Kaggle token, **SHA-256 `7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492`** (2026-11-20). Source-level `served_model_check=yes`, publisher-license metadata whitelist, 40–250 words, previous risk exclusions. Holdout source names selected with **exactly the previous frozen `ntl-heldout-20261117` SHA-mod5 rule**, not selected after seeing model outcomes. Across train vs valid: publisher sources, normalized prompt contents/IDs, normalized responses disjoint. **This is a source-disjoint prospective training run, not an independent benchmark provenance guarantee.**

**Actual pretraining feasibility #44681721339 (SUCCESS) before any GPU:**
- Train **8,347 rows / 8 sources / all 18 classes**, validation **1,800 / 3 sources / exactly 100 per class**. All text only ephemeral Actions/owner-private Colab, never committed/exported.
- **32** candidate model-pair label combinations in two-prompt/two-generator rectangles, **13 of 18 labels** eligible for contrastive positives/hard negatives. Five labels only enter CE fallback; **0.702 overall transfer cannot be inferred from any subset-only contrastive gain.**
- Each arm sees **identical 160 four-response batches**, 120 source/prompt-paired rectangles + 40 class-balanced ordinary batches. Among 120 rectangles only **12 (10%)** pair across *different* training sources, so the mechanism explicitly targets **prompt invariance within training sources**, not a proven source-invariant representation. Holdout sources differ.
- Each rectangle: `[classA@prompt1,classB@prompt1,classA@prompt2,classB@prompt2]`. CE uses the same rows in both arms, all 18 output labels. C1 additionally uses one InfoNCE loss with positives offsets 2/3/0/1 (same generator/different prompt), and all other responses including *same-prompt different-model negatives*.
- Model: **Qwen/Qwen3-1.7B-Base**, existing Apache-2.0 pinned revision `658b6ee46470b73f3e0e04e57fc9fb7031fd7daa`. Using **1.7B instead of 8B in stage 0** is a bounded mechanism pilot; failure would **not disprove** the same mechanism at 8B but would prevent blind 8B budget spending. 4-bit NF4 QLoRA r8 alpha16, all attention and MLP projection modules, 18-class trainable score head, tokenizer max 192 tokens. Same identical initialization `SEED=20261120`, same optimizer steps/data order, same LR `2e-4`, AdamW WD `0.01`, max norm `1`, batch size `4`.
- **C0** = 18-way standard CE. **C1** = same CE + `0.2 * InfoNCE` on rectangle batches only, cosine temperature `0.10`, both use the same full-base 18-class head and identical training budget. Neither uses prompt-ID or source-ID during inference.
- No wall-time equality claim (C1 adds a cheap pair matrix), only same model forward tokens, optimizer steps and batches.
- No retraining on official train, no direct submission and no comparison to fabricated 0.702 OOF.

## Frozen readout / decisions

Evaluate both models on *identical* 1,800 source-disjoint examples. Primary paired **18-class macro BA delta**, secondary all 18 per-class recalls, paired rescues/regressions, cross-source support, a **source-stratified prompt-cluster bootstrap CI with 1,500 replicates** (conditional on the heldout data; **NOT** a CI over unseen source distribution, only 3 valid source units). The source-quality, model-version and prompt-bank uncertainties are expressly unresolved.

- **Advance to independent source repetition + shuffled-negative control**, not directly to 8B or Kaggle: C1−C0 **≥+0.04 BA** and CI lower **>0** on these heldout responses, with no catastrophic class regressions. If this passes, perform a second research round: shuffle prompt correspondence while preserving label/group marginal structure and compare against C1 to establish *prompt-specific* benefit; measure on new source/protocols, then only decide on a full 8B budget and 0.702 complementary evidence.
- **Reject v0 on this cohort** if Δ≤0 (not all potential contrastive methods); no hyperparameter sweep of temperature/weight on this holdout.
- Between 0 and +0.04 or CI includes zero: inconclusive; no official submission.
- A pilot above 0.04 is **not evidence of outperforming 0.702** or a reason to replace Kaggle champion. One source-disjoint sample is not enough; existing public leaderboard is not a tuning loop.
- Capture only aggregate JSON and optionally a valid checkpoint *after independent evidence*, not raw text, row logits or personal identifiers in public repo. No local Kaggle GPU, no Kaggle submissions, no paid provider API calls. Bounded one owner's Colab T4 and cleanup even if failed.

## Actual training scripts and reproducibility

- [Data manifest and rectangle construction](../../scripts/ntl_pccl_18source_prepare_20261120.py)
- [Two real QLoRA arms and paired source-disjoint validation](../../scripts/ntl_pccl_17b_real_qlora_colab_20261120.py)
- Preflight [Actions #44681721339](https://github.com/kolihhan/name-kaggle/actions/runs/44681721339) SUCCESS.
- A successful GHA workflow is **infrastructure**, not evidence of a model gain; only an actual both-arms `NTL_PCCL_REAL_T4_FINAL_AGGREGATE` gives a scientific pilot result.
