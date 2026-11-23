# NTL GCTH-v1 — Generator-Conditioned Token Head, pre-registered experiment

2026-11-23, branch `research/ntl-generator-token-head-v1-20261123`. Frozen BEFORE outcome/OOF scoring. **One CPU-only run, TRAIN-only.**

## New scientific question

FCS-v1 demonstrated a real pretrained LM context surprise profile beats a BOS-only unigram on historical 18-way Prompt-disjoint OOF (+.02038), but its global 55 summary features got only .19520 BA versus direct Sparse .36709. This experiment studies **generator-specific conditional next-token choices** rather than applying an extra classifier or ensemble to the incumbent.

H1: a frozen pretrained causal LM context embedding plus a model-specific next-token categorical distribution improves exact-model attribution beyond model-specific *unigram token frequency* and a context-destroyed equal-capacity negative control. H0: any improvement is merely vocabulary/source artifacts or overfitting to 120 train responses/model/fold. This is NOT the previous unexecuted `conditional-denoising18-v1` RoBERTa output-adapter experiment, which remains frozen. Nor is it genuine proprietary target model likelihood.

## Immutable data and frozen model

Official `data/train.csv` exactly 2699 rows/150 Prompt groups/18 sources and `artifacts/phase1/folds.csv`, cryptographically checked with `scripts.blind_prompt_contrast.load_snapshot`. These are **historical spent DEV folds, not fresh external transfer**. No Kaggle test, no target generator APIs, no externally generated texts, no external terms/rights issues.

Frozen pretrained `distilbert/distilgpt2@f241065e938b44ac52db2c5de82c8bd2fafc76d0` causal GPT2Model, FP32, CPU-only 2 threads. First 64 BPE tokens per response; prepended EOS/BOS. Hidden state at BOS / prior token predicts the **next** real token; no future-token leakage. Project frozen 768-D h into fixed 64-D random orthonormal projection (seed 20261123), with **train-fold-only** standardization and clip [-4,4]. Print truncation count.

Fold-local vocabulary: top 127 BPE token IDs by TRAIN-only frequency (tie by token ID), plus OTHER. Targets mapped into 128 categorical outcomes. Source m empirical Bayes prior over categories: `p0=(global_counts+0.5)/(total+.5*128)`; `p_m=(class_counts+64*p0)/(class_total+64)`; equal model priors. No heldout label/source/Prompt ID used at inference.

**A (unigram):** proper averaged `log p_m(target_category)`.
**B (true context):** per source train `logits_m(v|h)=log p_m(v)+h@W_m[:,v]` with normalized logsoftmax over all 128 categories. Each W starts zero, fixed Adam lr 0.01, exactly 3 epochs, batch 1024, L2=0.02*mean(W²), shuffle seed 20261123+fold+source. No pretrained backbone training. Score each *single* response under all 18 heads by mean category token log probability; argmax.
**C (context-destroyed negative control):** exactly B's train and optimizer but context h randomly permuted within the **same true model** across response IDs while keeping all token labels/marginals unchanged. Verify >=95% donor contexts from distinct response IDs. Score true untouched contexts at inference; shuffled-training control is OOD, so **B>A also required**.
**D (independent stand-alone strong baseline):** existing frozen 4-view Strong Sparse SVC with identical 5 training/heldout folds, no blending.

## Frozen outputs/gates

All 2699 historical OOF rows, all 18 BA, 5 fold BAs, 18 class recalls, paired rescues/regressions for B against A/C/D, paired by-Prompt 1000-bootstrap 95% CI, wall time and costs. Do not log original passages, token strings/IDs, per-row predictions or trained parameters.

**Mechanism PASS only if ALL:** B−A ≥+.02, B−C ≥+.02, both Prompt-bootstrap CI lower bounds >0, B beats A in ≥3/5 folds and C in ≥3/5 folds, and no class recall decrease >.15 vs A. Competitive signal separately requires B≥D ≈.36709 on same rows. A small historical mechanism PASS is NOT hidden performance evidence or submission permission.

**If FAIL:** no tuning token list, context cap, weights, seeds, regularizer or folds. Retire the exact frozen experiment. Real independent 18-class source/protocol-disjoint benchmark remains **BLOCKED**. Original 0.715 Public Champion is comparison context ONLY, not used in training or modified. No Kaggle submission, no GPU, no external training output; CPU GitHub Actions limit 38min.
