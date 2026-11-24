# PACE — Prompt-robust Attribution with Conditional Evidence (development v0)
2026-11-24. ATLAS-inspired own project hypothesis, NOT a faithful ATLAS reproduction, NOT an established novel algorithm or GOLD result.

## Mechanism and competing explanations
ATLAS A0/A1 on 300 historical GLM sibling rows: raw token aggregation did not reliably improve Global53 (283 -> 284 correct; CI includes zero). Learned ATLAS source prefixes/LoRA were not executed. GCTH's weak-reference-LM contextual output head lost to its unigram control; another token-identity classifier, temperature sweep, or GLM binary patch is not justified.

Use the two genuine aligned native GLM log-probability streams available for **all 2699 official TRAIN responses, 18 exact labels**. Let d=logp_full-logp_flash, s=-(logp_full+logp_flash)/2. s approximates shared token difficulty, not an observed ground-truth prompt or invariant nuisance. Hypothesis H1: source identity affects the conditional law P(d|s,y), beyond unconditional likelihood differences. H0: the marginal score or joint difficulty distribution accounts for any apparent discrimination; conditionalization adds no useful identity. Unlike previous ATLAS readout, learn one source-conditioned distribution for **each of the 18 classes**. Unlike GCTH, model observed native likelihood evidence, not reference-model token categories or contextual hidden states. No pretrained weights updated, no new forwards or paid inference.

## Fixed controls and scoring
Five permanent prompt-group held-out folds, original official SHA, original eight cache-file SHA checks and exact token alignments. Use all rows, no label subset selection. Within each training fold, compute pooled token quantiles for eight bins of d and eight of s (ties removed). Estimate all class-conditional distributions using only the training fold. Equal class priors. Each document contributes unit total mass to estimation (divide token counts by passage length); smooth via unit Dirichlet prior (one pseudo-document spread across cells). Scores are average per-token log probability, argmax across classes:
- A marginal P(d|y).
- B P(d|s,y), smoothed conditional from same joint histogram as C.
- C joint P(d,s|y): same learned histogram but retains difficulty distribution, a test for nuisance removal.
- D independently fitted 53 global/paired-statistic multinomial LogisticRegression C=1, same inputs and prompt splits. This is a new 18-way control, NOT the binary 0.9433, NOT 0.715. No ensemble.
- Report 18-way BA and per-class recall, fold BA, paired rescues/regressions, prompt-cluster bootstrap CI, wall-clock and actual cache count. No 0.715 comparison without matched outputs.
- Prespecified development continuation: B exceeds A and C by >=0.02 BA with lower paired prompt-bootstrap bound >0 for both, and B exceeds D by >=0.01 BA. Otherwise STOP this exact design, no bin/smoothing/threshold retuning on these spent rows. D is a same-input control, not necessarily competitive; compare historically 0.3671 Sparse only as context, not a new paired evaluation.
- If B passes, still require independent source transfer and competitive same-row strong scorer evidence before official competition submission. No false promise of 0.72+. No new training of a foundation model, no GPU, no submission in this experiment, no raw data/logit dumps. Learned histograms are inexpensive fold-local statistical estimators.
- Existing official 58808007 (GLM Rescue-per-Token) is unrelated; do not submit another version based on this diagnostic. 
## Provenance
ATLAS inspiration: https://github.com/komseok0109/ATLAS; prior experiment https://github.com/kolihhan/name-kaggle/actions/runs/45174502026. Code/contract in independent branch, original scorer unchanged.
