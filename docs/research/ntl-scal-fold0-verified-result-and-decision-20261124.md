# SCaL Fold 0 — verified negative result and research decision
Date: 2026-11-24, Malaysia time.
## Authentic evidence and scope
- Launch [Actions #45329725935](https://github.com/kolihhan/name-kaggle/actions/runs/45329725935): successful GitHub workflow means the one-shot private GPU notebook was pushed and worker queued; **it does not measure predictive quality**.
- Real Kaggle private worker status **COMPLETE**, verified from authenticated read-only [Actions #45348759150](https://github.com/kolihhan/name-kaggle/actions/runs/45348759150). The Kaggle output contained `ntl_scal_aggregate.json` with actual metrics. Original job logs read returned RC2, so the verified evidence is the aggregate output, not an inspected complete train trace.
- Evaluation: 2,159 train + 540 holdout rows, frozen official prompt Fold0 only, 18 exact labels. Both source-likelihood arms use the same pretrained RoBERTa masked states and source heads, differ in added sibling contrastive loss; source contrastive loss is an *experimental* idea and not a proved original research contribution.
- `proper_source_nll` BA = **0.07962962962962966**.
- `SCaL_sibling_contrastive` BA = **0.08333333333333334**.
- `direct_frozen_doc_CE` BA = **0.2074074074074074**.
- Contrastive minus source-NLL = **+0.0037037037037036813 BA**, 9 rescues, 7 regressions, 141 changed predictions; 95% prompt-group bootstrap interval **[-0.009259259259259273, +0.01851851851851853]**. Pilot success condition from frozen contract **FALSE**. No official competition submission.
- The other real 18-way likelihood diagnostic PACE had conditional BA 0.1574844643 vs same-source Global53 0.2182401193; different hypothesis, train protocol and source evidence. Its numbers must NOT be directly combined with SCaL Fold0 to form one independent comparison.

## Critical review — what is and is not established
1. **Fail, not a breakthrough.** Even the much simpler direct classifier over the same frozen RoBERTa masked-state representation beats source-likelihood by ~12.4 percentage points. This is a strong argument against this particular objective and training recipe in this cohort, not against every source-conditioned likelihood model.
2. **Not faithful ATLAS.** This is bidirectional, jointly masked 15% word evidence, not source-conditioned causal next-token modeling with learned prefix and LoRA as in [ATLAS](https://arxiv.org/abs/2610.19583). It is not a claim to reproduce ATLAS family-level MAABench, much less its 74.27% shifted-setting accuracy or 18-class exact NTL hidden BA.
3. **Own bug/invalid contrast label:** `sibling_indices` maps any singleton family (Kimi, Qwen) to the next alphabetically listed source, not a real within-family sibling. For multi-member families, it always chooses the first alternative label, creating asymmetric fixed-negative exposure. Thus “sibling-only contrastive” is false for some classes. This issue prevents attributing the loss failure specifically to the *concept* of within-family negative sampling.
4. **Insufficient external transfer:** only the repeatedly studied official TRAIN Fold0, 540 heldout rows; same source/collection and prompt-group holdout, not independently collected modern 18-class source data. This cannot justify a hidden-score assertion.
5. **Data/capacity:** ~2,159 train rows for 18 sources, only two training epochs, 50,265-way frozen MLM decoder with per-source biases and low rank residuals, 256-token truncated bidirectional contexts, single random mask per document. The poor result could reflect inadequate data/optimization, incomplete corruption context, non-generative conditional likelihood objective, source shift, or true lack of signal. No causal decomposition isolating these causes was performed.
6. **Preexisting research:** prior frozen `artifacts/candidates/conditional-denoising18-v1/CONTRACT.md` already specified source-specific RoBERTa masked-token output residuals. SCaL adds a contrastive term but is not by itself a publishable newly invented architecture. Earlier GCTH and FCS negative experiments likewise show this task is hard for generic frozen LM readouts.
7. **No verified hidden runtime feasibility.** The small Fold0 model training and evaluation do not verify full official ~4k hidden rerun or Kaggle 9h timeout. No official score was produced by SCaL.

## Honest account scoreboard as of authenticated receipt #45348759150
- Best completed **Public BA 0.718**, **submission 58755304**, description “Owner-requested public Evgen forensic EDA inference reproduction 2026-11-23”. This is *reproduced external code*, not original score-winning research.
- Earlier 0.715 **58680531**, externally reproduced Estopadilla v5.
- Rescue-per-token **58808007** and exact Sanpier V3 **58811345** remained PENDING and had blank publicScore at the snapshot timestamp (2026-11-24 15:32 MYT). Future completion, score and hidden runtime are UNKNOWN; do not assume the score increased or failed.
- All other older public/OOF numbers are different experimental populations or models and cannot be directly compared numerically with the official 0.718.

## Decision
**STOP the frozen SCaL v0 recipe.** No post hoc margin/negative selection patch on spent fold0, no five-fold extension, no GPU repush, no competition submission. Do not interpret failure as disproving original causal ATLAS. If exploring next, start new mechanism contract using legal source/protocol-diverse 18-label evidence and a separately scoped trainable representation that can beat strong exact-label same-row controls; reserve independent source-transfer examples before fitting and assess deployment cost.
