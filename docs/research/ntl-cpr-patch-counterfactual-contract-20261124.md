# CPR-v0 — Counterfactual Patch Retrieval, original falsifier (2026-11-24)

## Research claim and prior art
User's goal is competitive original authorship attribution, not replication. Best verified account Public BA 0.718 is **Evgen public-code reproduction**, not a new method. Existing original A0/A1 ATLAS, PACE and SCaL did not meet quality gates. Old prompt-residual LDA experiment (2026-11-01) degraded 11/13 heldout sources: **do not repeat it**.

Prior art: [HALvest-Contrastive](https://aclanthology.org/2026.emnlp-main.2660/) studies token/patch late interaction for authorship; we do NOT claim to invent patch retrieval. This experiment differs in **prompt-matched counterfactual distinctiveness of training patches**: explicitly downweight patches that look like other generators responding to the SAME prompt, then retrieve from this fixed fingerprint store for single-text inference. This is an original hypothesis, not proven new to the literature.

## Falsifiable hypotheses
- H1: within-prompt across-generator patch overlap identifies topical/boilerplate evidence; suppressing it increases source-specific retrieval across different prompts/sources.
- H0: patch overlap is not a stable nuisance, or downweighting destroys useful identity; retrieval is intrinsically too weak versus a strong text classifier.
- Counterfactual placebo: shuffle the within-prompt distinctiveness weights across training patches, preserving weight distribution but eliminating the real patch-weight association.

## Fixed same-data three arms plus strong control
- Train data: only official Kaggle TRAIN 2699 exact rows, SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`; frozen five-fold mapping SHA `94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d`.
- One prespecified prompt-disjoint Fold 0: 2159 official train rows, 540 dev. No trained encoder from past competition. All 18 labels, prediction can be any of 18.
- External **C1 360/12** labels and **C2 470/12** labels from same publisher beroli, unique prompt sets by prior audit; never train on either. Claimed served metadata is publisher self-report, **not independently verified** and **not full-18 validation**.
- Representation: contiguous response patches, 40 whitespace words per patch, maximum 4 patches per response, deterministic. TF-IDF character 2–5 grams with min_df=2, max_features=120000, trained ONLY on fold-training patches. L2 normalization. No QLoRA, GPU, external training data or API generation.
- Standard late-interaction: average over query patches of each patch's best cosine with any training patch belonging to a given model. A: unweighted. B: multiply candidate training patch match by `w_j=(1 - .9 max_{k:same_prompt,different_model} cos(j,k))^2` (if no competitor, w=1). C: same weights but permuted inside prompt using seed 20261124. One scalar fixed, **no tuning**.
- D: strong published 4-view sparse LinearSVC `build_fold_features/make_svc` on same 2159 training docs, tested on official Fold 0 and both C1/C2. This **same-row control** is the floor for competitive plausibility. All 18 predicted labels allowed on both 12-label external panels.
- Validate official and external schema, class support, folds, duplicate IDs, prompt overlaps and exact answer text overlap; do not pretend these checks prove provider identity. Exact response text duplication across official/external would invalidate transfer evidence.
- Metrics: BA18 official Fold0; BA12 external C1, BA12 external C2 (all 18 predicted labels, unsupported guess is an error); same-row rescued/regressed vs A/D, 2000 prompt-cluster bootstrap draws pooled C1+C2, wall-clock CPU. No source data or per-row predictions logged or committed.

## Predetermined decision
Continue toward learned strong-encoder patch representations ONLY if B beats A by >=0.02 BA on **both** C1 and C2, prompt-group pooled CI lower>0, B >= D on external average and B improves Fold0 vs A (do not chase public score). Else stop this exact construction, no new patch-length/rank/weight searches on spent panels. Even passing is only a cheap evidence screen: real stronger representation, source/protocol separation and 18-label transfer required before official competition submission.

Cost: GitHub Actions CPU <=22m, no GPU/official submit, no scheduled automation. Models/datasets downloaded only to ephemeral Actions runner, deleted after. Keep source data off public GitHub.
