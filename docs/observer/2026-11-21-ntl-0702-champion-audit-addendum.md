# NTL — Frozen 0.702 Champion Audit: Evidence Inventory Addendum

**Date:** 2026-11-21 (Asia/Kuala_Lumpur). **Project:** `kolihhan/name-kaggle`. **Branch:** `research/attribution-bottleneck-review-20261106`. **Type:** Observer documentation-only addendum to [2026-11-21 independent review](2026-11-21-independent-review.md), incorporating the user-provided NTL audit and rechecking its primary evidence where accessible.

## VERDICT

- **Performance claim: HOLD.** No verified new 18-class BA gain or scored candidate in this review.
- **Strong-champion stage attribution: BLOCKED.** Existing visible output cannot support independent per-class rescues/regressions for the exact scored 0.702 champion.
- **Research priority: PIVOT.** Focus on recovering genuinely independent full-18 labels + stage-aligned champion predictions before another expensive sibling model, GLM/Apex tweak or QLoRA submission.
- **Scope:** No code change, model run, Kaggle submission, GPU execution, PR, or new experiment; this file is a historical review, not a monitoring process.

## VERIFIED — original evidence checked

1. **Official incumbent historical receipt:** Kaggle ref `58514818`, `SubmissionStatus.COMPLETE`, Public Balanced Accuracy **0.702**, Private score blank, was authenticated through [Action 44674191096](https://github.com/kolihhan/name-kaggle/actions/runs/44674191096). Notebook `kolihhan/ntl-0702-apex-pinned-nov18`; source SHA-256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc` from the [original source comparator](../research/ntl-0702-vs-0704-true-source-diff-20261120/RESULT.md). **Live account status was not freshly queried in this documentation follow-up.** Do not call 0.702 a newly live-confirmed result.
2. **Frozen exact Champion missing stage outputs:** authenticated notebook output listing in [Action 44675693564](https://github.com/kolihhan/name-kaggle/actions/runs/44675693564) shows a saved preview `submission.parquet`, Python code and smoke files, but not a row-aligned full 18-class `final-logits.npz`, `post-glm.npz`, pre/post Apex probability matrix or hidden-test truth. Also inspected [earlier output-access Action 44487147476](https://github.com/kolihhan/name-kaggle/actions/runs/44487147476): other historical notebooks had incomplete/inaccessible output evidence and are not substitutes for the frozen 0.702 champion.
3. **Original upstream metadata:** [Action 44765273054](https://github.com/kolihhan/name-kaggle/actions/runs/44765273054) checked **4,758/4,758** publisher-recorded `requested_model` = `returned_model`, from **13 labels**. [Original row-route audit](../research/ntl-original-13generator-row-route-verify-20261120/RESULT.md). These are publisher fields, **not authenticated actual provider execution** or 18-class inference evidence.
4. **Strong source comparison:** [Action 44675311464](https://github.com/kolihhan/name-kaggle/actions/runs/44675311464) found 27/29 equal executable source parts between the champion and a public notebook titled 0.704. GLM routing and Apex changed together, so neither the notebook title nor a hypothetical 0.002 difference isolates a causal gain. [Source comparison](../research/ntl-0702-vs-0704-true-source-diff-20261120/RESULT.md).
5. **External pool coverage and rights risks:** the proposed 8,347-row training subset had **4,270** examples from one upstream provider (~51.16%), while publisher APIs/settings/rights and official generation-host similarity are not validated as full-18 training truth. [Source audit](../research/ntl-8k-original-source-audit-20261120/RESULT.md) and [provider risk review](../research/ntl-provider-output-use-terms-20261120/RESULT.md). These are provenance and policy-review findings, **not a legal opinion**.

## RECOVERY MATRIX — distinguish sourced facts from reported inventory

| Historical data or artifact | Labels / sample size | Exact scored 0.702 stage output | Usable as newly independent stage evaluation? | Evidence status |
| --- | --- | --- | --- | --- |
| Kaggle official TRAIN | 2,699 rows / 150 prompts / 18 classes | No validated champion out-of-fold components | **No**: original TRAIN involved in champion pipeline fitting; need provable full reader+calibrator OOF exclusion | **VERIFIED data count** in repo; **REPORTED** inference-time `Residual.fit(train.text, train.model)` call; requires exact source re-inspection before code-line claim |
| Oleg216 | 216 labeled examples (user report) | Only historical older-model predictions claimed | **No**, repeatedly inspected; no aligned champion stage outputs | **REPORTED** panel inventory; partial historic notebook output evidence in Action 44487147476 |
| External 347-row diagnostics | 347 rows; reportedly only 12 true classes | Older 0.679 prediction claimed | **No** full 18-class champion counterfactual | **REPORTED** specific panel detail, not independently verified this follow-up |
| GLM 252-row panel | Reportedly 18 classes | Older 0.683 prediction claimed | **No**, reused and original route provenance incomplete | **REPORTED** panel detail |
| PC-CL validation | 1,800 rows / 18 classes | No frozen 0.702 stage outputs | **No**, source-disjoint for a *different* classifier; label source and generation protocols do not establish champion transfer | **VERIFIED** training/validation design in [PC-CL result](../research/ntl-pccl-20261120/RESULT.md) |
| Open Pool | 231,036 total records | No saved champion stage outputs | **No** proven fresh post-training panel; only 33,603 parseable generation dates, none >= 2026-11-16 | **VERIFIED historical** metadata feasibility in [source/holdout audit](../research/ntl-0702-vs-0704-true-source-diff-20261120/RESULT.md) |
| Scored 0.702 Notebook preview | Saved `submission.parquet` | No hidden labels or full stage logits | **No**, preview is distinct from the private hidden evaluator | **VERIFIED** output file manifest |

An older model prediction array is not evidence of component uplift beyond the **0.702** champion. Rows marked REPORTED are not independently promoted by this addendum.

## Leakage, attribution and rights

- **TRAIN-as-OOF:** A fresh inference pass over original labeled training examples is **not** out-of-fold for the scored frozen ensemble. A fold label added after inference cannot undo residual/model selection exposure. This is a risk to scientific validity, **not proof of Kaggle rule violation**.
- **Provider identity vs provenance:** Publisher's exact requested/returned labels can be internally consistent while actual model serving, route, version, decoding settings and official contest-domain equivalence remain UNKNOWN. Published Gemini and Kimi style differences are *producer reports*, not our measured causal explanation.
- **Provider rights:** An uploader's CC license does not automatically resolve every upstream generated-output use restriction. The dated repo policy review identifies Z.ai training restrictions, Anthropic/Vertex scope questions, Google Cloud and OpenRouter upstream obligations, and unresolved Kimi terms. Verify current original-provider terms at the time of any proposed training; do not present this audit as legal clearance.
- **Causal attribution:** Even if fresh champion outputs appear, a before/after GLM or Apex contrast estimates the effect of these stages on that panel, not why the whole pipeline fails under hidden evaluation without a transferable control.

## SINGLE LARGEST MISSING INFORMATION

**True, untouched, permitted 18-class panel with frozen-champion ID-aligned stage predictions and verified complete model lineage.**

This is a **missing evaluation capability** rather than a demonstrated new generative information signal. Potential GPT/Gemini sibling likelihood information remains a hypothesis; do **not** claim it is already complementary to 0.702.

### Exact unblock package

1. Newly collected (not just newly published) 18-class labeled panel not used in the champion's training, tuning, calibration, residual fitting or adaptive experiments; provide example ID, original label, canonical prompt group, source, time, requested ID, independently supported served ID, full generation/decoding settings, relevant usage permission.
2. Freeze and record scored source SHA, reader/checkpoint assets, training folds, class ordering, source training exposure, all calibrators and `Residual.fit` lineage. Determine whether any prospective evaluation row was seen by **any** supervised component; mark `UNKNOWN` instead of assuming independence.
3. Exact same-row 18-way `pre-GLM` and `post-GLM` logits, `pre-Apex` and `post-Apex` predictions/probabilities, final prediction, and original labels, with stable hash/index and row-count consistency. Keep private samples/output in authorized secure evaluation storage, not a public GitHub repo.
4. Preregister BA18, per-class recall, paired rescues/regressions, prompt/source cluster uncertainty, resource use, and explicit go/no-go decisions before inference. Evaluate domain-transfer risk, repeated validation adaptation and inference time feasibility.

**Falsifier / stop:** If clean sample provenance, rights, component training lineage or complete per-row stage output cannot be established, verdict remains **BLOCKED**. If a future valid paired panel shows no robust independent component opportunity, stop new GLM/Apex/sibling tweaks and reconsider representation/data/global architecture rather than repeating local rules.

## Observer handoff

- **PROJECT:** NTL only.
- **LATEST HISTORICALLY VERIFIED OFFICIAL BASELINE:** Public BA 0.702, ref 58514818; **LIVE UNKNOWN** until a newly authenticated ledger checks post-2026-11-20 changes.
- **ROOT CAUSE:** Not identified. Source/generator identity confounding and redundant components are hypotheses; independent attribution gap is demonstrated.
- **NEXT TASK FOR EXECUTOR:** Read-only acquisition feasibility and source-lineage/rights audit for a genuinely independent full-18 panel, with precise block reasons; **do not** rerun 0.702 GPU inference until the panel and metrics are qualified.
- **CONFIDENCE:** HIGH on stored-output limitation and prior Kaggle receipt; MEDIUM on actual benchmark independence and source/rights inference; LOW on any claimed competitive mechanism gain.
- **EXECUTION:** Observer documentation only; no training, GPU, Kaggle submission, PR or model code changes.
