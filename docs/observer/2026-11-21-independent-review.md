# NTL Independent Observer — Baseline and Champion Audit

**Recorded:** 2026-11-21 (MYT). **Evidence reviewed:** authenticated GitHub Actions and repository research records through 2026-11-20. **Scope:** read-only independent review, plus this user-authorized documentation-only handoff. **Current decision:** **PIVOT in research prioritization / HOLD on new performance claims / BLOCKED on a defensible champion component-attribution benchmark.** These are different decisions, not conflicting score results.

## Verified evidence (not new experiments)

1. **Official incumbent:** Kaggle submission `58514818` = `SubmissionStatus.COMPLETE`, Public Balanced Accuracy **0.702**, Private score blank, as printed by authenticated Kaggle CLI in [Action 44674191096](https://github.com/kolihhan/name-kaggle/actions/runs/44674191096). Original scored notebook `kolihhan/ntl-0702-apex-pinned-nov18`; frozen executable source SHA-256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`, per [source comparison and proof inventory](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/artifacts/research/ntl-0702-vs-0704-true-source-diff-20261120/RESULT.md). This is the last reviewed receipt, not a guaranteed real-time Kaggle state.
2. **Strong vs public notebook source:** 27/29 executable code parts identical; changed GLM/Fable priority and Apex postprocessing are confounded. The public `0.704` title is not independently demonstrated as a single-intervention lift over our 0.702. [Action 44675311464](https://github.com/kolihhan/name-kaggle/actions/runs/44675311464).
3. **Missing genuine champion outputs:** Original notebook output manifest has preview `submission.parquet`, scripts and smoke binaries; no matched full-18 pre-GLM/post-GLM/pre-Apex/post-Apex logits or independent hidden labels. [Action 44675693564](https://github.com/kolihhan/name-kaggle/actions/runs/44675693564). Therefore exact strong-champion per-class confusion and component rescues/regressions are not presently identifiable from the saved outputs.
4. **Source internal consistency ≠ authenticated identity:** Upstream **4,758/4,758** publisher-recorded requested/returned model IDs agree, across **13** source labels. [Action 44765273054](https://github.com/kolihhan/name-kaggle/actions/runs/44765273054). No authenticated original provider receipts or guaranteed 18-class, competition-host-equivalent samples follow from this check.
5. **External training-data issue:** PC-CL frozen training set **8,347** rows across 8 sources, including **4,270** from one publisher (~51.16%); provider rights and route identities incomplete. [Action 44749886505](https://github.com/kolihhan/name-kaggle/actions/runs/44749886505). Original producer explicitly reports style/setting differences for Gemini and Kimi; treat these as producer observations, not our verified host transfer. [Original source audit](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/artifacts/research/ntl-4270-upstream-generator-domain-shift-20261120/RESULT.md).
6. **Negative local training mechanism:** Matched 160-step Qwen3-1.7B QLoRA CE **0.078889** vs CE+InfoNCE **0.071667** BA18 on **1,800** source-disjoint external examples. 62 rescues / 75 regressions; 95% paired cluster CI of difference [−0.018839, +0.004112]. [Action 44683814370](https://github.com/kolihhan/name-kaggle/actions/runs/44683814370). This rejects *that underexposed trial*, NOT contrastive learning as a class and NOT a comparison with 0.702 on same rows.

## Review actions completed

- Read `AGENTS.md`, research branch `NOW.md` and relevant source/results.
- Re-read authenticated official score and training workflow job logs.
- Inspected source-comparison, champion output-inventory, provenance and source-shift audit reports; checked Action evidence for the recent metadata and training claims.
- Distinguished actual scored results from CPU checks, historical OOF, publisher self-attestation, local diagnostics and hypotheses.
- Did **not** perform model inference, training, Kaggle submission, or modify candidate logic.

## BLOCKED: independent strong-champion attribution

The existing official TRAIN (2,699 / 150 prompts) is not a fresh champion OOF. The frozen Apex pipeline includes an inference-time residual fit against official TRAIN; training lineage of every other component is not proven disjoint either. Re-running this frozen champion on TRAIN and labeling it OOF without rigorous fold-exclusion would be invalid. Historic Oleg216, 347-row, GLM252 and PC-CL1800 diagnostic panels are **reported recovery candidates**, not proven fresh champion-aligned 18-way paired evaluations. At present no valid recovered full-18, model-identity-verified, untouched champion stage output exists.

**Minimum unblock requirements:**
1. An independent, untouched **18-class labeled** panel with identity/protocol provenance, prompt groups, source identifiers, timestamps/settings, and documented permitted evaluation rights.
2. Frozen champion executable SHA + provenance of all trained readers, calibrators, Apex/residual fits, and proof panel was not used to train/select/tune any component.
3. Exact row-ID-aligned truth + full 18-way logits/predictions at pre-GLM, post-GLM, pre-Apex, post-Apex checkpoints.
4. A preregistered BA18, per-class recall, paired rescues/regressions, source split, uncertainty and stop/go criterion.

## Problems and root-cause hypotheses

**VERIFIED limitation:** No reliable direct decomposition of the strong ensemble's errors; no independent evidence that a new component rescues enough 0.702 misses. Source/provider usage rights and model serving identity remain conditional.

**INFERENCE, not proven primary cause:** sibling-family identity confusion, provider-host/domain shift, and redundant fused readers might dominate some residual errors. Alternate explanations include calibration/prior correction, data coverage and inference budget effects. Do not declare any one the bottleneck before aligned evidence.

## Next best *proposed* experiment (NOT authorized/executed)

**Strong Champion Residual Attribution:** compare the exact frozen 0.702 pipeline against its own pre/post GLM and Apex stages on a newly qualified, clean, identity-verified full-18 panel. Primary endpoint BA18 and per-class same-row rescue/regression. Gate: proceed to one targeted new identity signal only if its error mass, independence and probable transfer justify budget; **STOP as BLOCKED** if clean panel, lineage, or full stage outputs cannot be obtained. No GPU rerun merely to reproduce contaminated old benchmark claims.

## Observer handoff

Next invocation: read [`OBSERVER.md`](../../OBSERVER.md), recheck current official Kaggle receipt from authenticated evidence, latest branch and latest Executor artifacts; compare with this dated snapshot. Never treat this file as a self-updating monitor. Only append a new dated review when explicitly authorized to write.

**No Executor source modifications, PR, GPU, training or submission occurred in this handoff.**
