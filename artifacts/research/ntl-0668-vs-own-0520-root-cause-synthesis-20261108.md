# Why reproduced Public 0.668 beats own Public 0.520 — root-cause research decision

Date: 2026-11-08 MYT. **This is an evidence-based architecture diagnosis, NOT a trained candidate or submission.** The user explicitly rejects another series of small CPU experiments on consumed OOF.

## 1. Verified benchmark anchors

- **0.520**: own independently developed Qwen3-8B LoRA24, submission 57908683. [Original promotion](../../artifacts/candidates/two-stage-lora24-v1/SUBMISSION_PROMOTION.md).
- **0.668**: reproduced external Ozertuu Dual-SVC publicly published V3, submissions 58022340 and 58045266. [Authenticated historical submissions](https://github.com/kolihhan/name-kaggle/actions/runs/42385772029). The external Notebook title says 0.672; **we measured 0.668**, not 0.672.
- **0.666**: reproduced another author's Lightning V47; distinct model from Ozertuu.
- Difference **+0.148 BA / 14.8 percentage points** is a between-pipelines Public LB comparison, not a model-component ablation.
- Own grouped official OOF 0.546761 over 2,699 rows / 150 prompts / 18 classes. No honest fresh aligned Ozertuu OOF is available. These different metrics must not be compared as if paired.

## 2. Own system: actual frozen source-code mechanism

[Own fixed contract](../../artifacts/candidates/two-stage-lora24-v1/CONTRACT.md), [real Gate B](../../artifacts/candidates/two-stage-lora24-v1/GATE_B_RESULT.md), and [final inference implementation](../../scripts/two_stage_lora24_submission.py):

- One Qwen3-8B-Base trunk, **24 layers** of the 36-layer checkpoint, LoRA rank16/alpha32, mean-pooling layers 2/4/6/8/12/16/20/24.
- One epoch external source-pretraining, then 2 epochs fold-isolated official adaptation, 18-way plain head and centered diagnostic head.
- Stage-1 A1 source-heldout BA **0.465713** vs A0 frozen-backbone **0.394243** (gain **0.071470**). Stage-2 honest grouped OOF plain **0.546761**; the centered-head OOF 0.547850 is not submitted.
- Submitted inference uses **only the single adapted backbone and plain classification head**. No sparse readout, no multi-encoder ensemble, no family likelihood head, no inferred test prompt grouping, no test-marginal calibration.
- Conclusion supported: our own task-adaptation learns meaningful model identity signal. NOT supported: its failure is merely a bug, that a bigger backbone alone repairs it, or that the centered head will reliably rescue errors.

## 3. Ozertuu exact-source evidence AND current access limitation

The exact pinned external Notebook V3 SHA256 was historically verified as **05367a7f8efd41b2f4959505bc7e2e07a5978b85ee426c08bb5969c60945e2c2**, on 2026-11-05 [version-pinning Actions](https://github.com/kolihhan/name-kaggle/actions/runs/42024702664). An earlier [authenticated source-metadata audit](https://github.com/kolihhan/name-kaggle/actions/runs/42022226813) verified **9 code cells, 19 dataset inputs, 8 Notebook inputs, 8 model inputs**, GPU and internet-disabled. This exact submitted source obtained **our 0.668 score**.

**2026-11-08 new read-only source access failed:** [Actions 42473824023](https://github.com/kolihhan/name-kaggle/actions/runs/42473824023) returns Kaggle GetKernel 404 for the original pinned V3; a 2026-11-06 source pull had already received permission denied 403. The 2026-11-05 archive artifact is no longer retrievable. Therefore **the exact V3 execution code is not currently accessible line-by-line**.

Do NOT invent its precise blending weights, full model identities, overlap-free dataset counts, gains per component, or assert that the 19 inputs are 19 independent data sources. Do NOT substitute Lightning V47's exact source as if it were Ozertuu V3.

**Strong-system corroboration, not proof about Ozertuu:** Lightning V47's fully audited 0.666 code has heterogeneous prefit LoRA/model readouts (8B/14B/Llama/DeBERTa), trained sparse signals, inferred prompt-group-conditioned heads, conditional family gates, likelihood experts, a 27B auxiliary, and calibrated mixing. [Exact V47 analysis](ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md). Independently, [PiKa919/kaggle-name-that-llm](https://github.com/PiKa919/kaggle-name-that-llm) reports within **its own changing pipeline** 0.539 with first 8B LoRA, 0.600 with 8B+14B and several mechanisms, then 0.654 after six-backbone integration and 0.661 with additional features. These jointly changed multiple factors; they do not isolate a causal contribution.

## 4. Ranked explanations rather than invented causal ablations

1. **Plausible leading gap: diverse trainable representations and independent signal families.** Our own scored model supplies exactly one adapted learned-view argmax; independently audited 0.66+ systems combine neural, sparse, prompt-aware and conditional cues. This is the strongest explanatory hypothesis, NOT an isolated measurement for the exact 0.668.
2. **Training-source coverage and checkpoint pretraining may be the larger resource gap.** Bigger public systems mount many prefit artifacts; however raw dataset count says nothing about duplicate content, source quality, proper provenance or actual labels. More confounded data can HURT transfer.
3. **Inference-time grouping, source-family correction and calibration may matter**, but prior project centered OOF only marginally exceeded plain and almost equal rescues/regressions; hidden test does NOT expose prompt_id. A hidden distribution assumption can destroy Public transfer.
4. **A bigger single LM alone is not established.** Short-TRACE surrogate token-rank histogram standalone Public 0.093, DeTeCtive encoder OOF 0.0922, LUSR OOF 0.3512; none isolate 14B backbone quality or negate all other representation approaches.
5. **Provider/sibling errors remain heterogeneous** in own old Gate-B OOF. Gemini/GPT commonly confuse sibling models, GLM often confuses providers. These are descriptors from repeatedly consumed official examples, NOT fresh error rates for 0.668.

### External scientific evidence (with scope restrictions)

- [Every Model Leaves a Trace, Modarres et al. 2026](https://arxiv.org/abs/2610.28434): fingerprints at behavioral/stylometric/lexical/semantic resolutions survive some prompt and text transformations, but depend on model similarity and text evidence. **Not our exact 18-label task.**
- [ATLAS / MAABench, Ko et al. 2026](https://komseok0109.github.io/ATLAS/): source-conditioned token likelihood works under domain/version shift for **5 family labels**, with newer versions scored as the same family. Not proof of Luna/Sol/Terra separate-class identification.
- [OpenTuringBench / OTBDetector, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1354/): contrastive authorship and OOD settings, not direct transfer metrics for these 18 frontier siblings.
- Reject active prompt fingerprinting and observed diffusion decoding trajectories as evidence for the competition's passive single-paragraph setting.

## 5. Global architectural research decision (NOT new micro-experiment)

**Working scientific bottleneck:** We have **not demonstrated an extra, source-transferable exact-model information channel that corrects a strong heterogeneous anchor's remaining errors**. Our independent 8B signal is useful, but it is not equal-resource to the prefit multi-signal baselines.

**The overarching architecture target is a learnable fine-grained attribution representation whose complementary residual evidence survives new prompts/sources**, compared on the SAME rows against an actually strong anchor rather than against a weak control. The high-level idea of multi-resolution fusion is NOT novel; V47 already does this. Source-robustness is NOT obtained automatically from reweighting, contrastive labels, or naming a new adapter.

**Park, do not destroy or silently mutate, the separately pre-frozen conditional-denoising18-v1 contract.** It is not automatically justified as our next step: V47 already uses family likelihood, and we have not shown an output-only masked token decoder contributes beyond a strong full system. The task is architecture/data/compute identification, not another CPU score-gate loop.

**Next ONE coherent research-to-submission program:**
1. Establish a verifiable **strong anchor** and its legal row-aligned, full 18-label outputs on a source/prompt-held-out benchmark; use audited Lightning V47 0.666 as fallback if Ozertuu 0.668 source remains unavailable. The 0.668 Public score is a competition target, not a scientifically aligned training reference.
2. Select exactly one representation/objective with a **materially distinct information channel** compared to that anchor. Reject a candidate at design time when it only duplicates SVC, family-gating or surrogate rank-histogram evidence.
3. Reserve a genuinely source/prompt-disjoint evaluation cohort before any model adaptation. Historical official 2,699 and the repeatedly used abdullah/semenoff/zakir banks remain diagnostic only. Do not invent another salt or retune on Public LB.
4. Determine real checkpoints, model/source licenses, compute budget and exact version/protocol provenance. With the current **no local GPU / no paid API** restriction, an architecture requiring new major GPU fitting is not executable. **Do not choose weak CPU SVC merely for convenience** and do not assume the official Kaggle hidden test rerun trains a submitted model for us.
5. If a single end-to-end candidate earns independent evidence and a full offline hidden-size preflight, freeze and send ONE official Kaggle Code Competition submission; compare its actual Public result with 0.668. No score or Submission ID can be claimed now.

**Current research outcome:** rank representations/data diversity as most plausible gap, document exact Ozertuu source-access blocker and the experimental non-identifiability, reject automatic small-tweak loops and freeze no new model. This is substantive scientific prioritization, not a scored breakthrough.
