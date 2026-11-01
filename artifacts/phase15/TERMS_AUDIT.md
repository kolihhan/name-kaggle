# Phase 15 — provider-terms-clean external-data admission

Date: 2026-11-01

## Why Phase 15 exists

Phase 14 produced a strong frozen local result (`0.4178100919711658` balanced accuracy), but its external-data reconnaissance audited prompt/source provenance rather than applying the same provider-output-terms standard used in Phase 13.

That is an internal consistency problem. Phase 13 intentionally rejected technically clean Gemini 3.7 traces because Google provider terms did not authorize using generated content to develop an ML model. Phase 14 cannot use a weaker standard merely because its local score is higher.

This document freezes the provider-terms decision **before** any Phase 15 permanent-fold result is observed.

## Current public terms evidence

### Google / Gemini — BLOCK

Google's current Generative AI Additional Terms state that the Services may not be used to develop machine-learning models or related technology. Google's general Terms also explicitly prohibit using AI-generated content from its services to develop machine-learning models or related AI technology.

Sources checked 2026-11-01:

- https://policies.google.com/terms/generative-ai
- https://policies.google.com/terms

Consequence: no external Gemini-labelled generated passage is admitted for classifier training under this repository's conservative standard.

Blocked labels:

- `gemini-3.1-pro-preview`
- `gemini-3.6-flash`
- `gemini-3.7-flash`
- `gemini-3.8-flash`

### Kimi — BLOCK / AMBIGUOUS

Kimi's current public Terms state, among other restrictions, that without written consent users have no authority to use Kimi-generated content in a commercial manner, and separately prohibit training/improving algorithms or models that directly or indirectly compete with Kimi.

Source checked 2026-11-01:

- https://www.kimi.com/user/agreement/en/modelUse

A prize-bearing competition creates enough ambiguity around the commercial-use clause that this repository does not treat public dataset licensing alone as a clear downstream-output grant. The raw pool also does not prove a cleaner provider route for every retained Kimi passage.

Consequence: external `kimi-k3` passages are not admitted for Phase 15.

### Zhipu / GLM — UNKNOWN

Current official BigModel/Zhipu documentation confirms model/API usage, but this audit did not locate a sufficiently explicit downstream-output grant that resolves using third-party shared GLM outputs as training data for a new classifier.

Sources checked 2026-11-01:

- https://www.bigmodel.cn/
- https://docs.bigmodel.cn/

Under the repository's existing conservative rule, UNKNOWN is not PASS.

Blocked labels:

- `glm-5.3`
- `glm-5.3-flash`

### OpenAI-labelled outputs — ADMITTED FOR THIS USE

Current OpenAI restrictions focus on using outputs to develop models intended as alternatives to or substitutes for OpenAI models/services. The Name That LLM attribution classifier is not a substitute foundation model or service.

Sources checked 2026-11-01:

- https://openai.com/policies/row-terms-of-use/
- https://openai.com/policies/usage-policies/

This is not a general permission statement; it is only the admission judgment for this narrow non-competing attribution-classifier use.

### Anthropic-labelled outputs — ADMITTED FOR THIS USE

Anthropic's commercial terms assign output rights to the customer to the extent permitted by law and restrict using the Services to build a competing product/service or train competing AI models. This attribution classifier is not a competing foundation-model product.

Source checked 2026-11-01:

- https://www.anthropic.com/legal/commercial-terms

### Qwen-labelled outputs — ADMITTED UNDER AVAILABLE EVIDENCE

Alibaba Cloud Model Studio terms describe customer Output rights/assignment and do not surface the broad Google-style prohibition against using generated content to develop any ML model. The same platform explicitly supports model training workflows.

Sources checked 2026-11-01:

- https://www.alibabacloud.com/help/en/legal/latest/alibaba-cloud-international-website-product-terms-of-service
- https://www.alibabacloud.com/help/en/model-studio/model-training-and-deployment-billing

## Frozen Phase 15 external filter

Start from the exact mechanically rebuilt Phase 14 subset and remove external rows whose `model` is one of:

```text
gemini-3.1-pro-preview
gemini-3.6-flash
gemini-3.7-flash
gemini-3.8-flash
glm-5.3
glm-5.3-flash
kimi-k3
```

Do not change official competition training rows.

From the frozen Phase 14 per-label counts, this removes exactly **17,756** external rows and leaves an expected **26,025** external rows before the audit workflow independently verifies the count.

This filter is based only on provider-terms evidence and the pre-existing label field. It does not inspect Phase 14 OOF errors, margins, fold outcomes, or per-class gains.

## Next gate

1. Rebuild the exact Phase 14 43,781-row subset from the pinned raw pool.
2. Apply the frozen seven-label external exclusion above.
3. Audit exact row count, remaining label counts, prompt count, nulls, and raw-pool SHA.
4. Only if that audit passes, freeze a Phase 15 permanent-fold contract using the **same Phase 14 four-view feature recipe and same LinearSVC configuration**.
5. Promotion should compare against the last submission-eligible promoted system, Phase 7 (`0.3700521998508576`), not require matching the terms-blocked Phase 14 score.

No Kaggle submission is authorized by this document alone.
