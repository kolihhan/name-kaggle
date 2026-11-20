# NTL — provider original-output usage policy gate, 2026-11-20

## Scope
Public/current first-party policy review for prospective **18-label authorship classifier training**, combined with original Kaggle generator source audit. No raw dataset rows, model training, GPU, API data generation, competition submission, or payment. This is a **research risk inventory, not a legal opinion or license grant**.

## First-party references and interpretation

### Z.ai / GLM: exclude until explicit permission
Official Z.ai [API Additional Terms](https://chat.z.ai/legal-agreement/terms-of-service) (last updated 2026-04-14), Section API 1(f)(xii), restricts use of Z.ai-generated content for training, labeling and optimizing external models except specified authorized integration. It appears to reach classifiers as well as generative-model competitors. Therefore **do not add external Z.ai-origin GLM rows to NTL classifier training without provider-specific permission**. A publisher CC BY 4.0 label is not a waiver of Z.ai terms. This is the most concrete restrictive case.

### Anthropic / Claude: noncompeting classifiers expressly listed, subject to actual access terms
Anthropic [official help article](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model) (2026-03-16) expressly describes noncompeting specialist classifiers and content categorization as examples of permitted training on Claude Outputs, while restricting competing generative models. An authorship-attribution classifier plausibly falls in the noncompeting category, **but** the dataset producer used Claude on **Google Cloud Vertex AI**. Apply the actual customer's Vertex/third-party service agreement, transfer/license rights and source-specific terms; don't mark the uploaded dataset blanket-cleared simply from Anthropic's general help page.

### Google Cloud / Gemini: model-similarity/competition restrictions plus strong observed domain mismatch
[Google Cloud current service-specific terms](https://cloud.google.com/terms/service-terms) prohibit using generated AI output to substitute Google Models or create/improve models similar to Google Models, and competitive service development. A narrow attribution classifier is **not obviously similar** to a general-purpose Google Model, but proper contract scope/authorization must be confirmed. The original [lxdlxdlxd source README](https://www.kaggle.com/datasets/lxdlxdlxd/ntl-synth-generations-v1) reports strong Gemini-style mismatch to official competition TRAIN. **Do not treat these rows as host-matched just because generated from Gemini API.**

### OpenRouter / Qwen and other routed providers: second-layer provider model terms
[OpenRouter Terms of Service](https://openrouter.ai/terms), model terms Sections 5-6, explicitly keep each upstream provider's model-specific terms applicable. Output ownership is subject to the model's own terms. Thus a CC-licensed OpenRouter dataset **cannot be deemed training-authorized from the router's license alone**. Review actual provider/model route and terms before admission.

### Kimi / Z.AI-adjacent output
Moonshot/Kimi original service-output training terms and the exact route used for the frozen source **not verified in this review**. Keep Kimi examples quarantined pending original service terms/permissions. Avoid confusing Z.ai GLM restrictions with Kimi, which is a different model provider.

## Data-science decision

**NO-GO** on full 8,347-row, 18-class external CE with unfiltered source rights and domain shift. Stronger source lineage for the 4,270-row publisher **does not** overcome these restrictions. **Narrow next use:** official Kaggle TRAIN-only, true GroupKFold by prompt_id to establish a CE learning/exposure baseline using lawfully available competition data (DEV-only), or source-by-source admissible noncompeting-classifier data with a frozen family-aware domain-shift evaluation. The latter needs explicit source terms/route verification before training. Never label 0.702 improvement without authenticated official COMPLETE receipt.

Factual provenance and policy are separate from score prediction. These policies can change; recheck before score-facing promotion.