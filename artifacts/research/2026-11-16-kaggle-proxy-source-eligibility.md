# NTL — Public Kaggle Benchmarks Proxy dataset: prospective source admission audit
Date: 2026-11-16 (MYT). Scope: **metadata/data-card audit only, no model inference**.

## Predeclared decision rule
Inherited unchanged from [VALIDATION.md](../../VALIDATION.md) and [independent source gate](ntl-18class-independent-validation-gate-20261111.md). Admit as independent full 18-way validation **only if** all 18 true served-model labels, exact prompt/response lineage, licensing, new/disjoint group coverage and prefit isolation can be established, before candidate inference. If a necessary condition fails, mark **NOT ADMITTED (for 18-way confirmation)**; this is *not* an accuracy comparison or permanent data-source ban. CI is N/A for n=0 model predictions.

## Candidate — new in this audit
- [Kaggle dataset: zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations](https://www.kaggle.com/datasets/zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations).
- Public data card describes free Kaggle Benchmarks Model Proxy generations from GPT and Gemini models; explicit CC BY 4.0, version 1 advertised **930.91 kB** CSV + README. **No raw rows materialized, no row count or model-by-model sample counts independently verified here.**
- Description says one fresh request per row with exact request text, prompt bank attribution to seyitkaangunes/prvsiyan, target word count, model proxy slug, reasoning low, temp 1, max_tokens 4096 for 'daily' rows. Other ablation runs use different settings on same prompts.
- The dataset card does not claim 18-lab coverage: **GPT + Gemini only**. Official competition classes include five GPT and four Gemini among 18 total, so this channel can cover **at most 9/18** model classes; actual observed number may be less. Remaining **at least nine** (Anthropic five, Qwen one, ZAI two, Kimi one) are out of scope by source description.
- Parent prompt banks seyitkaangunes and prvsiyan also appear in our [existing intake](ntl-18class-independent-validation-gate-20261111.md), and prvsiyan was directly mounted by champion; exact prompt overlap and checkpoint prefit overlap UNKNOWN. Per-row exact request is a useful feature, but not proof of independent unseen generation.
- The model `proxy_model_slug` records requested route. **Actual served model identity/version** is not established from the public data-card excerpt; no claim of false identity is made.

## Falsifier and result
Hypothesis: this newly found public source can provide a complete, independent 18-label same-row holdout.

**Observed falsifier [VERIFIED from source data-card scope + official 18 labels]:** GPT/Gemini-only provider coverage is strictly a subset of the competition's six labs. Maximum possible label coverage **9/18**; actual per-label response counts, unique independent prompts, served ID, source-overlap and legality beyond CC BY **not fully audited**.

- Number of audited source cards: **1**.
- Number of actually scored model passages: **n=0**. Cluster bootstrap confidence interval: **N/A**.
- Number of Kaggle official submissions: **0**. GPU use/training/API calls: **0**.
- Provenance: [original Kaggle source card](https://www.kaggle.com/datasets/zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations); [official 18 labels](https://www.kaggle.com/competitions/name-that-llm/overview/description).

## Decision
**[VERIFIED source-scope rejection] NOT ADMITTED as 18-class independent benchmark.** Reject only the claim that this dataset meets the full 18-class evidence gate. It could be a future **conditional GPT/Gemini source/channel diagnostic**, but not a fair 18-class BA or fresh unseen holdout without overlap checks. No relaxation of known held-out or model-ID requirements.

**Single next action:** pivot source discovery away from repeatedly counting incomplete third-party corpora. Specify and seek a source-provenance channel covering the currently missing Anthropic/Qwen/ZAI/Kimi labels and compatible GPT/Gemini groups, with actual served IDs and license, before any inference or use of paid APIs.
