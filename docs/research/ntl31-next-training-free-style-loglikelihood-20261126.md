# NTL next research after 31B + Ensemble: training-free style likelihood, not another small head

**2026-11-26. Research stage only; NOT a submitted candidate.** Current official owner champion remains **0.718**, #58755304. One owner-authorized 31B Instruct GGUF + public-source Lightning v52 Ensemble candidate is undergoing real Kaggle runtime/preflight; do not start another score-facing notebook or retry it via an automation.

## New independent research evidence

**One-shot Style Transfer LLM log-probabilities for Authorship Attribution and Verification**, Miralles-González et al., Findings of EMNLP 2026 (pages 278–297): https://aclanthology.org/2026.findings-emnlp.14/

The authors use a causal LM's **conditional token log-probabilities** to estimate how transferrable a candidate source's writing style is to an unknown passage **without explicit fine-tuning**. Their abstract reports better authorship verification than direct unsupervised prompting at similar model size; performance improves with larger models. **This finding is for other authorship datasets, not for the Kaggle Name That LLM 18-way frontier-generator task.** No 0.76 claim is justified.

**ATLAS**, *Who's Behind the Text? Robust Model Authorship Attribution*, Ko et al., arXiv 2610.19583, https://arxiv.org/abs/2610.19583/. ATLAS compares clipped token-level likelihoods from **source-conditioned** LMs. The public Lightning 0.724 and current 0.718 lineages already use ATLAS-family upstream experts, so simply importing ATLAS is not a novel contribution. It also relies on source conditioning/learned components; it must not be falsely described as the same training-free mechanism.

## Single prospective zero-training experiment, conditional on valid 31B runtime

**Hypothesis:** A normalized conditional log-likelihood ratio under the **same 31B** for two putative generator styles contains source-specific information that its prompted direct yes/no classification loses.

For the same text x and source references r_A, r_B (reference prompts disjoint from x):
- `s_A = mean_t log p_31B(x_t | fixed_neutral_instruction, r_A, prefix(x)_{<t})`.
- `s_B = mean_t log p_31B(x_t | fixed_neutral_instruction, r_B, prefix(x)_{<t})`.
- Candidate is `argmax(s_A,s_B)`. Use identical tokenization, text span, truncation, instruction and reference lengths across A/B. Never compare differently-tokenized raw totals. Use token-level clipping only if a *predeclared*, source-blind reason exists.
- A no-example/common-neutral prompt computes a control for spurious token-frequency preference.
- Random/shuffled-source reference is a mandatory negative control; if source labels cease to matter, retire the hypothesis.

**Scope:** ONLY the same lowest-margin top-two rows already eligible in the 31B competition candidate; no uniform 18-class quota, no inference on hidden labels, no gradient updates. Predeclare batch size and a resource budget *after real hardware throughput* — do not add an 18× full-corpus pass to an 8–9h pipeline.

**Comparators:** 31B direct ICL from the current candidate, its frozen Ensemble prediction, and identical model/runtimes without source references. Compare paired wins/losses **on the same prompt-disjoint truth set**, with 18-way BA and per-class recall when enough labels are represented. Avoid treating previously inspected 252 Fable rows or external Open Pool-overlapping panels as independent. Prefer new rights-cleared, documented API-generation source or official TRAIN prompt-group development with honest DEV labeling.

**Engineering feasibility finding:** The published llama.cpp server README documents `n_probs` for **generated tokens**, not a guaranteed teacher-forced log-probability for every input/prompt token (https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). A 2026-10-31 implementation audit reports silent token-ID/logprob loss for some multi-byte token cases (https://huggingface.co/blog/bytkim/llama-cpp-logprob-loss). Consequently, scoring full unknown passages via the chat-completion endpoint is **not yet a reproducible implementation** of the EMNLP objective. Before a score-facing candidate, pin a library/engine with actual teacher-forced log-probabilities, exact token-span masks, UTF-8/token-ID round trips and a known-good regression test; or reject this lane as too expensive. Do not reinterpret generated label-token probabilities as conditional passage likelihood.

**Kill condition:** No documented source-disjoint signal, no actual real GGUF token logprobs, or paired rescues ≤ regressions (especially for GPT sibling classes) means **no Kaggle submission**. A score improvement from a different experimental source is not a reliable predicted Public BA.

## Why this, not E5 again

Prior experiments found:
- Frozen 335M E5 binary Kimi/GPT BA 70%, **ties a one-feature em-dash control**.
- The E5 hard switch degraded historically spent 18-class paired results 151/252 → 148/252.
- Prompt-matched contrastive C1 on the full 2,699 official TRAIN had 18-way grouped OOF **0.183** versus plain-CE **0.192**. This disproved only that narrow frozen-representation contrastive objective; it does not show all big-model methods fail.

**31B current candidate still has not returned an official numeric Kaggle score at the time this note was authored.** Its operational failures (preview-only NB_T0; CUDA 12 runtime) are startup issues and do not measure model discrimination.

## Source references
- Miralles-González et al., ACL Anthology 2026: https://aclanthology.org/2026.findings-emnlp.14/
- Ko et al., ATLAS paper: https://arxiv.org/abs/2610.19583
- Our 31B one-shot decision contract: `docs/research/ntl31-gemma4-inference-only-one-shot-contract-20261126.md`.
