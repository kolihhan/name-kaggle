# ATLAS (2026) versus SupCon versus NTL 0.715 — verified research comparison

**Decision 2026-11-23: ATLAS is an ACTIVE paper candidate, but its full retraining is deferred; its robust readout is the first zero-additional-inference feasibility target.** No Kaggle submission or paid API.

## Verified original research, not a title-based extrapolation

- Paper: Minseok Ko et al., *Who's Behind the Text? Robust Model Authorship Attribution* (Oct 14, 2026): https://arxiv.org/abs/2610.19583
- Official authors' site: https://komseok0109.github.io/ATLAS/
- Official open implementation: https://github.com/komseok0109/ATLAS . Examined README.md, src/atlas/model.py, src/atlas/collect_scores.py and src/atlas/eval.py.
- Genuine published number: **74.27% accuracy** on MAABench's joint unseen-domain AND newer-version shift. +**9.00 pp** versus that paper's strongest baseline. There are **five model families** (GPT, Claude, Gemini, Mistral, Llama); this is not NTL **18-way exact sibling Balanced Accuracy**.
- Paper's base: pretrained **Qwen2.5-7B** causal LM, frozen main weights, one learned source-control prefix per source and per-source LoRA up-projection B_k with shared LoRA down-projection A. Optimize **next-token likelihood under true source** instead of ordinary multiclass CE.
- Inference requires **all K source conditions** for each text. Official model code expands input from B to K*B conditioned rows. It is locally run, not one paid API call per source, but increases token-forward compute significantly.
- Exact main readout (src/atlas/model.py:_prod_readout): for source k and token t, score = exp( - Σ_j max(logp_j,t - logp_k,t, 0) / T ). Average across valid tokens, then over fixed T = {1,2,4,8}; argmax chooses the winning source. Bounded [0,1] evidence prevents a few extreme token-level differences from entirely dominating. This is distinct from ordinary mean log probability.

## Honest comparison

| Dimension | Current own scored 0.715 | ATLAS | Prompt-SupCon paper route |
|---|---|---|---|
| Prediction | 18 exact model labels | Five model families | Train 18 exact model labels |
| Signal | Qwen/other pretrained experts, GLM likelihood, style, prompt grouping, family residual | **Source-conditioned autoregressive token likelihood** | **Contrastive prompt-invariant embeddings** |
| Training | Multiple inherited tuned assets | Source prefix and shared-A / specific-B LoRA | Small projection + CE versus CE+SupCon on frozen genuinely strong embeddings |
| Domain shift | Oleg216 diagnostic tested old 0.691 in sensitive topics, not current 0.715 | Joint domain/version shift benchmark reported | Must validate cross-prompt and independent source |
| Compute | Already heavy offline Kaggle run | K conditioned LM forward passes per sample | Frozen embedding once then small head (if feature assets exist) |
| What remains unproven | True paired 0.715 external labels | Direct NTL BA or fixed 9h viability | Actual 18-class BA improvement and strong-feature access |

## ATLAS-derived experiments — ranked by scientific and compute value

**A0: CPU token-evidence readout, no additional generator compute (priority now).**
1. Reuse only **SHA-pinned existing aligned per-token likelihood arrays** of GLM-5.3 versus GLM-5.3-Flash, studied in scripts/ntl_glm_distributed_evidence_20261122.py; validate identical IDs, tokenizer, offsets, token alignment, label support, historical prompt-fold and train exposure.
2. C0 = ordinary mean token log likelihood per candidate; C1 = **exact original ATLAS argmax-prod** bounded token aggregation with **all fixed temperatures {1,2,4,8}**, no fitted thresholds, training, external calls or new LM forward pass. For only two GLM candidates this is a **two-sibling mechanism diagnostic**, not a complete ATLAS 5-source reproduction or 18-class NTL result.
3. Compute same-row correct/rescued/regressed, family/sibling BA where defined and input source drift. **Reject** if true per-token vectors are unavailable, candidate has nonaligned tokens, or improvement exists only in already-spent OOF without evidence of transfer. Never promote a 2-way result to an 18-way competition claim.

**A1: Faithful learned-source ATLAS, deferred and not authorized at current NTL cost boundary.**
1. Keep original next-token likelihood objective, source prefix and per-source LoRA structure. An SVC on char ngrams or a posthoc threshold is **not ATLAS**.
2. Decide a priori 6-family versus 18-label scope. A 6-family classifier cannot solve within-OpenAI sibling confusions; 18 conditioned forwards per row greatly increase runtime.
3. Model exact GPU-hours, resident weights, K-way token forward count and end-to-end Kaggle offline **9-hour** limit. User rejects paid APIs / expensive stacking and requested no new NTL QLoRA. Do not launch this retraining unless separately permitted and fixed-budget feasibility verified.
4. Compare valid real predictions to existing frozen 0.715 on **the same independent 18-way labelled rows**. Require source/prompt group controls, exact rescues/regressions, per-class BA and elapsed cost.

**B: SupCon stays an independent competing paper mechanism**, implemented in scripts/ntl_paper_prompt_supcon_20261123.py. SupCon uses cross-prompt positives and within-prompt other-model hard negatives; ATLAS models each class's distribution instead of training a source label discriminator. Do not call a blend a tested new invention.

## Execution/status boundary

- Full source and paper **verified** (original author's code inspected). A0 real GLM evaluation **not run** as of this note. SupCon smoke / strong-feature recovery jobs currently queued.
- Official own Champion **0.715 Public BA** unchanged. ATLAS 74.27% is a different 5-family task and **not a numerical forecast** for this competition.
- No paid API, no new QLoRA, no hidden test, no new official Kaggle submission.
