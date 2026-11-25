# NTL: Catch-up research pivot — 2026-11-26 (MYT)

**Research status:** evidence-based plan, **NOT** a new model result or submitted score. **No official Kaggle submission, no GPU launch, no training.** Confidence Fusion stays retired after two identical official zero-byte format errors. Current authenticated best remains **Public BA 0.718** (#58755304).

## Live competitive target

Authenticated Kaggle CLI leaderboard as of the read-only [Action #45659328905](https://github.com/kolihhan/name-kaggle/actions/runs/45659328905):
- Public leader 0.813 (NguyenThanhNhan).
- Rank 10 threshold 0.777, rank 20 threshold 0.761.
- Our 0.718 => **+0.043 BA** needed to equal the observed top-20 threshold. This is a moving target, not a guaranteed medal.
- Best recently reviewed *public-code* notebook, Lightning v52: Public BA **0.724** (prior authenticated version audit). Its +0.006 over 0.718 is far too small to fill the 0.043 gap. No evidence of the private top teams' methods. Don't falsely claim leaders train on a specific corpus.

Competition format: 18 labels, hidden ~4000 rows, unseen prompts, 40% Public / 60% Private, per-model frequency shifts; official 9h GPU ceiling. Sources:
- https://www.kaggle.com/competitions/name-that-llm/data
- https://www.kaggle.com/competitions/name-that-llm/leaderboard

## Highest-value measured bottlenecks, conditional on provenance

Existing archived read-only *spent* publisher-labelled 252-row panel from [docs/research/ntl-cross-family-headroom-result-20261125.md](ntl-cross-family-headroom-result-20261125.md):
- 151/252 correct, 51 errors cross-provider and 50 within-provider.
- **Gemini 3.7 Flash 0/14** correct; 8/14 mistaken for Gemini 3.8.
- **Kimi K3 1/14** correct; 10/14 mistaken for GPT labels.
- Gemini 3.6 Flash 4/14; GLM truths 12/28.
- 14/label is tiny. These are publisher-declared identities, not independent provider authentication; this panel has already influenced experiments. The patterns are hypotheses, not hidden-set error rates.
- If exactly 11 additional 252-panel rows changed from wrong to right without regression, its own BA would move by ~0.04365. This is a *design-scale illustration*, not an expected Kaggle improvement.

Known negative results:
- Same-run confidence fusion GSQ correct **147→145** (4 rescues, 6 regressions). Fully retired.
- PB115 token evidence: GLM full-vs-Flash scalar rank already **0.98** on 150 historical matched pairs; none of 61 individual PB115 features surpassed scalar rank on the original GLM-vs-non-GLM ranking. [Read existing audit](ntl-pb115-token-evidence-result-20261125.md). No more generic PB115 weights.
- The original Phase7 vs Phase15 used 26,025 external rows supporting 11 of 18 classes. On spent grouped OOF, BA +0.035, but official Public BA **-0.012**. Seven unsupported labels had mean recall loss -0.110; 11 supported labels gained +0.128. Confounded model changes, but this warns that a bigger *unbalanced* corpus can regress true 18-way BA.
- Kaggle forum explicitly reports prompt-cluster + Hungarian one-of-each assignment raised local OOF 0.374→0.481, but Public declined 0.353→0.350. Do **not** use mandatory test-cluster matching: https://www.kaggle.com/competitions/name-that-llm/discussion/751802

## What new public data actually provides

Maniceet Sahay's 7,753-row, 10-label API-generation corpus: https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations

Published source inventory, **not independently raw-row verified**:
- Gemini 3.7: **22**; Kimi K3: **63**; Qwen 3.8 Max: **5**.
- GLM full: 1661 OpenRouter cheapest-provider + 178 official-provider; GLM Flash: 2238 + 590. Many samples have source routing/protocol changes, not uniform served provider or generation settings.
- The publisher reports `provider`, `served_model`, `prompt_exact`, `source_tag`; such fields enable inspection but DO NOT independently authenticate model provenance.
- Public dataset includes paid-API generations and a CC BY 4.0 license. Repo's current eligibility policy **still requires competition-rule and rights review** before using any paid-origin output for training or scoring.
- Authenticated read-only [Action #45659699184](https://github.com/kolihhan/name-kaggle/actions/runs/45659699184) attempted public Kaggle dataset download; Kaggle returned nonzero (dataset listing and download unavailable). This is a **blocked source**, not an inspected training set.
- Source counts target our exact weak classes badly. Do not spend additional time on a blocked download, or equate 7753 with 7753 independently useful correct-label examples.

## Prospective high-upside hypothesis (one experiment, not a patch sequence)

**H1: Part of the catastrophic Kimi→GPT and Gemini 3.7→3.8 attribution failures reflects *served-model/protocol misalignment* plus source-specific representation learned from unsupported model corpora, rather than insufficient ensemble calibration.** Competing hypothesis: the models' genuine generation signatures are intrinsically too close in 40–250 word single paragraphs; additional provider-verified rows do not transfer under host prompt/length protocol.

**Priority A (CPU data gate, no model fitting):** Find source-consented, accessible examples with original request, *actual served-model identifier*, generation date/version, provider/router, temperature/reasoning/template, target-word count, and paragraph formatting. Focus on **Kimi K3, Gemini 3.7/3.8, GPT 5.6 variants**, not another 3000 GLM rows. Exclude provider-fallback ambiguous records, unofficial label aliases, overlap with official responses, reconstructed official train prompts, non-disjoint prompt groups, and unlicensed/private material. Preserve class and source counts. Try legal public sources already accessible before adding calls. Stop if sample authenticity/rights cannot be verified; no fabricating label coverage.

**Priority B (one predeclared distinct model hypothesis only after A passes):** Compare the exact unchanged frozen strong source ensemble to one **provider-verified, prompt-invariant hard-negative identity model** on the *same independent prompt-disjoint source/test rows*. The new model must create distinct evidence for (Kimi vs GPT) and (Gemini 3.7 vs 3.8), not a threshold patch to existing predictions. Use a strong frozen proxy representation and train/evaluate a *source-conditional contrastive discriminator* only with admitted material, with:
- matched and shuffled-source negative controls;
- leave-one-prompt-group and leave-one-provider-routing holdouts;
- no classifier trained on current 252 panel or Kaggle hidden feedback;
- full 18-class macro BA and per-class rescues/regressions, not just a positive hard-pair accuracy;
- frozen inference runtime within 9h; strict output-ID/row schema contract on a full-size dry run before any future score attempt.

**Evidence gate:** Would need convincing provider-authenticated independent per-label evidence (not a tiny 22/63 sample) and an observed material paired gain on an independently held source split before contemplating any new code competition score bet. No further Official/Confidence Fusion submissions without separate explicit owner authorization.

## Research alternatives checked and rejected for now

1. **READER (2026)**: strong *multi-response* attribution uses accumulation over 50–100 separate prompts; Name That LLM scores each isolated single paragraph with no cross-query identity link. Its multi-response gains are not directly applicable; the paper reports materially lower single-response accuracy. https://arxiv.org/abs/2606.10794
2. **ATLAS (2026)**: likelihood source modeling improved OOD 5-family attribution on MAABench, but the 0.718 architecture already includes an ATLAS lineage, and family-level gains do not establish 18-way sibling separation. Blindly adding it again is not a new independent source. https://arxiv.org/abs/2610.19583
3. **Prompt Hungarian quotas**: the local-vs-Public counterexample above directly challenges test-cluster assumptions. No forced quota heuristic.

## Research decision

**Do not do another 0.718→0.720 calibration patch.** Race to top 20 requires a *qualitatively new, trustworthy identity signal* on the actual weak model classes. The main constraint is authentic source/protocol comparability and verified rights, not an untested ensemble coefficient. The next worthwhile action is a focused **source-level, per-class authenticity admission audit**; if authentic Kimi/Gemini evidence cannot be sourced, report the limitation and change to a different validated modeling class rather than burning another official submission.
