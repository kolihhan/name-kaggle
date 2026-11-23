# NTL — Fresh 18-Class Cross-Source Identity: Feasibility Gate (2026-11-23)

**Purpose:** obtain the evidence necessary to test a genuinely new **source-robust model-family identity mechanism**, rather than revisiting GLM-only likelihood gates or spent official grouped OOF.

## Verified starting points
- Own authenticated **Public BA = 0.715**; freeze the scored incumbent. Do not compare a 0.715 Public score numerically to an offline 0.3671 OOF score as if they came from the same evaluation.
- The 64-prompt GLM family misses **33/128** were measured on the **old 0.683** classifier, *not the 0.715 incumbent*. The 95 family-positive texts were 88/95 exact-correct, explaining why old sibling-only reranking had little upside.
- GLM source shift: a 150-pair official versus 64-pair original-generator analysis found some source marginals moving, but all 10 Full-minus-Flash paired cross-source change CIs included zero.
- Earlier frozen 0.691 champion scored **89/216 = 0.412037** BA on an external 18-label source; this was a particular *spent* source/domain, not a fresh 0.715 diagnosis.
- Original `lxdlxdlxd/ntl-synth-generations-v1` covers only **13/18** labels, with remaining source-rights and provider-protocol issues. The previously catalogued 18-label pools are not established as unused and independent.

## Competing hypotheses (future experiment; not yet executed)
**H1 — identity failure:** once generation source and prompt protocol are controlled, an independently sourced family identity representation reduces cross-family errors (without drawing foreign classes into GLM/Claude/GPT).

**H2 — channel dominance:** model identity, source, formatting and prompt protocol remain confounded; family routing changes only the error allocation and makes false-positive capture worse, even when some target-family recall improves.

**Discriminator:** on *previously unused* source-disjoint matched-prompt panels containing all exact 18 truth labels, score the unchanged 0.715 candidate and one frozen identity representation on exactly the same rows. Report per-provider-family recall and specificity/false-positive capture from every other family, per-label BA, rescued and regressed rows, and prompt-cluster intervals. No subset BA passed off as 18-way BA. Compare H1 vs H2 over at least two genuinely distinct collection sources/protocols, not merely two random seeds from one provider.

## Phase 0 only — actual authorized read-only check
Check live **Kaggle Benchmarks Model Proxy available model slugs** and **daily/monthly inference quota** through the existing `KAGGLE_API_TOKEN` secret. No prompts sent, no token inference, no new datasets, no training, no public repository text uploads, no competition submission. Pure model catalog availability is *not* proof that the provider actually served that model/version; this check only determines whether targeted independent acquisition is feasible.

Success criteria for acquisition eligibility:
1. All **18 exact named targets** available or an explicit route with verified *returned* model/version identity for the missing ones (the catalog alone is not enough).
2. Enough verified zero-out-of-pocket quota for an explicitly frozen, small full-18 matched-prompt source; **don't assume an example $5 daily quota describes this account**.
3. Permission to evaluate outputs, provenance of user prompts, response ids, true provider-returned model versions, system/reasoning/temperature/top-p/length settings, near-duplicate and prefit exposure controls.
4. At least one additional independent acquisition channel is necessary before calling any result **cross-source**.

If any of these is unavailable, report **BLOCKED** with exact missing prerequisites; do **not** automatically generate content or silently spend quota.

## Later promotion gate (not authorized by this Phase 0)
Only consider a frozen candidate after a genuinely unused full-18 panel exists with sufficiently independent prompt clusters and a second source; evaluate exact original 0.715 source and any new model on identical rows. Predeclare a paired improvement threshold, accept full-18 class-specific failure costs, and do not use the already-spent 64 GLM/216 Oleg/2699 official folds or Public leaderboard to select thresholds. No new training, GPU, official Kaggle submission or score promotion on this branch unless separately enabled by a later evidence-backed contract.

## Research status
**Phase 0: executable GitHub Actions audit.** This is an acquisition *feasibility* check, not a scored classification experiment. External evidence requirements govern further work.
