# Phase 18 — external-data transfer-risk audit

Date: 2026-11-01

## Verdict

**Phase 15's external corpus is materially closer to the official training responses than the Phase 7 external corpus under a common text-similarity audit. Treat this as a transfer-risk signal, not proof of leaked official prompts.**

This audit was opened after Phase 15 improved permanent-fold OOF from the Phase 7 reference `0.3700521998508576` to `0.4053218990802883`, but then scored only `0.357` on the Public leaderboard versus Phase 7's `0.369`.

No model was fitted, no permanent-fold score was read, and no Kaggle submission was made in Phase 18.

## Controlled comparison

All three corpora were embedded in one shared word-TF-IDF space:

- official train: **2,699** responses;
- Phase 7 external: **2,503** responses / 8 labels;
- Phase 15 external: **26,025** responses / 11 labels.

Frozen audit representation:

- analyzer: word;
- ngrams: 1–2;
- lowercase: true;
- min_df: 1;
- max_features: 80,000;
- sublinear_tf: true;
- vocabulary fitted jointly on official + Phase 7 external + Phase 15 external.

For every official response, the audit measured its maximum cosine similarity to external responses, including a **cross-model** nearest-neighbor statistic that excludes external rows with the official row's true model label.

## Headline result

### Official -> Phase 7 external, cross-model nearest neighbor

- p50: `0.07735948264598846`
- p95: `0.1120489105582237`
- p99: `0.13387077540159226`
- max: `0.1786506474018097`
- rows above `0.25`: **0**
- rows above `0.50`: **0**
- exact normalized response overlap: **0**

### Official -> Phase 15 external, cross-model nearest neighbor

- p50: `0.10356060415506363`
- p95: `0.1840477243065834`
- p99: `0.2398239168524742`
- max: `0.3643922209739685`
- rows above `0.25`: **24**
- rows above `0.50`: **0**
- exact normalized response overlap: **0**

### Direct Phase 15 minus Phase 7 difference

- cross-model p95: **+`0.07199881374835969`**
- cross-model p99: **+`0.10595314145088194`**
- official rows with cross-model nearest similarity > `0.25`: **+24**

The Phase 15 cross-model p95 is higher than Phase 7 for every official model label in this audit.

## Where the highest similarities come from

Among the 24 official rows whose Phase 15 cross-model nearest neighbor exceeds `0.25`, nearest-neighbor external sources are:

- `maniceet`: **17**
- `hailey`: **4**
- `keshav`: **2**
- `seyit`: **1**

This is diagnostic evidence only. **Do not now remove or downweight these sources based on this result.** That would be post-result tuning after observing both Phase 15 OOF and Public LB.

## Why this matters

Phase 15's local uplift was concentrated in model families represented by its large external corpus, while several labels without eligible external rows lost recall. The Public-LB regression then showed that the local uplift did not transfer.

Phase 18 provides a plausible mechanism: the large Phase 15 corpus is much closer to the official training-response distribution than the much smaller Phase 7 corpus. That can make grouped OOF on the same 150 official prompt groups look unusually strong even when the learned signal does not generalize to hidden prompts/topics.

This audit **does not prove prompt leakage**:

- exact normalized response overlap is zero;
- official prompt text is not available for direct semantic prompt matching;
- high response similarity can arise from topic/domain proximity without an official prompt being copied.

The correct conclusion is therefore **transfer-risk / domain-proximity evidence**, not a leakage accusation.

## New external-data gate

Future external-data proposals should be screened **before** any permanent-fold model result is observed.

At minimum, a proposal must pre-register:

1. exact source identities and immutable data hashes;
2. prompt provenance / independence evidence;
3. provider-output terms / competition eligibility;
4. exact normalized overlap against official responses;
5. a fixed official-to-external similarity audit using a representation and thresholds frozen before modeling;
6. a comparison against a known-transfer control corpus such as Phase 7 external data;
7. the action to take if the new corpus is materially closer to official responses than the control.

Phase 18 does **not** retroactively define a numeric cutoff for the current pool. Choosing a threshold now from the observed Phase 15 failure would itself be result-dependent. A later experiment must freeze its gate before inspecting its own OOF result.

## Current competition state

Phase 7 remains the best verified Kaggle system:

- local OOF BA: `0.3700521998508576`;
- Public LB: `0.369`;
- submission ref: `57838845`.

Phase 15 remains rejected as competition champion:

- local OOF BA: `0.4053218990802883`;
- Public LB: `0.357`;
- submission ref: `57861337`.

Phase 17 global balancing was also rejected locally at `0.3667238379318916` and was not submitted.

## Evidence

- audit workflow: `41282112530`
- artifact: `phase18-external-transfer-risk-audit`
- artifact ID: `13074295237`
- artifact digest: `sha256:17beddbc2ddf78d9ce78bd9943dd7a9018a0e889d223f13086a7d348503d9e8a`
- Phase 15 submission record: `artifacts/submission/SECOND_SUBMISSION.md`
- Phase 15 local result: `artifacts/phase15/terms_clean_sparse.json`
- Phase 7 local result: `artifacts/phase7/gated_specialist.json`
- Phase 7 original external audit: `artifacts/phase2/gpt_gemini_raw_audit.json`

## Next gate

Do not authorize another Kaggle submission from external-data uplift alone. The next candidate should either:

- use a genuinely different, official-only representation/mechanism; or
- introduce new external evidence that passes a precommitted transfer-risk gate before permanent-fold evaluation.

Because the same permanent folds have already supported many sequential decisions, the next modeling phase should also add a precommitted robustness check rather than treating one OOF number as sufficient promotion evidence.
