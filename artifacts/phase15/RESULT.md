# Phase 15 — provider-terms-clean public sparse result

Date: 2026-11-01

## Final verdict

**LOCAL PROMOTION PASSED; KAGGLE PROMOTION FAILED. Keep Phase 7 as the competition champion.**

The single frozen permanent-fold run passed every committed local invariant and materially exceeded Phase 7 in OOF validation. A subsequently authorized second real Kaggle submission completed successfully but scored **0.357 Public**, below Phase 7's **0.369 Public**.

Therefore:

- Phase 15 remains a valid frozen local experiment result;
- Phase 15 is **not** the current competition system;
- the Phase 14/15 external-pool OOF uplift is not a reliable hidden-test proxy;
- no result-dependent tuning of Phase 15 is authorized from this leaderboard outcome.

## Frozen local comparison

| System | OOF balanced accuracy | Delta vs Phase 7 |
|---|---:|---:|
| Phase 7 | 0.3700521998508576 | — |
| Phase 14 diagnostic (terms-blocked) | 0.4178100919711658 | +0.04775789212030823 |
| **Phase 15 terms-clean** | **0.4053218990802883** | **+0.03526969922943074** |

Fold scores:

`0.416667 / 0.381481 / 0.392593 / 0.416667 / 0.419093`

## Real Kaggle comparison

| Submission | System | Public LB |
|---|---|---:|
| `57838845` | Phase 7 gated specialist | **0.369** |
| `57861337` | Phase 15 terms-clean public sparse | **0.357** |

Phase 15 validation-to-Public gap: `-0.0483218990802883`.

Phase 15 Public delta vs Phase 7 Public: `-0.012`.

The Phase 15 submission used private kernel `kolihhan/name-that-llm-phase15-terms-clean`, version `1`, and completed through GitHub Actions run `41276327535`. Full evidence is recorded in `artifacts/submission/SECOND_SUBMISSION.md`.

## Data / compliance invariants

Phase 15 started from the exact Phase 14 conservative public subset, then applied the provider-terms filter frozen in `TERMS_AUDIT.md` before any Phase 15 permanent-fold result was visible.

- Phase 14 external rows before terms filtering: **43,781**
- provider-terms-blocked/ambiguous external rows removed: **17,756**
- retained external rows: **26,025**
- retained external prompt IDs: **2,801**
- retained external labels: **11**
- blocked external labels remaining: **0**
- official competition rows: **2,699**, unchanged
- external validation rows in every fold: **0**
- OOF rows / unique IDs: **2,699 / 2,699**
- permanent fold fingerprint unchanged

Blocked external labels were:

- `gemini-3.1-pro-preview`
- `gemini-3.6-flash`
- `gemini-3.7-flash`
- `gemini-3.8-flash`
- `glm-5.3`
- `glm-5.3-flash`
- `kimi-k3`

## Model

The model recipe is exactly Phase 14's four-view sparse system with no changed model parameter:

1. char TF-IDF 3–5, max 250k, weight 1.0
2. word TF-IDF 1–3, max 180k, weight 0.7
3. shape-char TF-IDF 2–6, max 60k, weight 0.5
4. small/function-word TF-IDF 1–3, max 50k, weight 0.5

Classifier: `LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")`.

The only Phase 15 modeling change from Phase 14 was the independently justified external-data eligibility filter.

## Decision history

Before the leaderboard result, the frozen contract correctly produced `promote_phase15 = true` under the **local** promotion rule, and authorized one second Kaggle submission. That submission has now supplied the missing external anchor.

After observing the real Public score, the competition-level decision is:

`competition_promote_phase15 = false`.

The promoted competition system remains Phase 7 (`OOF 0.3700521998508576`, Public `0.369`).

Do **not** tune external-source membership, SVC C/class weights, view weights, ngram ranges, prompt filters, provider-label exclusions, or post-hoc overrides using the Phase 15 Public score. Any future submission requires a new pre-specified hypothesis and stronger validation evidence that addresses the observed transfer failure.

## Evidence

- provider-terms decision: `artifacts/phase15/TERMS_AUDIT.md`
- audit-only result: `artifacts/phase15/terms_clean_audit.json`
- frozen contract: `artifacts/phase15/CONTRACT.md`
- machine-readable local result: `artifacts/phase15/terms_clean_sparse.json`
- successful frozen workflow: `41270900644`
- frozen workflow artifact: `13069183594`
- frozen artifact digest: `sha256:d4462b2bc0b8d6e4029766ae60f8bda410cbb60bdb8d4b5d511324fbdf08dba2`
- second submission record: `artifacts/submission/SECOND_SUBMISSION.md`
- second submission workflow: `41276327535`
- second submission ref: `57861337`
- second submission Public score: `0.357`
