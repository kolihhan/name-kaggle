# Phase 15 — provider-terms-clean public sparse result

Date: 2026-11-01

## Verdict

**PROMOTE Phase 15 and authorize a second Kaggle submission.**

The single frozen permanent-fold run passed every committed invariant and materially exceeded the last submission-eligible promoted system, Phase 7.

## Frozen comparison

| System | OOF balanced accuracy | Delta vs Phase 7 |
|---|---:|---:|
| Phase 7 | 0.3700521998508576 | — |
| Phase 14 diagnostic (terms-blocked) | 0.4178100919711658 | +0.04775789212030823 |
| **Phase 15 terms-clean** | **0.4053218990802883** | **+0.03526969922943074** |

Phase 15 is lower than the Phase 14 diagnostic by `-0.012488192890877492`, but Phase 14 is not submission-eligible under the repository's conservative provider-output-terms standard. The Phase 15 contract explicitly made Phase 7 the eligibility baseline before this result was observed.

Fold scores:

`0.416667 / 0.381481 / 0.392593 / 0.416667 / 0.419093`

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

The only Phase 15 change is the independently justified external-data eligibility filter.

## Decision

`promote_phase15 = true`.

Phase 15 becomes the current best **submission-eligible** validated system at **0.4053218990802883** OOF balanced accuracy.

A second Kaggle submission is authorized, provided the submission notebook:

1. rebuilds the same pinned raw public pool;
2. reproduces the exact 43,781-row Phase 14 conservative subset;
3. applies the exact Phase 15 seven-label exclusion to obtain 26,025 external rows;
4. uses the frozen Phase 14 feature/model recipe unchanged;
5. passes a fresh read-only Kaggle preflight confirming no unexpected duplicate submission state.

## Evidence

- provider-terms decision: `artifacts/phase15/TERMS_AUDIT.md`
- audit-only result: `artifacts/phase15/terms_clean_audit.json`
- frozen contract: `artifacts/phase15/CONTRACT.md`
- machine-readable result: `artifacts/phase15/terms_clean_sparse.json`
- successful frozen workflow: `41270900644`
- workflow artifact: `13069183594`
- artifact digest: `sha256:d4462b2bc0b8d6e4029766ae60f8bda410cbb60bdb8d4b5d511324fbdf08dba2`
