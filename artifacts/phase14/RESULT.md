# Phase 14 — conservative public 4-view SVC result

Date: 2026-11-01

## Verdict

**PROMOTE Phase 14.**

The single frozen permanent-fold run completed successfully under the committed contract and materially exceeded the promoted Phase 7 system.

## Frozen comparison

- Phase 7 reference balanced accuracy: **0.3700521998508576**
- Phase 14 balanced accuracy: **0.4178100919711658**
- delta vs Phase 7: **+0.04775789212030823**
- fold scores: `0.409259 / 0.401852 / 0.416667 / 0.435185 / 0.425990`

This is not numerical equality or a marginal gain. The promotion gate passes.

## Data / leakage invariants

The run rebuilt the external subset mechanically from the pinned public raw pool:

- raw pool rows: **75,783**
- raw SHA-256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`
- retained external rows: **43,781**
- retained external prompt IDs: **3,736**
- retained labels: **18 / 18**
- official OOF rows: **2,699 / 2,699 unique IDs**
- external validation rows in every fold: **0**

The subset rule, feature recipe, view weights, and LinearSVC configuration were frozen before the permanent-fold result was visible.

## Model

Four sparse views are concatenated:

1. char TF-IDF 3–5, max 250k, weight 1.0
2. word TF-IDF 1–3, max 180k, weight 0.7
3. shape-char TF-IDF 2–6, max 60k, weight 0.5
4. small/function-word TF-IDF 1–3, max 50k, weight 0.5

Classifier: `LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")`.

## Per-model note

Phase 14 is substantially stronger overall, but not uniformly better on every class. In particular, `claude-fable-5` remains weak at **0.08725** recall. Promotion is based on the precommitted overall balanced-accuracy gate, not post-hoc per-class cherry-picking.

## Decision

`promote_phase14 = true`.

Phase 14 becomes the current best validated local system at **0.4178100919711658** OOF balanced accuracy.

The original Phase 14 contract authorized only local evaluation, so a second Kaggle submission still requires a separate submission preflight and an explicit inference package that rebuilds the same frozen subset/recipe without changing the model after seeing this result.

## Evidence

- frozen contract: `artifacts/phase14/CONTRACT.md`
- reconnaissance / provenance audit: `artifacts/phase14/RECON.md`
- machine-readable result: `artifacts/phase14/public_sparse.json`
- successful frozen workflow: `41267190987`
- workflow artifact: `13067907516`
- artifact digest: `sha256:3dbe664866f099a806d24daa937bf2d1ba0ddacc87108dc90625009ba39ae4f0`
