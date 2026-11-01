# Phase 19 — frozen shadow validation protocol

Date: 2026-11-01

## Purpose

The original permanent 5-fold `GroupKFold(prompt_id)` has supported many sequential model decisions. It remains useful for continuity, but future promotion should not depend on a single prompt-to-fold assignment.

Phase 19 freezes a **shadow validation panel** before another candidate model is evaluated.

This is validation infrastructure, not a model experiment. It uses no Kaggle leaderboard information to choose split assignments.

## Important limitation

Shadow splits are **not new data**. They measure sensitivity to prompt-to-fold assignment; they do not solve hidden-distribution shift.

External-data candidates must separately satisfy the Phase 18 transfer-risk gate.

## Data invariants

Use the unchanged official training set:

- rows: **2,699**;
- unique prompt IDs: **150**;
- model labels: **18**;
- every row from one `prompt_id` must remain in exactly one validation fold within a split.

## Frozen shadow panel

Create exactly **3** shadow 5-fold assignments, with these literal salts:

1. `ntl-shadow-v1-a`
2. `ntl-shadow-v1-b`
3. `ntl-shadow-v1-c`

For each salt independently:

1. take the 150 unique prompt IDs as strings;
2. compute `sha256(f"{salt}|{prompt_id}".encode()).hexdigest()`;
3. sort prompt IDs by `(hash, prompt_id)` ascending;
4. assign fold ID by sorted rank modulo 5;
5. map every official response row to its prompt's fold.

Because 150 prompt groups are distributed round-robin, every shadow fold must contain exactly **30 prompt IDs**.

No salt search, seed search, stratification search, or result-dependent reassignment is allowed.

## Frozen artifacts

Persist:

- `artifacts/phase19/shadow_folds.csv`
  - columns: `id`, `prompt_id`, `shadow_a`, `shadow_b`, `shadow_c`;
- `artifacts/phase19/shadow_fold_manifest.json`
  - row / prompt / label counts;
  - per-shadow prompt counts per fold;
  - SHA-256 of the canonical CSV bytes;
  - pairwise number of prompt IDs assigned to different folds between shadow splits;
  - comparison against the original permanent `fold_id` assignment.

The generator must be deterministic: repeated generation from the same official data must produce byte-identical CSV output.

## Future promotion rule

Phase 19 does not retroactively re-rank past phases. Phase 7 remains the competition champion until a future candidate is evaluated under a separately frozen experiment contract.

For a **new candidate**, its contract must state before evaluation how the shadow panel is used. Default policy:

1. candidate must beat its predeclared reference on the original permanent split;
2. candidate must show positive balanced-accuracy delta on **all 3** shadow assignments;
3. median shadow delta must be positive;
4. no shadow split may be discarded after seeing results;
5. Kaggle submission remains separately gated and is not automatically authorized by local robustness.

For expensive neural experiments, a contract may predeclare a cheaper shadow use, but it must do so before results are visible.

## Prohibited behavior

After generating the panel, do not:

- replace a salt because a baseline looks inconvenient;
- add more salts until a candidate passes;
- select only favorable shadow splits;
- tune model parameters separately on each shadow split;
- infer that shadow robustness guarantees hidden-test transfer.

## Current references

- Phase 7: local `0.3700521998508576`, Public LB `0.369`.
- Phase 15: local `0.4053218990802883`, Public LB `0.357`; rejected transfer.
- Phase 17: local `0.3667238379318916`; rejected, not submitted.
- Phase 18: external transfer-risk audit recorded in `artifacts/phase18/TRANSFER_AUDIT.md`.
