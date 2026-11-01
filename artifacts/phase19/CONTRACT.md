# Phase 19 — frozen historical shadow-validation diagnostic

Date: 2026-11-01

## Purpose

Determine whether the large Phase 15 uplift over Phase 7 was robust to prompt-to-fold assignment, or whether repeated use of one permanent `GroupKFold(prompt_id)` mapping has become too influential to remain the sole model-selection oracle.

This is **validation diagnosis**, not a new modeling phase.

The only systems compared are the already-frozen Phase 7 and Phase 15 recipes. No model, feature, external-source, routing, threshold, or hyperparameter choice may change after this contract.

## Why this diagnostic is necessary

The original permanent 5-fold split has supported many sequential decisions. Individual experiments were often frozen before their own result, but later hypotheses were designed after earlier results on the same mapping were visible. That creates an adaptive-validation risk at the project level even when each individual phase is one-shot.

Phase 15 is the critical failure case:

- Phase 7 permanent-fold OOF BA: `0.3700521998508576`
- Phase 7 Public LB: `0.369`
- Phase 15 permanent-fold OOF BA: `0.4053218990802883`
- Phase 15 Public LB: `0.357`
- permanent-fold Phase15 − Phase7 delta: `+0.03526969922943074`
- Public Phase15 − Phase7 delta: `-0.012`

Phase 18 separately found that Phase 15 external responses are materially closer to official-training responses than the Phase 7 external corpus under a common text-similarity audit. That supports a competing external-distribution explanation, so this diagnostic must not assume the collapse came from split reuse alone.

## Data invariants

Use the unchanged official training set:

- rows: **2,699**
- unique prompt IDs: **150**
- model labels: **18**
- every row from one `prompt_id` stays in exactly one validation fold within a split.

Use the exact existing external data for each frozen system:

- Phase 7: the repository's frozen 2,503-row GPT/Gemini external snapshot;
- Phase 15: the exact provider-terms-clean 26,025-row subset reconstructed under the Phase 15 contract.

No new external rows are admitted.

## Frozen shadow panel

Create exactly **3** additional 5-fold prompt-group assignments with these literal salts:

1. `ntl-shadow-v1-a`
2. `ntl-shadow-v1-b`
3. `ntl-shadow-v1-c`

For each salt independently:

1. take the 150 unique prompt IDs as strings;
2. compute `sha256(f"{salt}|{prompt_id}".encode()).hexdigest()`;
3. sort prompt IDs by `(hash, prompt_id)` ascending;
4. assign fold ID by sorted rank modulo 5;
5. map every official response row to its prompt's fold.

Every shadow fold must contain exactly **30 prompt IDs**.

No salt search, seed search, stratification search, or result-dependent reassignment is allowed.

## Frozen systems

### Phase 7

Run the exact frozen Phase 7 system:

- Phase 5b official-only strong sparse anchor;
- the same 2,503-row / 8-label external specialist snapshot;
- the same deterministic features and `LinearSVC(C=1.5)` recipe;
- the same structural gate: replace only when the anchor predicts one of the eight covered labels.

The recipe is retrained from scratch inside each shadow split, as ordinary OOF requires. Nothing is retuned.

### Phase 15

Run the exact frozen Phase 15 system:

- the same 26,025-row terms-clean external subset;
- the same four sparse views and weights;
- `LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")`;
- external rows training-only.

Again, retrain from scratch inside each shadow split with no tuning.

## Required outputs

For every shadow assignment report:

- Phase 7 OOF balanced accuracy;
- Phase 15 OOF balanced accuracy;
- paired `Phase15 − Phase7` delta;
- all 18 per-class recalls for both systems and their deltas;
- paired correctness counts (`both`, `Phase7 only`, `Phase15 only`, `neither`);
- one prompt-group bootstrap of the paired BA delta using exactly 5,000 replicates and seed `20261101`.

Across all three shadows report:

- mean / sample standard deviation / min / max for Phase 7 BA;
- the same for Phase 15 BA;
- mean / sample standard deviation / min / max for the paired delta;
- whether the delta is positive on all three shadows;
- which per-class recall gains/losses are stable versus split-sensitive.

The bootstrap is uncertainty description only. It must not define a new threshold or be used to tune either system.

## Interpretation frozen before results

- If Phase 15 beats Phase 7 on all three independent grouped assignments with a similar positive delta, **split-assignment fragility is not a sufficient explanation** for the Public-LB failure; external/hidden distribution shift becomes the stronger explanation.
- If Phase 15's advantage materially shrinks, disappears, or changes sign across shadows, the original permanent mapping was **not robust enough to rank the two systems by itself**.
- Both mechanisms may contribute.
- No outcome from this diagnostic proves hidden-test transfer because all shadows reuse the same 2,699 official rows.

## Critical consumption rule

These three shadow assignments are being used now to diagnose historical Phase 7 versus Phase 15 behavior.

**After their results are observed, they are spent diagnostic folds.**

They must not later be described as unseen holdouts or reused as the sole promotion oracle for a new candidate developed with knowledge of this diagnostic. Future candidate-selection policy must be frozen separately before that candidate's result is observed.

This replaces the earlier draft policy that reserved these exact shadows for future promotion. No Phase 19 shadow result had been observed when this contract was changed; the change is made to satisfy the higher-priority validation-diagnosis objective before another modeling phase.

## Prohibited behavior

After results are visible, do not:

- replace or add salts until Phase 15 wins or loses;
- tune either frozen recipe on a shadow;
- remove unfavorable shadow splits;
- change external-source membership;
- alter Phase 7 routing/gating;
- tune Phase 15 views, `C`, class weights, source weights, or label balance;
- use the Public leaderboard as a tuning target;
- authorize a Kaggle submission from this diagnostic.

## Artifacts

Persist or archive:

- `shadow_folds.csv`;
- `shadow_fold_manifest.json`;
- one result JSON and OOF CSV per shadow;
- a final Phase 19 result summary after all three runs complete.

## Current competition state

Until this diagnostic says otherwise about validation reliability, the competition champion remains Phase 7:

- OOF BA: `0.3700521998508576`
- Public LB: `0.369`
- submission ref: `57838845`

Phase 15 remains rejected as competition champion:

- OOF BA: `0.4053218990802883`
- Public LB: `0.357`
- submission ref: `57861337`
