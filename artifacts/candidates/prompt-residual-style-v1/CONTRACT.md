# Candidate — prompt-residual style representation v1

Status: **FROZEN CONTRACT — NOT YET IMPLEMENTED**

Date: 2026-11-01

This is the first candidate governed by `EVALUATION_POLICY.md`. It is deliberately not numbered as a new phase yet.

## Question

Can a representation trained to isolate **model-specific style after removing prompt-content direction** transfer better than naïvely fitting a classifier on external text distributions?

The candidate is designed around the failure mode exposed by Phase 15: a large external corpus can improve classification across held-out official prompt groups while still failing on the hidden distribution. The proposed mechanism therefore uses external prompt groups to learn a small style subspace, then fits the competition classifier only on official fold-training rows.

## Why this is materially different

This is not:

- Phase 15's external sparse concatenation;
- Phase 4's end-to-end official-only ModernBERT fine-tune;
- Phase 3's untouched generic frozen embedding;
- a public final checkpoint that has already seen all official competition rows;
- a post-hoc ensemble or gate.

Phase 4 showed that direct end-to-end fine-tuning on 2,699 official rows performs poorly. Phase 16 showed that currently discovered public LoRA/DeBERTa final checkpoints are contaminated for honest OOF, but explicitly left open a new resource-bounded representation trained from an independently frozen contract.

## Data

### Official

Use the unchanged official 2,699-row training snapshot and permanent prompt-grouped fold mapping only after the external transfer gate passes.

Official validation rows never enter representation fitting.

### External representation data

Use exactly the existing Phase 15 provider-terms-clean reconstruction:

- rows: `26,025`;
- unique external prompt IDs: `2,801`;
- labels: 11;
- source lineage: Phase 14 conservative public-pool subset followed by the frozen Phase 15 provider-terms exclusions.

No new source, label, row weighting, or source weighting may be added after this contract.

## Frozen base encoder

Reuse the Phase 3 encoder configuration exactly:

- model: `answerdotai/ModernBERT-base`;
- encoder weights: frozen;
- max sequence length: 256 tokens;
- representation: attention-mask-aware mean of `last_hidden_state`;
- L2 normalize each pooled embedding.

The encoder is never fine-tuned in this candidate.

## Prompt-residual style subspace

Given frozen external embeddings `z_i`:

1. group external rows by `prompt_id`;
2. for each prompt group, compute the mean embedding across all retained rows in that group;
3. compute residual `r_i = z_i - mean_prompt(z)`;
4. L2-normalize non-zero residuals;
5. fit `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="svd")` on residuals using the 11 external model labels;
6. use the fitted LDA transform as the fixed style projection.

Rows whose prompt group contains fewer than two distinct retained model labels are excluded **only from fitting the style projection** because they cannot identify a within-prompt model contrast. They remain governed by the frozen external dataset and may not be selectively restored based on results.

No LDA solver, dimensionality, shrinkage, source subset, or preprocessing sweep is allowed.

The fitted style projection is then applied directly to each raw frozen ModernBERT embedding. Test-time prompt IDs are not required.

## Gate A — external source-transfer check

This gate runs before any new official-fold score is observed.

Purpose: test whether the style subspace captures model identity that transfers across public source/prompt-bank construction rather than merely making the external training distribution easier.

Use the existing external `source` and `prompt_id` metadata from the Phase 14 reconstruction.

For every Phase 14 seed source that, after the Phase 15 terms filter, has at least:

- 100 retained rows;
- 20 unique prompt IDs; and
- 3 model labels shared with the remaining training rows,

run one leave-source-out diagnostic:

1. define the held-out prompt set as every `prompt_id` appearing in that source;
2. remove **all rows from all sources** whose `prompt_id` belongs to that held-out set from the external training partition, preventing prompt-bank leakage through inherited sources;
3. fit the prompt-residual LDA projection only on the remaining external training partition;
4. compare two classifiers on the same held-out rows and shared-label scope:
   - raw frozen ModernBERT embeddings + `LinearSVC(class_weight="balanced", C=1.0)`;
   - prompt-residual LDA features + the same `LinearSVC`;
5. report balanced accuracy and per-label recall for both.

Aggregate across eligible held-out sources with an unweighted mean of source-level balanced accuracy.

### Gate A pass rule

Proceed to official OOF only if:

- mean source-level BA delta, `style_projection - raw_frozen_embedding`, is strictly positive; and
- the style projection is not worse on more than half of eligible held-out sources.

If fewer than two sources satisfy the frozen eligibility rule, Gate A is **inconclusive**, not passed, and the candidate must stop for review rather than inventing a replacement threshold/source split.

No source may be removed because its result is unfavorable.

## Gate B — official grouped OOF

Only after Gate A passes:

1. fit the final prompt-residual LDA projection once using all 26,025 frozen Phase 15 external rows that satisfy the projection-fit rule;
2. encode all official texts with the frozen ModernBERT encoder;
3. apply the fixed external-only style projection;
4. for each permanent official fold, fit `LinearSVC(class_weight="balanced", C=1.0)` only on the four official training folds;
5. score the untouched official validation prompt groups;
6. preserve all 2,699 OOF predictions and decision outputs.

Required reporting:

- overall balanced accuracy;
- five fold scores;
- all 18 per-class recalls;
- delta versus Phase 7 on every class;
- exact correctness overlap with Phase 7 (`both`, `Phase7 only`, `candidate only`, `neither`);
- candidate-vs-Phase7 disagreement rate.

No new shadow folds are created.

## Promotion rule

This v1 candidate is a **standalone representation test**, not an ensemble experiment.

It becomes a Kaggle submission candidate only if:

1. Gate A passes exactly as frozen;
2. Gate B OOF balanced accuracy is strictly greater than Phase 7's `0.3700521998508576`;
3. no implementation/provenance/leakage invariant fails;
4. the gain is not accompanied by evidence of a broken class-coverage mechanism that contradicts the stated style-transfer hypothesis.

If the standalone candidate does not beat Phase 7, do **not** search a blend weight or routing threshold. A complementary ensemble would require a separate candidate contract frozen from the archived OOF predictions.

## Stop conditions

Stop this candidate without rescue if any of the following occurs:

- Gate A fails or is inconclusive;
- the required external source/prompt metadata cannot be reconstructed exactly;
- the LDA projection cannot be fit deterministically under the frozen solver;
- official fold isolation is violated;
- Gate B does not beat Phase 7;
- a provider/provenance rule needed for the 26,025-row external set becomes invalid.

Do not respond with:

- LDA hyperparameter search;
- ModernBERT fine-tuning;
- source cherry-picking;
- new prompt split salts;
- threshold or ensemble tuning;
- Kaggle Public-LB tuning.

## Implementation boundary

No model run or Kaggle submission is authorized by this contract alone.

The next step is a small implementation plan and tests for:

- prompt-residual construction;
- source/prompt leakage exclusion in Gate A;
- deterministic LDA projection;
- fold-local official classifier isolation;
- required OOF/error-overlap outputs.
