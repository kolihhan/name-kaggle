# Phase 17 — frozen global prior balancing contract

Date: 2026-11-01

## Question

Can a label-free, test-time global class-balance transform add complementary signal to the canonical official-only Phase 5b strong-sparse anchor and exceed the current best verified local system, Phase 7?

This is a **single permanent-fold shot**. The transform and promotion rule are frozen before the OOF result is observed.

## Motivation

Public high-scoring Name That LLM systems use global distribution balancing / Sinkhorn-like assignment after per-row model scores. This mechanism is materially different from the failed Phase 10–16 routes: it adds no synthetic data and no pretrained checkpoint. It changes only the joint assignment of a batch of predictions.

The Phase 15 Public-LB regression is not used to tune this transform.

## Frozen base model

Use the exact Phase 5b official-only strong sparse anchor:

- char TF-IDF 1–5;
- word TF-IDF 1–2;
- boundary-char TF-IDF 3–5;
- punctuation-aware TF-IDF;
- deterministic style features, weight `0.2`;
- `LinearSVC(C=1.5, class_weight="balanced", dual=True, max_iter=15000)`;
- same per-fold seed rule already used by `scripts/run_strong_sparse_anchor.py`.

No external rows are permitted.

## Frozen validation

Use the existing permanent validation unchanged:

- official rows: **2,699**;
- prompt groups: **150**;
- labels: **18**;
- 5-fold `GroupKFold(prompt_id)`;
- fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`;
- metric: balanced accuracy.

For each fold, fit Phase 5b only on the other four official folds and collect the untouched validation fold's full 18-class `LinearSVC.decision_function` margins.

After all 2,699 rows have exactly one OOF margin vector, apply the frozen global transform once to the complete OOF batch. The transform may use only the margin matrix shape/values; it must not inspect labels, prompt IDs, fold IDs, correctness, or per-class recall.

## Frozen transform

Let `M` be the `N x K` margin matrix, with `N=2699` and `K=18`, columns in the exact sorted official-label order.

### Step 1 — row softmax

Use temperature exactly `1.0`:

```python
Z = M - M.max(axis=1, keepdims=True)
P = np.exp(Z)
P /= P.sum(axis=1, keepdims=True)
```

No calibration, Platt scaling, temperature search, clipping, blending, or score normalization is allowed.

### Step 2 — uniform iterative proportional fitting

Set the target column mass to exactly `N / K` for every class.

Run exactly **100 iterations**:

```python
target = np.full(K, N / K, dtype=np.float64)
for _ in range(100):
    P *= target / P.sum(axis=0)
    P /= P.sum(axis=1, keepdims=True)
```

No early stopping and no tuned class prior.

### Step 3 — prediction

Final prediction is the label at `argmax(P, axis=1)`.

Tie behavior is NumPy's standard first-maximum behavior in the frozen sorted-label column order.

## Required invariants

Before scoring, assert:

1. margin matrix shape is exactly `(2699, 18)`;
2. every row has exactly one OOF margin vector;
3. all margins and transformed values are finite;
4. every transformed row sums to 1 within numerical tolerance;
5. raw `argmax` over the OOF margins exactly reproduces the canonical Phase 5b anchor score within `1e-12`;
6. the Sinkhorn transform receives no true-label, prompt, fold, or leaderboard information;
7. no external training rows are used.

Report:

- raw Phase 5b OOF BA;
- Phase 17 OOF BA;
- delta vs Phase 5b;
- delta vs Phase 7;
- fold BAs after the global transform;
- all 18 per-model recalls before and after;
- predicted class counts before and after;
- final transformed column masses.

## Promotion rule

Promote Phase 17 only if all invariants pass and:

`Phase17_BA > 0.3700521998508576 + 1e-12`

The Phase 7 score is the promotion threshold because it remains the current best verified local system and best Kaggle Public-LB system (`0.369`).

A numeric tie or improvement over Phase 5b that does not exceed Phase 7 is a rejection.

## One-shot consequence

After seeing the result, do **not** tune on these permanent folds:

- softmax temperature;
- number of Sinkhorn iterations;
- target class prior;
- score clipping/scaling;
- per-class multipliers;
- blend weight with raw predictions;
- fold-specific or label-specific exceptions.

A rejected result closes this exact post-processing mechanism as tested. A later experiment must introduce an independently justified new mechanism.

## Kaggle rule

This contract authorizes only local permanent-fold evaluation. It does **not** authorize another Kaggle submission.
