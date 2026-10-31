# Phase 5 — Honest multiview style baseline contract

Date: 2026-10-31

## Question

Can a stronger CPU authorship baseline close the obvious gap left by the Phase 1 character-only model before more GPU work?

The candidate is fixed before this OOF run. This is not a hyperparameter sweep.

## Validation contract

- Official rows: **2,699**.
- Prompts: **150**.
- Labels: **18**.
- External rows: **0**.
- Split: permanent deterministic **5-fold `GroupKFold(prompt_id)`**.
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Official train SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- Metric: balanced accuracy.
- Every vocabulary / scaler is fit inside each training fold only.
- OOF denominator: official rows only.

## Sanity gate

First reproduce the frozen Phase 1 char baseline using its canonical configuration:

- char TF-IDF 1–5;
- `max_features=80,000`;
- `min_df=1`;
- `sublinear_tf=False`;
- balanced LinearSVC.

Target: **0.31929654** OOF balanced accuracy.

Abort the experiment if the reproduction differs by more than **0.002**.

## Fixed multiview candidate

Feature views:

1. case-preserving char TF-IDF 1–5, max 180k;
2. case-preserving word TF-IDF 1–2, max 150k;
3. boundary `char_wb` TF-IDF 3–5, max 100k;
4. word-or-punctuation TF-IDF 1–2, max 100k;
5. deterministic rich style / typography features, standardized and weighted by 0.2.

Style features include length/sentence statistics, lexical diversity, capitalization/digit/non-ASCII ratios, contractions, markdown/list structure, punctuation count/rate features, and function-word rates.

Classifier:

- `LinearSVC(C=1.5)`;
- `class_weight="balanced"`;
- fixed seed per fold.

## Required output

- `artifacts/phase5/honest_style_baseline.json`
- `artifacts/phase5/honest_style_oof.csv`

The report must include the Phase 1 reproduction delta, all five multiview fold scores, aggregate OOF balanced accuracy, and per-model recall.

## Decision rule

- If the multiview candidate materially beats **0.31930**, promote it as the new CPU anchor.
- Compare against public grouped-CV references only after our own frozen OOF is complete.
- Do not use Hungarian assignment or prompt-cluster matching in this phase; public evidence already shows that route can inflate local OOF without transferring to the real leaderboard.
- Do not add external data until the official-only anchor is established.
