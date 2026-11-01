# Candidate Contract — two-stage-lora24-v1

Status: **FROZEN BEFORE MODEL EXECUTION**

Date: 2026-11-01

This contract defines the smallest credible reproduction of the public two-stage LoRA24 mechanism under this repository's evaluation policy. No model result from this candidate has been observed when this contract is frozen.

## 1. Hypothesis

A Qwen3-8B attribution model adapted with LoRA on provenance-audited public passages can learn model-specific writing signal that transfers across public sources better than the same frozen backbone with only trainable readout heads.

If that external-source transfer gate passes, initializing every official fold from the same public-only Stage-1 checkpoint and then fine-tuning only on that fold's official training prompt groups should outperform the current Phase 7 sparse champion under honest prompt-grouped OOF.

This candidate is intentionally testing the **training mechanism**, not copying a public final checkpoint.

## 2. Hard safety boundaries

Before Gate A passes:

- official competition rows scored: `0`;
- official OOF runs: `0`;
- competition test rows used: `0`;
- Public LB used for tuning: `0`;
- Kaggle submissions authorized: **false**.

Public Stage-2/final checkpoints that were trained on all 2,699 official rows are forbidden for OOF.

Phase 19 shadow folds are forbidden as fresh holdouts.

No post-result change to source split, source eligibility, backbone, LoRA rank, pooled layers, learning rates, epochs, classifier/head structure, or pass thresholds is allowed within `v1`.

## 3. Stage-1 public data

Use the existing Phase 14 conservative public subset as the only Stage-1 corpus.

Frozen provenance:

- raw public pool: `nishantdahal/ntl-handwriting-blend`, `pool_v4.parquet`;
- raw SHA256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`;
- Phase 14 conservative subset: `43,781` rows;
- labels: `18`;
- safe prompt IDs: `3,736`;
- exact normalized answer overlap with official train: `0` under the existing Phase 14 audit;
- subset construction must reuse the existing `build_conservative_subset(..., validate_full_contract=True)` implementation unchanged.

Do **not** apply the later Phase 15 terms filter: it leaves only 11 represented labels and is not suitable for an all-18 Stage-1 attribution model.

Do **not** add the public notebook's host-style punctuation filters to v1. Those filters use official-train distribution statistics and would confound the external-transfer gate we are trying to test.

## 4. Frozen Stage-1 source holdout

The holdout is source-level, not row-level.

Let `S` be the sorted unique `source` values in the Phase 14 conservative subset.

Frozen salt:

```text
ntl-two-stage-lora24-v1-source-holdout
```

For every source `s`, compute:

```text
SHA256(salt + "|" + s)
```

Sort sources lexicographically by the full hexadecimal digest. The first **3** sources are held out; all remaining sources are candidate training sources.

This rule is fixed before inspecting model results and cannot be replaced because of inconvenient source identity, size, or score.

Leakage removal:

- after selecting the three held-out sources, remove from Stage-1 training every row whose `prompt_id` appears in the held-out partition;
- also remove every training row whose `prompt_key` appears in the held-out partition when `prompt_key` is available;
- held-out rows are never removed merely because they have no training counterpart.

Eligible held-out labels are labels that:

- occur in Stage-1 training after leakage removal; and
- have at least `20` examples in the union of the three held-out sources.

Gate A is **inconclusive** if fewer than `8` labels are eligible or fewer than `300` held-out rows remain across eligible labels. Do not mint another source split if this occurs.

## 5. Frozen backbone and representation

Backbone:

```text
Qwen3-8B-Base
```

Use only the first `24` transformer layers.

LoRA target modules:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Frozen LoRA configuration:

- rank `r = 16`;
- alpha `32`;
- dropout `0.05`;
- no bias adaptation.

Frozen pooled layers:

```text
[2, 4, 6, 8, 12, 16, 20, 24]
```

Representation:

1. tokenize text without adding model-specific special tokens beyond the explicit final EOS;
2. truncate to `447` tokens before appending EOS;
3. right-pad batches;
4. mask the first token out of mean pooling;
5. attention-mask mean-pool each selected hidden layer;
6. apply one trainable LayerNorm per selected layer;
7. concatenate the eight normalized pooled vectors.

Heads:

- `plain`: linear 18-class head on concatenated representation;
- `centered`: linear 18-class head on within-triplet centered representation.

A dropout of `0.1` is applied before both heads during training, matching the readable LoRA24 reproduction.

## 6. Frozen triplet training objective

Training batches are groups of up to three passages sharing the same `prompt_id`, shuffled deterministically within prompt and across prompt groups.

Loss for one triplet/microbatch:

```text
CE(plain(drop(z)), y)
+
CE(centered(drop(z - mean_triplet(z))), y)
```

The Stage-1 public corpus is class-imbalanced, so both cross-entropy terms use the same class weights computed **only from the Stage-1 training partition** as inverse-frequency weights normalized to mean 1.

No held-out-source label frequencies may enter the weights.

## 7. Gate A conditions

Train exactly two Stage-1 conditions on the same public training partition and same deterministic triplet ordering.

### A0 — frozen-backbone control

- Qwen3-8B backbone frozen;
- no LoRA modules trained;
- train only the per-layer LayerNorms plus `plain` and `centered` heads;
- one epoch.

### A1 — LoRA adaptation

- same backbone truncation and representation;
- train LoRA modules plus LayerNorms and both heads;
- one epoch.

Frozen Stage-1 optimization for both conditions:

- AdamW;
- LoRA LR for A1: `1e-4`;
- heads/norms LR: `1e-3`;
- weight decay `0.01`;
- warm-up `10%` of optimizer steps;
- cosine decay after warm-up;
- gradient accumulation equivalent to `6` triplets per optimizer step;
- gradient clipping `1.0`;
- one epoch;
- deterministic seed `20261101`.

### Gate-A metric

Primary metric is **plain-head balanced accuracy** over eligible labels in held-out public sources.

Report:

- pooled held-out balanced accuracy for A0 and A1;
- A1 minus A0 delta;
- per-held-out-source balanced accuracy over labels eligible in that source;
- per-source A1 minus A0 delta;
- per-label recall for both conditions;
- number of train/held-out rows after leakage removal;
- prompt-id and prompt-key overlap counts, both required to be zero.

### Gate-A PASS

Gate A passes only if all are true:

1. eligibility is not inconclusive;
2. pooled held-out `BA(A1) > BA(A0)`;
3. mean of the three source-level deltas is `> 0`;
4. A1 is worse than A0 on at most **1 of 3** held-out sources.

Otherwise Gate A is `FAIL` or `INCONCLUSIVE` and this candidate stops before official OOF.

Do not rescue one favorable held-out source, change the hash split, change the pool, or retune Stage 1 after seeing Gate-A results.

## 8. Gate B — official fold-local Stage 2

Gate B is unauthorized until Gate A returns PASS.

If authorized, use the repository's existing permanent 5-fold prompt-group assignment exactly as the historical comparability frame.

For each fold `f`:

1. load the **same exact A1 Stage-1 checkpoint** produced before any official OOF;
2. official fit rows are only rows with `fold_id != f`;
3. official validation rows are only rows with `fold_id == f`;
4. no official validation row or prompt group may affect training, early stopping, class weights, calibration, or threshold selection;
5. do not mix public Stage-1 rows back into Stage 2.

Frozen Stage-2 training:

- two epochs;
- LoRA LR `5e-5`;
- heads/norms LR `5e-4`;
- weight decay `0.01`;
- 10% warm-up then cosine;
- same triplet objective;
- same max length / pooled layers / heads;
- seed `20261101` for v1.

There is no early stopping and no epoch selection from validation results in v1.

## 9. Gate-B evaluation

Primary candidate prediction is the **plain head only**.

Centered-head predictions are recorded as a secondary diagnostic and are not automatically blended into the primary score.

Compare primary OOF against the frozen Phase 7 OOF predictions on the same official rows.

Required report:

- candidate overall balanced accuracy;
- Phase 7 overall balanced accuracy;
- delta;
- five fold BAs for candidate and Phase 7;
- per-class recall and delta;
- candidate-vs-Phase7 disagreement rate;
- conditional accuracy when only candidate is correct / only Phase7 is correct;
- paired prompt-group bootstrap of candidate-minus-Phase7 balanced accuracy, `5,000` resamples, seed `20261101`.

### Gate-B PASS

Gate B passes only if:

1. candidate overall BA > Phase 7 BA;
2. paired prompt-group bootstrap 95% CI lower bound for delta is `> 0`;
3. candidate beats Phase 7 on at least `3 / 5` folds;
4. no class with Phase-7 recall `>= 0.25` loses more than `0.15` absolute recall without an explicitly documented compensating mechanism visible in the predeclared metrics.

Condition 4 is a safety veto, not a tuning rule. If it fires, v1 is not promoted; do not repair the class post hoc inside v1.

## 10. Centered-head follow-up rule

The centered head is not part of the v1 primary score.

After Gate B is evaluated, report its standalone OOF and its disagreement/complementarity with the plain head. A future blend or adaptive centering mechanism requires a **new candidate contract** frozen before its blend weight/cluster rule is evaluated.

No blend-weight search is allowed in v1.

## 11. Submission rule

Even if Gate A and Gate B pass, this contract does **not** automatically submit to Kaggle.

A separate promotion record must verify:

- data/checkpoint lineage;
- no validation contamination;
- Gate-A and Gate-B evidence;
- submission schema/inference reproducibility;
- no result-driven changes after this contract.

Only then may one new external Kaggle submission be explicitly authorized.

## 12. Stop list

For `two-stage-lora24-v1`, do not:

- start with 14B;
- ensemble multiple seeds/backbones;
- use public final Stage-2 checkpoints for OOF;
- add Sinkhorn, Hungarian assignment, family gates, or post-hoc calibration;
- add the public notebook's official-train-derived style filters;
- change the source holdout after seeing results;
- change Stage-1/Stage-2 epochs after seeing results;
- blend the centered head into the primary score;
- tune against Public LB;
- create a Kaggle submission before explicit promotion.

The candidate exists to answer one question first: **does public-source LoRA adaptation itself transfer better than a frozen Qwen attribution readout?**