# prompt-invariant-contrastive-v2 — Frozen Candidate Contract

Status: **FROZEN BEFORE ANY V1-vs-V2 TRANSFER SCORE**  
Date: 2026-11-03 MYT

## Hypothesis

The v1 two-stage LoRA representation transfers because Stage 1 learns generator-specific fingerprint signal across many prompts. Its remaining errors are dominated by sibling/version confusion and are worse on short texts, suggesting that prompt/topic nuisance still consumes representation capacity.

V2 tests one mechanism only: add a prompt-invariant exact-model contrastive objective during **Stage-1 A1 LoRA training** so representations from the same generator but different prompts are pulled toward historical cross-prompt prototypes, while different generators remain separated.

This candidate is **not** an ensemble, hierarchy, new backbone, new pooling rule, new data source, threshold search, or Public-LB optimization.

## Scientific variable

V1 Stage-1 A1 loss:

`plain CE + centered CE`

V2 Stage-1 A1 loss:

`plain CE + centered CE + 1.0 * cross-prompt prototype CE`

Frozen contrastive constants:

- weight: `1.0`
- temperature: `0.2`
- prompt-bucket seed: `20261101`
- two deterministic prompt buckets from SHA-256 of `(seed, prompt_id)`
- anchor in bucket `b` may use prototypes only from bucket `1-b`
- exact model label is the positive class
- prototype state is stop-gradient cumulative mean
- prototype update occurs only after a successful optimizer step
- skipped AMP/GradScaler step => current representations are discarded from prototype memory
- no EMA, queue-size tuning, projection head, margin, hard-negative threshold, family routing, or extra backbone forward

A0 remains exactly the v1 frozen-backbone/head-only control. The contrastive objective is applied only to A1.

## Frozen v1 components that must not change

- Stage-1 corpus: exact Phase14 conservative subset from `pool_v4.parquet`
- raw pool SHA-256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`
- Qwen3-8B-Base revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- first 24 transformer layers
- pooled layers `[2,4,6,8,12,16,20,24]`
- same-prompt triplet microbatches
- triplet ordering seed `20261101`
- gradient accumulation `6` triplets
- plain + centered heads
- class weighting
- LoRA rank/alpha/dropout/targets
- optimizer, learning rates, weight decay, warmup, clipping
- Stage-1 exactly one epoch
- Stage-2 recipe exactly v1 if the candidate reaches official OOF

## Why no community corpus is the primary transfer oracle

Two candidate corpora were audited **before any model score was computed** and rejected as independent transfer evidence:

1. `arxitekton/ntl-claude-gen-v2` — uses competition train/test for dedup/8-gram filtering, train-prompt proximity filtering, and a train-only host-likeness judge.
2. `zeminglii/ntl-own-exact-route-generations` — its text generations do not send competition text to providers, but its prompt bank is downstream of `crtproduction/name-that-llm-open-pool`, whose construction/removal logs explicitly use official-train typography matching, train-prompt overlap/proximity, train-passage cosine filtering, and BA-based data-quality selection.

They may be useful data assets but are not treated as fresh hidden-distribution validators.

## Predeclared mechanism transfer check

The external transfer test therefore uses a **new whole-source holdout** inside the exact v1 public pool. This is not claimed to be fresh hidden-Kaggle evidence; it is a predeclared source-shift A/B mechanism test that does not consume official competition rows or Public LB.

The source set was selected deterministically **before any v1-v2 score** by SHA-256 ranking with salt:

`ntl-prompt-invariant-v2-transfer-holdout-v1`

The original v1 Gate-A sources were excluded from selection. Frozen holdout:

- `lucasboesen`
- `jadkutbi`
- `maniceet`

Frozen split evidence: `TRANSFER_SPLIT_LOCK.json`

Properties:

- train rows: `24,099`
- heldout rows: `10,745`
- eligible heldout rows: `10,723`
- eligible labels: `11`
- prompt-id overlap: `0`
- prompt-key overlap: `0`

Any training row sharing a heldout prompt ID/key is removed.

## Paired transfer experiment

Train two Stage-1 A1 systems from the same base revision, same split, same seed, same triplet order and same v1 hyperparameters:

- control: exact v1 A1 objective
- treatment: v2 A1 objective above

Score both on the exact frozen whole-source holdout.

### PASS gate — all required

1. pooled balanced accuracy: `v2 > v1`
2. 5,000-resample **prompt-group** bootstrap, seed `20261102`: lower 95% CI of `v2 - v1` is `> 0`
3. mean of the three source-level balanced-accuracy deltas is `> 0`
4. at most one of the three heldout sources has a negative BA delta
5. per-label safety veto: for any label with at least `20` heldout rows, v2 may not lose more than `0.15` recall versus v1

No rule may be changed after the paired result is visible.

### FAIL behavior

If the paired transfer gate fails:

- reject this v2 mechanism;
- do not tune temperature, weight, prototype scheme, source set, label subset, or threshold on the observed result;
- do not run official OOF merely to rescue the candidate;
- do not submit it to Kaggle.

A new experiment requires a materially different mechanism and a new contract.

## Official OOF gate if paired transfer PASSes

Only after paired transfer PASS:

1. train a v2 Stage-1 checkpoint under the original v1 Stage-1 split/lineage;
2. run the existing permanent prompt-grouped five-fold Gate-B Stage-2 OOF, with Stage 2 unchanged;
3. preserve row-level plain/centered predictions;
4. compare the primary **plain** head against v1 on the same rows;
5. treat this official OOF as consumed/adaptive comparison evidence, not fresh hidden proof.

All official-OOF promotion conditions are frozen before the paired transfer result is observed:

- v2 primary plain OOF BA must exceed v1 `0.5467611235396471`;
- v2 must win at least `3/5` folds against v1;
- 5,000-resample paired **prompt-group** bootstrap with seed `20261101` must have lower 95% CI for `v2-v1` greater than `0`;
- class-regression safety veto uses the exact v1 Gate-B rule: for any label where **v1 recall is at least `0.25`**, if **v1 recall minus v2 recall is greater than `0.15`**, the candidate is vetoed.

The safety thresholds and bootstrap seed above are not tunable after any v2 official OOF prediction is observed.

## Deployment / Kaggle rule

If and only if both the paired source-transfer gate and official OOF gate pass:

- perform one frozen full-data final fit;
- use the plain head as the primary submission head unless a future separately frozen candidate changes that rule;
- run internet-disabled Kaggle preflight;
- submit at most one competition-facing v2 entry;
- record Public LB as external transfer evidence only.

After observing Public LB, do not alter v2 weights, thresholds, data, routing, labels, or hyperparameters toward that score.

## Stop list

For this candidate, do not:

- sweep contrastive weight or temperature;
- select a different alternate holdout after seeing scores;
- use Arxitekton/Zeminglii as an independent promotion oracle;
- train an OOF router on v1 plain/centered/sparse errors;
- add family hard routing;
- change pooling/token evidence;
- add a second backbone;
- add SVC/Gaussian/ensemble logic;
- create new official split salts;
- use Public LB as model selection.

Those are separate hypotheses, not rescue knobs for this experiment.
