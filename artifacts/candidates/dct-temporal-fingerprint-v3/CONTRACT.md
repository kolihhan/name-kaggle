# dct-temporal-fingerprint-v3 — Frozen Candidate Contract

Date frozen: 2026-11-03 MYT

**Status: FROZEN BEFORE ANY V1-VS-V3 MODEL SCORE.**

## Hypothesis

The v1 LoRA representation loses generator-specific information by reducing every selected hidden-state trajectory to one arithmetic token mean. Adding exactly the first non-constant length-normalized DCT-II component will preserve the v1 mean signal while exposing temporal response fingerprints that transfer across unseen source corpora.

External motivation is fixed before evaluation: READER uses length-normalized DCT-II modes `[0,1]` over response-token hidden states; its mode zero is the arithmetic token mean and mode one is the first temporal AC component.

## Only major scientific change

### Control — exact v1 Stage-1 A1

For pooled layers `[2,4,6,8,12,16,20,24]`:

`z_v1 = concat(LN_l(masked_mean(tokens_l)))`

### Treatment — v3

For the same token span and pooled layers:

`z_v3 = concat(LN_l(DC0_l), LN_l(AC1_l))`

where the fixed length-normalized DCT-II weights over an effective span of length `N` are:

- `DC0(t) = 1/N`
- `AC1(t) = sqrt(2)/N * cos(pi*(t+0.5)/N)`

Rules:

- modes are exactly `[0,1]`;
- token position zero remains excluded exactly as in v1;
- padding is excluded;
- if the effective span is empty, both modes are zero;
- the same layer-specific LayerNorm is reused for both modes;
- no attention pooling, top-k selection, learned mode weights, clipping threshold, mode sweep, extra backbone, family router, contrastive loss, or ensemble is allowed.

The doubled representation width is intrinsic to the predeclared DC+AC representation and is not separately tuned.

## Frozen v1 components

Everything not explicitly changed above is v1:

- exact public pool SHA-256 `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`;
- exact Phase14 conservative subset construction;
- `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`;
- first 24 transformer layers;
- max length before EOS `447`;
- same-prompt triplet microbatches;
- seed `20261101`;
- inverse-frequency class weights;
- plain CE + centered CE;
- LoRA r16 / alpha32 / dropout0.05 / same target modules;
- head dropout `0.1`;
- gradient accumulation 6 triplets;
- adapter LR `1e-4`, head LR `1e-3`;
- AdamW weight decay `0.01`, 10% warmup cosine schedule, grad clip `1.0`;
- Stage-1 exactly one epoch.

Rejected v2 prototype contrastive learning is absent.

## Frozen source-transfer split

Source lock: `artifacts/candidates/dct-temporal-fingerprint-v3/TRANSFER_SPLIT_LOCK.json`.

Selection happened without loading or scoring any model.

- salt: `ntl-dct-temporal-v3-transfer-holdout-v1`
- heldout sources: `alperen`, `hailey`, `willrice`
- heldout rows: `14,506`
- eligible heldout rows: `14,506`
- eligible labels: `17`
- train rows after heldout prompt-bank removal: `14,502`
- prompt-id overlap: `0`
- prompt-key overlap: `0`
- selection digest: `03edcd6d665d5d7992e777b1a83d88845260b06727bc8c7760e73355c0b0af29`

The selector excluded all previously consumed source-holdout evidence:

- v1 Gate-A: `abdullah`, `semenoff`, `zakir`
- rejected v2: `jadkutbi`, `lucasboesen`, `maniceet`

This is a new external source-shift mechanism gate inside the frozen public pool. It is supporting transfer evidence, not a hidden-Kaggle oracle.

## Paired transfer PASS gate

Run v1 control and v3 treatment from the same base model, frozen split, seed and triplet order.

All conditions are required:

1. v3 pooled balanced accuracy > v1;
2. 5,000 prompt-group bootstrap, seed `20261103`, lower 95% CI of `v3-v1` > 0;
3. mean of the three source-level BA deltas > 0;
4. at most one source-level BA delta is negative;
5. for every label with at least 20 heldout examples, v3 recall may not fall by more than `0.15` vs v1.

No threshold may change after scores are visible.

### Transfer FAIL

Reject v3 immediately. Do not tune DCT modes, mode weights, layers, source set, training parameters or label subsets. Do not run official OOF and do not submit to Kaggle.

## Official OOF gate — only after transfer PASS

Initialize Stage-2 from the exact preserved v3 Stage-1 checkpoint and keep the v1 Stage-2 recipe unchanged apart from representation/readout shape.

Require all:

1. primary plain OOF BA > v1 `0.5467611235396471`;
2. v3 wins at least `3 / 5` permanent folds;
3. 5,000-resample prompt-group bootstrap, seed `20261101`, lower 95% CI for `v3-v1` > 0;
4. v1 class-recall safety veto remains untriggered (`v1 recall >= 0.25` and drop > `0.15`).

Official OOF is adaptive evidence only.

## Deployment authorization

If and only if transfer and official OOF both PASS:

- final Stage-2 fit on all 2,699 official rows for exactly two epochs;
- plain head only for submission;
- no test rows for training/model selection;
- internet-disabled Kaggle execution;
- at most one v3 Kaggle submission;
- Public LB recorded once as external evidence and never used to retune v3.

## Permanent stop list

Do not rescue this candidate with:

- DCT mode-count sweeps;
- learned temporal attention;
- token top-k heuristics;
- ATLAS likelihood clipping;
- contrastive-v2;
- sparse/Gaussian blending;
- family routing;
- second backbone;
- new official split salts;
- Public-LB feedback.
