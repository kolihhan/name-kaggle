# DCT Temporal Fingerprint V3 — Design

Date: 2026-11-03 MYT

## Intent

Test one materially different hypothesis after `prompt-invariant-contrastive-v2` failed its frozen transfer gate: the v1 representation loses useful generator evidence because arithmetic mean pooling collapses the token trajectory.

The candidate is inspired by READER's public temporal fingerprint implementation, which uses a length-normalized DCT-II over response hidden states with modes `[0, 1]`. Mode 0 is exactly the arithmetic token mean; mode 1 captures the first non-constant temporal component. This makes the change conservative: preserve v1's mean signal and add one fixed temporal component rather than learning attention weights or selecting tokens post hoc.

## Hypothesis

For model-authorship attribution, generator-specific evidence is not uniformly distributed through a response. Adding the first length-normalized DCT-II AC component to each of v1's pooled hidden layers will improve source-shift attribution beyond v1 mean pooling without changing training data, backbone, LoRA, optimization, or objective.

## Scientific variable

Control representation, v1:

`concat(LN_l(mean(tokens_l)))` for pooled layers `[2,4,6,8,12,16,20,24]`.

Treatment representation, v3:

`concat(LN_l(DC_l), LN_l(AC1_l))` for the same pooled layers, where:

- `DC_l` is length-normalized DCT-II mode 0 over the same non-padding token span used by v1 after excluding token position 0;
- `AC1_l` is length-normalized DCT-II mode 1 over that same span;
- the same layer-specific `LayerNorm` module is reused for both modes;
- mode count is fixed to `2`;
- no learned attention, top-k selection, token clipping threshold, extra pooling head, or mode sweep is allowed.

For a response span of length `N`, the fixed weights are:

- mode 0: `1/N`;
- mode 1: `sqrt(2)/N * cos(pi * (t + 0.5) / N)`.

If excluding position zero leaves no token, both modes are defined as zero so the edge case matches v1's masked-mean behavior.

The treatment doubles the final representation width from 8 hidden-state blocks to 16 blocks. This is part of the predeclared DCT representation and is not independently tuned.

## Frozen components

Keep exactly as v1 unless this document explicitly changes them:

- Stage-1 corpus: exact Phase14 conservative subset from `pool_v4.parquet`;
- pool SHA-256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`;
- `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`;
- first 24 transformer layers;
- pooled layers `[2,4,6,8,12,16,20,24]`;
- max length before EOS `447`;
- same-prompt triplet microbatches;
- triplet seed `20261101`;
- gradient accumulation `6`;
- plain + centered CE objectives;
- class weighting;
- LoRA rank/alpha/dropout/targets;
- head dropout;
- optimizer, learning rates, weight decay, warmup, clipping;
- Stage-1 exactly one epoch.

No contrastive loss from rejected v2 is carried forward.

## Anti-overfit transfer gate

Before any v3 model score is computed, freeze a new whole-source holdout from the exact v1 public pool.

Permanently exclude already consumed source-holdout evidence:

- v1 Gate-A: `abdullah`, `semenoff`, `zakir`;
- rejected v2: `jadkutbi`, `lucasboesen`, `maniceet`.

Selection uses only source names, row/label coverage and prompt-leakage checks. It may not use any model prediction or score.

Frozen selection salt:

`ntl-dct-temporal-v3-transfer-holdout-v1`

Selection procedure:

1. enumerate all 3-source combinations from non-excluded sources;
2. for each combination, remove from training every row sharing a heldout `prompt_id` or `prompt_key`;
3. require at least 8 eligible labels, at least 300 eligible heldout rows, positive eligible rows from every heldout source, and zero prompt-id/key overlap;
4. among valid combinations, choose the combination with the smallest SHA-256 digest of `salt|source1|source2|source3` after sorting source names.

Once chosen, commit the exact split lock before training.

## Paired transfer experiment

Train two Stage-1 A1 systems from the same base, split, seed, triplet order and frozen v1 settings:

- control: exact v1 mean representation;
- treatment: v3 DC+AC1 representation.

Score both on the exact frozen new whole-source holdout.

### PASS gate — all required

1. v3 pooled balanced accuracy > control;
2. 5,000-resample prompt-group bootstrap, seed `20261103`, lower 95% CI of `v3-control` > 0;
3. mean heldout-source BA delta > 0;
4. at most one heldout source has negative BA delta;
5. for every label with at least 20 heldout rows, v3 recall may not fall by more than `0.15` vs control.

No rule may change after scores are visible.

### FAIL behavior

If the paired transfer gate fails:

- reject this DCT candidate;
- do not change mode count, mode weighting, layer set, source set, thresholds or pooling formula using the observed result;
- do not run official OOF to rescue it;
- do not submit it to Kaggle.

## Official OOF gate if transfer PASSes

Only after transfer PASS:

- train v3 Stage-1 under the original v1 Stage-1 lineage;
- run the existing permanent prompt-grouped five-fold Stage-2 OOF with Stage-2 training settings unchanged;
- compare primary plain-head v3 against v1 `0.5467611235396471` on the same official rows;
- require v3 > v1, at least 3/5 fold wins, a 5,000-resample prompt-group bootstrap with seed `20261101` whose lower 95% CI is > 0, and the exact v1 class-regression veto.

Official OOF is adaptive comparison evidence, not fresh hidden evidence.

## Deployment rule

Only if both source-transfer and official OOF gates PASS:

1. perform one frozen full-data final fit;
2. use the plain head as primary submission output;
3. run internet-disabled Kaggle preflight;
4. make at most one v3 competition submission;
5. record Public LB as external evidence only and do not tune toward it.

## Stop list

Do not combine this candidate with:

- contrastive v2;
- family routing;
- learned attention pooling;
- top-k token selection;
- token likelihood clipping;
- SVC/Gaussian/ensemble logic;
- a second backbone;
- new official split salts;
- Public-LB-driven changes.

Those are separate future hypotheses.