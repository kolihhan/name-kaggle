# Gate B run authorization — `two-stage-lora24-v1`

Date: 2026-11-02 (MYT)

This record authorizes **exactly one scientific Gate-B official OOF execution** for `two-stage-lora24-v1`, using the already-frozen candidate contract and the verified implementation identified below.

## Authorized source

- implementation commit: `eff43d97a1339f1deeea667b90b45269133dffbd`
- candidate contract: `artifacts/candidates/two-stage-lora24-v1/CONTRACT.md`
- implementation plan: `docs/superpowers/plans/2026-11-02-two-stage-lora24-gate-b.md`
- plan commit: `b39bb4b721441f016ecd18f8d6eb2deba6fc3011`

The authorization commit itself may add only this record / state documentation. It must not change model, data, fold, training, evaluation, or promotion logic.

## Exact Stage-1 checkpoint

Gate B must start every fold from the same exact Gate-A A1 checkpoint; a retrained equivalent is forbidden.

- source Actions run: `41333631967`
- source artifact ID: `13094069155`
- source artifact size: `116273873` bytes
- source artifact digest: `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`
- private Kaggle dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`
- dataset version: `1`
- exact 12-file checkpoint manifest SHA256: `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`
- deterministic pre-upload transport SHA256: `8f3c6dabe6a3d910b80b6acafce6155ae6350dafb28b1b987f7cdb7eff7bf32a`
- attach-only byte verification run: `41374175606`
- recovery evidence artifact: `13110507634`
- recovery evidence digest: `sha256:ed0666b1d214299d1194782d021fb476a6535ebd4ed0bb0cecfda30b774e62d0`

The recovery evidence verified dataset status `ready`, version `1`, identical checkpoint bytes, and no dataset recreate/version operation.

## Official OOF frame

- official train SHA256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
- permanent prompt-group fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`
- expected official rows: `2699`
- folds: exactly `0,1,2,3,4`
- for fold `f`: fit only `fold_id != f`; validate only `fold_id == f`
- zero prompt-group overlap is required for every fold
- no Stage-1 public rows may be mixed into Stage 2
- no competition-test rows may be used

## Frozen model / Stage-2 configuration

- model: `Qwen/Qwen3-8B-Base`
- model revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- runtime pins used by the scientific worker: `transformers==4.57.1`, `peft==0.17.1`
- first `24` Qwen layers
- pooled hidden layers: `[2,4,6,8,12,16,20,24]`
- truncate to `447` tokens before explicit final EOS
- right padding; first token masked from mean pooling
- LoRA configuration unchanged from Gate A: rank `16`, alpha `32`, dropout `0.05`, bias `none`, targets `q/k/v/o/gate/up/down`
- eight trainable pooled-layer LayerNorms
- plain and centered 18-class heads; head dropout `0.1`
- deterministic same-prompt triplets; no triplet padding
- class weights computed from each fold's official fit rows only
- exactly `2` Stage-2 epochs
- adapter LR `5e-5`
- heads/norms LR `5e-4`
- AdamW; weight decay `0.01`
- `10%` warm-up then cosine decay across both epochs
- gradient accumulation equivalent to up to `6` triplets per optimizer step
- gradient clipping `1.0`
- seed `20261101`
- no early stopping and no validation-driven epoch/model selection
- every fold must load a fresh copy of the exact Stage-1 checkpoint

## Frozen evaluation / decision

Primary prediction is **plain head only**. Centered-head OOF is diagnostic only and may not be blended into v1.

The candidate is compared against regenerated row-level Phase-7 OOF on the exact same official rows. Paired prompt-group bootstrap uses `5000` resamples with seed `20261101`.

Gate B passes only if all four predeclared conditions are true:

1. candidate overall balanced accuracy is greater than Phase 7;
2. paired prompt-group bootstrap 95% CI lower bound for candidate-minus-Phase7 BA is greater than `0`;
3. candidate beats Phase 7 on at least `3 / 5` folds;
4. the class-recall safety veto does not fire: any class with Phase-7 recall `>= 0.25` may not lose more than `0.15` absolute recall.

No compensating post-result mechanism, blend, calibration, threshold change, source change, fold change, or epoch change is authorized inside v1.

## Phase-7 reference preflight

A real CPU regeneration of the frozen Phase-7 row-level pipeline completed before this authorization:

- Actions run: `41378723918`
- evidence artifact: `13113150078`
- artifact digest: `sha256:bdd25f43b202c6bfbbbd74a99fa8dbbf815c56071d679c253c2b88f4d4171079`
- regenerated rows: `2699`
- regenerated overall BA: `0.3700521998508576`
- regenerated fold BAs: `[0.38333333333333336, 0.33888888888888885, 0.37222222222222223, 0.38703703703703707, 0.36883780332056193]`
- overall BA, all five fold BAs, and per-class recalls matched the frozen Phase-7 report within absolute tolerance `1e-12`

## Verification before authorization

- final implementation head: `eff43d97a1339f1deeea667b90b45269133dffbd`
- full repository verification: Actions run `41379434008`
- exact result: `204 passed in 233.47s (0:03:53)`
- earlier Gate-B focused workflow at implementation commit `bf765e438eb3e169033e2778dcc73be8b04c7b18`: run `41378046450`, contract tests succeeded and `kaggle-gpu` was skipped
- checkpoint contract workflow at the same implementation commit: run `41378046514`, contract tests succeeded; both materialization/recovery jobs were skipped
- authorization verification: Actions run `41379951873`, Gate-B contract test job succeeded and `kaggle-gpu` was skipped
- scientific-drift review from plan commit `b39bb4b721441f016ecd18f8d6eb2deba6fc3011` through the verified implementation found no change to the frozen backbone, LoRA configuration, representation, Stage-2 epochs/LRs, fold frame, seed, fit-only class weights, primary head, bootstrap settings, or four Gate-B PASS conditions

## One-shot execution boundary

This authorization permits one `[gate-b-run]` scientific execution that produces five fold-local Stage-2 fits and one complete official OOF bundle.

It does **not** authorize:

- Public-LB tuning;
- competition-test inference;
- a Kaggle competition submission;
- reuse of Phase19 shadow folds;
- post-result retuning or rescue;
- a second scientific Gate-B run because the result is disappointing.

A valid Gate-B result must be consumed exactly once and evaluated mechanically against the frozen rules above.

An infrastructure-only retry is permissible **only** if the failed attempt produced no valid scientific OOF result, and only if the repair leaves all scientific data, checkpoint bytes, model configuration, fold membership, seeds, training hyperparameters, evaluation logic, and PASS thresholds unchanged. Any valid scientific result closes the one-shot authorization even if downstream orchestration fails.

## First trigger attempt — infrastructure failure before scientific execution

The first trigger-only commit was `8f1199f08e45ce7ee7b616714048da76a05cc1c1`; Actions run `41380075106` reached the Gate-B GPU job but failed at `kaggle kernels push` before Kaggle accepted or created the kernel.

Exact failure:

- Kaggle `SaveKernel` returned HTTP `400 Bad Request`;
- error: `Your kernel title does not resolve to the specified id`;
- the failure happened before the poll/download/evidence steps;
- those downstream steps were skipped;
- no fold-local Stage-2 fit started;
- no candidate OOF row was produced;
- no valid scientific Gate-B result exists from this attempt.

Therefore this attempt does **not** consume the one-shot scientific result boundary and qualifies for the already-authorized infrastructure-only retry.

Root cause was metadata-only: the kernel id slug was `name-that-llm-two-stage-lora24-gate-b`, while the title slug resolved to `name-that-llm-lora24-gate-b-oof`.

The repair changed only the Kaggle kernel title to `Name That LLM Two-stage LoRA24 Gate B`, so its slug exactly matches the pre-existing kernel id. No scientific code, data, checkpoint bytes, folds, model configuration, hyperparameters, evaluation logic, or PASS thresholds changed.

Repair / verification lineage:

- failing regression test commit: `273431674d5a3d5e7f3406fba46ef74335780a62`; focused run `41380262020` reproduced exactly `1 failed, 36 passed` on the slug mismatch;
- metadata-only fix commit: `442cb2f84370a34da4171210519320c42fef0ba6`; focused run `41380385290` succeeded and `kaggle-gpu` was skipped;
- regression-guard verification head: `b0b78bf7301c0611aff18d220710486c3c653873`;
- focused Gate-B run `41381901207` succeeded and `kaggle-gpu` was skipped;
- full repository verification run `41381901250`: `205 passed in 251.01s (0:04:11)`;
- infra-failure record commit: `a2eb9582143903fe42dea3a25a730e1cab9955d4`; focused run `41382504229` succeeded and `kaggle-gpu` was skipped.

The next `[gate-b-run]` trigger is therefore an **infra-only retry of the same frozen scientific execution**, not a second scientific experiment.

## Infra-only retry trigger

This documentation-only change is the retry trigger. It does not modify scientific code, official data, folds, checkpoint bytes, model configuration, training hyperparameters, evaluation logic, or PASS thresholds. The retry must use the same locked checkpoint dataset version `1` and the same one-shot Gate-B OOF contract.
