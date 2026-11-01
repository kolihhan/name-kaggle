# two-stage-lora24-v1 — Gate A Run Authorization

Status: **ONE EXTERNAL-ONLY GATE A EXECUTION AUTHORIZED**

Date: 2026-11-01

This record is frozen before any `two-stage-lora24-v1` Gate A model metric has been observed.

## Authorized execution

Exactly one scientific execution of Gate A is authorized under the frozen candidate contract and model/runtime lock:

- candidate: `two-stage-lora24-v1`
- base model: `Qwen/Qwen3-8B-Base`
- pinned model revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- Stage-1 data: exact Phase 14 conservative public subset rebuilt from pinned `nishantdahal/ntl-handwriting-blend/pool_v4.parquet`
- source holdout: frozen by `CONTRACT.md`
- compared conditions: frozen `A0` and `A1` only
- execution surface: private Kaggle GPU kernel
- competition sources attached: none
- official competition rows scored: `0`
- official OOF authorized before Gate A result: **false**
- competition test rows used: `0`
- Public leaderboard tuning: **false**
- Kaggle submission authorized: **false**

The execution must use the source tree whose implementation passed the verification gates below. No model/data/threshold/training change is authorized by this record.

## Verification evidence before authorization

Source head verified before this authorization:

```text
9210b54a5d483881113e9e8a0a8bfec8b41f9ea0
```

Candidate-specific workflow:

- workflow run: `41308946179`
- frozen Gate A tests: `19 passed`
- heavy Kaggle job: skipped, so no candidate model metric was produced

Full repository verification:

- workflow run: `41308946182`
- result: `154 passed in 232.70s`
- failures: `0`

The final pre-run implementation also passed the adversarial checks recorded in `MODEL_LOCK.md`:

- exact one-epoch Stage-1 consumption; no triplet padding to fill the final accumulation group;
- pinned Qwen stock final normalization retained;
- first-token-masked pooling and triplet centering are tested primitives;
- exact A1 adapter + LayerNorm/head checkpoint is preserved by the run if a valid A1 condition completes;
- no competition source or submission command exists in the Gate A workflow.

## Result-consumption rule

A valid Gate A result is consumed exactly once.

- `PASS`: record the exact evidence and preserve the exact A1 checkpoint. Only then may a separate Gate B implementation plan be created. This does not authorize Gate B execution by itself.
- `FAIL`: close `two-stage-lora24-v1`; do not run official OOF.
- `INCONCLUSIVE`: close v1 as inconclusive; do not mint a new source split inside v1.

Do not rerun because the scientific result is inconvenient.

An infra-only rerun is admissible only if the attempted execution fails before producing a valid Gate A result. Any infra repair must preserve the frozen scientific contract.

## Still unauthorized

This record does **not** authorize:

- official competition OOF;
- Phase 19 shadow-fold reuse;
- Stage-2 training;
- hyperparameter/source/threshold retuning;
- Public-LB tuning;
- a Kaggle competition submission.
