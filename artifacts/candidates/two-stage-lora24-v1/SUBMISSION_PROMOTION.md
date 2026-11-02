# Submission Promotion — `two-stage-lora24-v1`

Date: 2026-11-02 (MYT)

## Status

**BLOCKED — IMPLEMENTATION READY, EXTERNAL PREFLIGHT PENDING**

Kaggle competition submission is **NOT AUTHORIZED**.

Gate A and Gate B passed, and the direct final-fit/offline-kernel implementation is now locally verified. However, the exact offline base-model dataset and offline runtime/wheel dataset have not yet been materialized and re-verified on Kaggle, and the internet-disabled GPU preflight kernel has not yet completed. Code readiness is not equivalent to submission readiness.

## Frozen candidate being promoted

- candidate: `two-stage-lora24-v1`
- exact base: `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- exact Stage-1 checkpoint dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`, version `1`
- checkpoint manifest: `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`
- exact official train SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
- final Stage-2 fit: all `2699` official rows, exactly `2` epochs, frozen Gate-B optimizer/objective/seed
- competition prediction: **plain head only**
- centered head role: diagnostic only; not used for submission
- Public-LB tuning: forbidden

## Implementation

Implemented without changing Gate-B scientific primitives:

- `scripts/two_stage_lora24_submission.py`
  - validates the exact official-train SHA before CSV loading/training;
  - uses all 2,699 official rows as the final-fit partition;
  - reuses Gate-B `load_exact_stage1_checkpoint(...)` and `train_two_stage2_epochs(...)`;
  - tokenizes/reads test for inference only after the two-epoch training call returns;
  - predicts with the plain head only;
  - writes `submission.parquet` and `submission_lineage.json`.
- `scripts/materialize_two_stage_lora24_offline_inputs.py`
  - prepares the exact Hugging Face base revision as a byte-manifested private Kaggle dataset;
  - prepares a byte-manifested offline wheel/runtime dataset;
  - refuses silent overwrite/versioning of the fixed refs;
  - requires a flat dataset payload because the Kaggle CLI `datasets create` default directory mode skips nested directories.
- `scripts/build_two_stage_lora24_kaggle_kernel.py`
  - builds a private T4 script kernel;
  - `enable_internet=false`;
  - exact dataset inputs are checkpoint + base snapshot + offline runtime;
  - offline pip uses `--no-index` and local wheels only;
  - embeds the frozen Gate-B runtime modules plus the final-fit module;
  - contains no Kaggle competition-submit side effect.

## Fresh local verification

Focused tests after the final review fixes:

```text
15 passed in 0.06s
```

Covered behaviors include:

- official train/test schema and ID boundaries;
- exact official-train SHA fail-closed behavior;
- all-official-row final-fit boundary;
- plain-head-only submission frame and allowed labels;
- contamination/selection flags in lineage;
- deterministic byte manifests;
- fixed model/revision/dataset refs;
- refuse-existing dataset semantics;
- flat Kaggle payload enforcement;
- private/offline/T4 kernel metadata;
- local-wheel-only pip arguments;
- no submit side effect.

A real kernel bundle was then built using the actual runtime modules preserved from the successful Gate-B evidence, not test sentinel modules:

```text
kernel_main_bytes = 111202
kernel_main_under_1MB = true
python compile = PASS
metadata enable_internet = false
metadata machine_shape = NvidiaTeslaT4
metadata competition_sources = [name-that-llm]
metadata dataset_sources = [
  kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint,
  kolihhan/name-that-llm-qwen3-8b-base-49e3418f,
  kolihhan/name-that-llm-two-stage-lora24-v1-offline-runtime
]
has_no_submit_command = true
has_no_index = true
has_frozen_train_sha = true
```

The three new production scripts also pass `python -m py_compile`.

## Review fixes before promotion record

Two real preflight defects were found and fixed before this record:

1. **Official-train byte drift:** the first implementation only recorded the train SHA in lineage. It now rejects any `train.csv` whose SHA-256 differs from the frozen `40a22c...` digest before final fitting.
2. **Kaggle nested dataset payload risk:** the first materializer placed model/wheel files in subdirectories even though the Kaggle CLI create path defaults to skipping directories. Payloads are now flat-root and nested files are rejected before dataset creation; the kernel verifies the flat payload bytes against lineage manifests.

## Not yet verified

The following are **UNKNOWN / pending external preflight**, not implied by local tests:

- `kolihhan/name-that-llm-qwen3-8b-base-49e3418f` has not yet been created, downloaded back, and byte-verified on Kaggle;
- `kolihhan/name-that-llm-two-stage-lora24-v1-offline-runtime` has not yet been created, downloaded back, and byte-verified on Kaggle;
- no internet-disabled Kaggle GPU preflight has yet run the final full-data two-epoch Stage-2 fit;
- no hidden competition-test `submission.parquet` from this candidate has yet been verified;
- no submission version has been authorized or sent to the competition.

A current-head full repository pytest suite was not run in this takeover because the working container does not contain a full repository checkout. The new focused tests and real generated-kernel compile are verified; do not generalize them into a current-head full-suite claim.

## Promotion gate

Submission remains blocked until one clean external preflight establishes all of:

1. exact base dataset is ready version 1 and downloaded bytes match its frozen manifest;
2. exact runtime dataset is ready version 1 and downloaded wheel bytes match its frozen manifest;
3. existing Stage-1 checkpoint remains ready version 1 and exact locked manifest;
4. internet-disabled private Kaggle GPU kernel completes final fitting with the frozen two-epoch configuration;
5. `submission_lineage.json` reports the exact official train SHA, exact model/checkpoint/runtime identities, `primary_head=plain`, zero test rows used for training/model selection, and `public_lb_tuning=false`;
6. `submission.parquet` has exactly the competition test IDs once each, exactly columns `id, model`, no nulls, and only official labels;
7. no scientific code/config change occurs after this preflight result.

Only then may a separate explicit record change the state to `AUTHORIZED`.

## Next single action

**Materialize and byte-verify the two fixed private Kaggle offline datasets (exact Qwen base + offline runtime), then run exactly one internet-disabled submission-preflight kernel. Do not submit the result yet.**
