# Two-stage LoRA24 v1 Submission Preflight Design

Date: 2026-11-02
Status: **FROZEN FOR IMPLEMENTATION — DOES NOT AUTHORIZE SUBMISSION**

## Intent

Turn the already-passed `two-stage-lora24-v1` candidate into a reproducible Kaggle **submission candidate** without changing its scientific identity. The deliverable is an offline final-fit/inference bundle plus preflight evidence; it must not submit to Kaggle.

## Goals

- initialize from the exact locked Stage-1 A1 checkpoint;
- fit Stage 2 on all 2,699 official training rows for exactly two epochs using the frozen Gate-B optimization/objective;
- infer the competition test with the **plain head only**;
- emit exactly `submission.parquet` with columns `id, model` plus machine-readable lineage evidence;
- make the Kaggle notebook runnable with internet disabled;
- fail closed on model/checkpoint/runtime/data drift.

## Non-goals / anti-goals

- no centered-head blend;
- no Phase-14 SVC ensemble or router;
- no new seed/backbone/data/filter/calibration;
- no Public-LB-driven change;
- no GitHub Actions execution;
- no Kaggle competition submission;
- no final-fit checkpoint promotion as a new scientific candidate.

## Frozen scientific identity

Use the candidate contract unchanged:

- base: `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`;
- first 24 layers;
- exact Stage-1 checkpoint dataset `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`, version `1`, manifest `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`;
- Stage 2: all 2,699 official rows, two epochs, LoRA LR `5e-5`, head/norm LR `5e-4`, weight decay `0.01`, 10% warmup cosine, gradient clip `1.0`, seed `20261101`;
- same prompt-triplet objective and tokenization/pooling as Gate B;
- final prediction: plain head only.

The final fit is the deployment analogue of one Gate-B fold: the fit partition becomes all official training rows. Test rows may be read only after training state is finalized and must never affect training, class weights, scheduling, calibration, or model selection.

## Offline Kaggle architecture

A code-competition submission notebook must run with internet disabled. Gate B downloaded the Hugging Face base model at runtime, so submission needs locked offline inputs.

Use three immutable/private inputs:

1. **Stage-1 checkpoint** — existing locked Kaggle dataset version 1.
2. **Exact Qwen base snapshot** — one private Kaggle dataset containing only the exact Hugging Face revision plus a deterministic per-file SHA-256 manifest and lineage JSON.
3. **Offline runtime/wheelhouse** — one private Kaggle dataset containing the pinned Python wheels required to reproduce the Gate-B runtime (`transformers==4.57.1`, `peft==0.17.1`, `safetensors` and required transitive wheels) plus runtime-lineage metadata.

The repository implementation must provide local/operator materialization helpers for (2) and (3), but this design does not claim those datasets exist until their bytes are uploaded and re-downloaded/verified.

## Components

### `scripts/two_stage_lora24_submission.py`

Pure submission/final-fit helpers and the GPU entry point. It reuses Gate-B checkpoint loading and training primitives rather than cloning the scientific implementation.

Responsibilities:

- validate official train/test schema and row invariants;
- construct the full-fit index and fit-only class weights from all official train rows;
- execute the existing exact Stage-2 two-epoch trainer;
- perform **plain-head-only** batched test inference after training;
- build/validate `submission.parquet`;
- write `submission_lineage.json` containing exact model/checkpoint/runtime/data/training identity and explicit contamination flags.

### `scripts/build_two_stage_lora24_kaggle_kernel.py`

Build a private GPU notebook bundle with:

- `enable_internet=false`;
- `competition_sources=["name-that-llm"]`;
- dataset inputs restricted to the exact Stage-1 checkpoint, exact base snapshot, and exact offline runtime/wheelhouse;
- offline pip installation only (`--no-index` / local wheels);
- no Kaggle submit command.

### `scripts/materialize_two_stage_lora24_offline_inputs.py`

Operator-facing, fail-closed materialization helper. It may run only on a machine with Hugging Face/Kaggle access. It:

- downloads the exact base revision;
- creates deterministic file manifests and lineage records;
- downloads pinned wheels to a wheelhouse;
- creates private Kaggle datasets only if the fixed refs do not already exist;
- refuses silent overwrite/versioning.

It never includes competition train/test data.

### Promotion record

`artifacts/candidates/two-stage-lora24-v1/SUBMISSION_PROMOTION.md` records preflight state. Initial state is `BLOCKED` until exact offline inputs are materialized and an internet-disabled Kaggle notebook execution produces a schema-valid submission with verified lineage. Only a later explicit record may set `AUTHORIZED`.

## Approaches considered

1. **Download Qwen from Hugging Face inside the submission notebook.** Rejected: competition submission execution requires internet disabled.
2. **Substitute an available Kaggle `Qwen3-8B` asset.** Rejected: it is not proven byte/model identical to the frozen `Qwen3-8B-Base` revision.
3. **Lock the exact HF base + wheelhouse as private Kaggle inputs.** Chosen: preserves scientific identity and supports offline reproducibility.

## Verification

- TDD for submission-frame validation, final-fit input boundaries, deterministic manifest generation, kernel metadata/source contract, and materialization refusal semantics.
- focused tests run locally without GPU/network;
- broader available local tests run after focused green;
- actual Kaggle preflight remains **not verified** until the offline inputs are materialized and the notebook completes on Kaggle.

## Submission authorization boundary

This implementation and its tests do **not** authorize a Kaggle submission. Authorization requires a separate verified preflight result showing:

- exact offline base/checkpoint/runtime lineage;
- final fit completed with the frozen two-epoch configuration;
- competition test excluded from training/model selection;
- plain-head-only predictions;
- output schema/row count/labels valid;
- notebook internet disabled;
- no result-driven scientific change since the v1 contract.
