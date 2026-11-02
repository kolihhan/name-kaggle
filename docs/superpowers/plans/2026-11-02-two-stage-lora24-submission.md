# Two-stage LoRA24 v1 Submission Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, offline, non-submitting final-fit/inference path for `two-stage-lora24-v1` that preserves the frozen candidate exactly.

**Architecture:** Reuse Gate-B checkpoint/training primitives for one full-data Stage-2 fit, add plain-head-only test inference and lineage generation, then build an internet-disabled Kaggle notebook that consumes immutable checkpoint/base/runtime inputs. Separate operator materialization code prepares exact offline base/runtime inputs without GitHub Actions.

**Tech Stack:** Python 3.11, pandas, numpy, PyTorch, transformers 4.57.1, peft 0.17.1, safetensors, huggingface_hub, Kaggle CLI/API, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-two-stage-lora24-submission-design.md`

## Global Constraints

- Preserve `two-stage-lora24-v1` frozen scientific config exactly.
- Final fit uses all 2,699 official train rows for exactly two Stage-2 epochs.
- Plain head only for competition-test prediction.
- Competition test never affects training, weighting, scheduling, calibration, or selection.
- Final Kaggle notebook has internet disabled.
- Do not use GitHub Actions.
- Do not submit to Kaggle.
- Offline base revision is exactly `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Existing Stage-1 checkpoint dataset remains version 1 and immutable.

## Review Focus

- Test frame accidentally entering Stage-2 training or class-weight computation.
- Centered-head predictions accidentally entering final submission.
- Offline notebook silently downloading from internet or pip index.
- Model/checkpoint/runtime dataset drift or silent version overwrite.
- Submission ID/row/label mismatch against competition test.

---

### Task 1: Final-fit and submission primitives

**Files:**
- Create: `scripts/two_stage_lora24_submission.py`
- Create: `tests/test_two_stage_lora24_submission.py`

**Interfaces:**
- Consumes: Gate-B constants and helpers from `scripts.run_two_stage_lora24_gate_b_gpu` and `scripts.run_two_stage_lora24_gate_b`.
- Produces: `validate_final_fit_frames(...)`, `build_submission_frame(...)`, `plain_test_batches(...)`, `build_submission_lineage(...)`, and GPU `run_final_fit(...)`.

- [ ] **Step 1: Write failing tests** for train/test schema, duplicate/null IDs, all-official fit boundary, plain-only output, label validation, and lineage contamination flags.
- [ ] **Step 2: Run focused tests and verify RED** because the submission module does not exist.
- [ ] **Step 3: Implement minimal pure helpers and GPU final-fit entry point** reusing the exact Gate-B Stage-2 trainer/checkpoint loader.
- [ ] **Step 4: Run focused tests and verify GREEN.**
- [ ] **Step 5: Run existing Gate-B focused tests** to detect scientific-contract regression.
- [ ] **Step 6: Commit with `[skip ci]`.**

### Task 2: Offline input materialization

**Files:**
- Create: `scripts/materialize_two_stage_lora24_offline_inputs.py`
- Create: `tests/test_two_stage_lora24_offline_inputs.py`

**Interfaces:**
- Consumes: frozen model/revision/version constants.
- Produces: deterministic manifest/lineage helpers and an operator CLI that creates fixed private Kaggle dataset refs while refusing existing refs.

- [ ] **Step 1: Write failing tests** for deterministic manifests, exact base identity, no competition-data packaging, fixed wheel pins, and refuse-existing semantics.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement minimal materialization helpers/CLI.**
- [ ] **Step 4: Verify GREEN without network/upload.**
- [ ] **Step 5: Commit with `[skip ci]`.**

### Task 3: Offline Kaggle kernel builder

**Files:**
- Create: `scripts/build_two_stage_lora24_kaggle_kernel.py`
- Create: `tests/test_two_stage_lora24_kaggle_kernel.py`

**Interfaces:**
- Consumes: final-fit module plus fixed offline dataset refs from Task 2.
- Produces: notebook and `kernel-metadata.json` bundle for a private GPU run; does not submit.

- [ ] **Step 1: Write failing tests** that execute the builder and assert private GPU metadata, internet disabled, exact three dataset inputs plus competition source, offline pip only, plain-head final-fit invocation, output path, and absence of a submit side effect.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement the smallest kernel builder.**
- [ ] **Step 4: Verify GREEN.**
- [ ] **Step 5: Commit with `[skip ci]`.**

### Task 4: Promotion/preflight record and final verification

**Files:**
- Create: `artifacts/candidates/two-stage-lora24-v1/SUBMISSION_PROMOTION.md`
- Modify: `NOW.md`

**Interfaces:**
- Consumes: tested code from Tasks 1–3.
- Produces: canonical status stating code preflight readiness vs external/Kaggle blockers.

- [ ] **Step 1: Run all new focused tests plus existing Gate-B contract tests available in the local checkout.**
- [ ] **Step 2: Run the broadest available local pytest suite and record any unavailable dependencies/files explicitly.**
- [ ] **Step 3: Self-review the implementation against the design and frozen candidate contract.**
- [ ] **Step 4: Write promotion record as `BLOCKED` until exact base/runtime datasets are materially uploaded and an internet-disabled Kaggle notebook run succeeds.**
- [ ] **Step 5: Update `NOW.md` with the verified implementation/preflight state and exactly one next action.**
- [ ] **Step 6: Re-fetch modified files/branch and verify no workflow run was triggered by `[skip ci]` commits.**
