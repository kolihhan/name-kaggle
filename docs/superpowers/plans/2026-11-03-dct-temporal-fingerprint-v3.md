# DCT Temporal Fingerprint V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test and, only if frozen gates pass, deploy a two-mode DCT temporal representation as the sole scientific change from the submitted v1 anchor.

**Architecture:** Keep the v1 Qwen3-8B LoRA training pipeline intact. Add a reusable two-mode length-normalized DCT pooling helper, materialize a v3 worker from the v1 Stage-1 worker, and evaluate v1 vs v3 on a newly frozen whole-source holdout before any official OOF. Only a transfer PASS unlocks official OOF; only an OOF PASS unlocks final fit and one Kaggle submission.

**Tech Stack:** Python, PyTorch, pandas/numpy, pytest, Kaggle kernels, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-03-dct-temporal-fingerprint-v3-design.md`

## Global Constraints

- Do not carry rejected v2 contrastive code into v3.
- DCT modes are exactly `[0,1]`; no sweep.
- Pool/model/training settings remain frozen to v1 except representation width/pooling.
- New transfer sources are frozen before any score and exclude all v1/v2 consumed holdout sources.
- Public LB is never a tuning signal.
- A failed gate stops the candidate.

## Review Focus

- One-token/empty-effective-span behavior must match v1 masked mean (zero representation).
- DCT mode 0 must numerically equal v1 masked mean for ordinary examples.
- Padding must not affect either DCT mode.
- Source split must remove prompt-id and prompt-key leakage and never use scores.
- Recovery workflows must attach to existing Kaggle executions rather than accidentally retraining after orchestration failures.

---

### Task 1: Freeze the new transfer split

**Files:**
- Create: `scripts/dct_v3_transfer_split.py`
- Create: `tests/test_dct_v3_transfer_split.py`
- Create: `.github/workflows/v3-freeze-transfer-split.yml`
- Create after successful non-scoring run: `artifacts/candidates/dct-temporal-fingerprint-v3/TRANSFER_SPLIT_LOCK.json`

**Interfaces:**
- Consumes: Phase14 safe pool and v1 `select_eligible_labels`.
- Produces: `build_transfer_split(frame) -> dict[str, object]` with exact source list, row counts, eligible labels, and zero-overlap evidence.

- [ ] Write tests for deterministic selection, exclusion of six spent sources, zero prompt leakage, and minimum coverage.
- [ ] Implement combination ranking exactly as the design specifies.
- [ ] Run focused pytest and verify PASS.
- [ ] Run the Actions split-freeze workflow; it downloads exact `pool_v4.parquet`, verifies SHA, computes no predictions, and uploads the lock.
- [ ] Commit the returned immutable lock before any training.

### Task 2: Implement two-mode DCT pooling

**Files:**
- Create: `scripts/dct_temporal_pool.py`
- Create: `tests/test_dct_temporal_pool.py`

**Interfaces:**
- Produces: `masked_dct_pool(hidden, attention_mask, *, mask_first_token=True, n_modes=2)` returning `(batch, 2, hidden)`.

- [ ] Write tests proving mode 0 equals `masked_mean_pool`, padding invariance, constant-trajectory AC1≈0, and zero effective span -> zeros.
- [ ] Implement the exact length-normalized DCT-II equations from the design.
- [ ] Run focused tests and verify PASS.

### Task 3: Materialize the v3 Stage-1 worker

**Files:**
- Create: `scripts/materialize_dct_v3_worker.py`
- Create: `tests/test_materialize_dct_v3_worker.py`

**Interfaces:**
- Consumes: exact v1 `scripts/run_two_stage_lora24_gate_a_gpu.py` source.
- Produces: a worker whose only scientific change is representation pooling/head width.

- [ ] Write static contract tests: same data/split/triplet/optimizer/LoRA constants; no contrastive code; exact two DCT modes; shared LayerNorm per layer/mode; doubled representation width.
- [ ] Implement the smallest source transformation/materialization path.
- [ ] Compile materialized worker and run focused tests.

### Task 4: One paired source-transfer execution

**Files:**
- Create: `scripts/dct_v3_source_eval.py`
- Create: `scripts/build_dct_v3_transfer_kernel.py`
- Create: `.github/workflows/v3-paired-transfer.yml`
- Create after execution: `artifacts/candidates/dct-temporal-fingerprint-v3/RESULT.md`

**Interfaces:**
- Consumes: frozen transfer lock, exact v1 worker, materialized v3 worker, exact model/runtime/pool.
- Produces: v1/v3 row predictions plus frozen gate verdict.

- [ ] Implement the frozen 5,000 prompt-bootstrap and safety/source rules with seed `20261103`.
- [ ] Build a private internet-off T4 Kaggle kernel and verify metadata/source identities before push.
- [ ] Execute exactly one scientific paired run.
- [ ] If orchestration fails after push, attach/recover; do not push again.
- [ ] Record PASS/FAIL and evidence artifact IDs/digests.
- [ ] If FAIL, stop candidate permanently.

### Task 5: Official OOF only after transfer PASS

**Files:**
- Create/modify only after Task 4 PASS: v3 Gate-B worker/builder/workflow and `GATE_B_RESULT.md`.

**Interfaces:**
- Consumes: exact successful v3 Stage-1 lineage.
- Produces: five-fold official OOF plain/centered predictions and frozen promotion verdict.

- [ ] Preserve v1 Stage-2 settings exactly; only checkpoint/readout shape changes for DCT representation.
- [ ] Run all five folds once.
- [ ] Apply the predeclared v1 comparison/fold/bootstrap/class-veto gate.
- [ ] If FAIL, stop; no submission.

### Task 6: Final fit and deploy only after both gates PASS

**Files:**
- Create/modify only after Task 5 PASS: final-fit script/kernel/workflow and submission record.

**Interfaces:**
- Consumes: frozen v3 Stage-1 checkpoint and v1 Stage-2 final-fit settings.
- Produces: one verified `submission.parquet` and one Kaggle submission.

- [ ] Full-data Stage-2 exactly two epochs; plain head only for submission.
- [ ] Internet-off Kaggle preflight; verify lineage and output schema/hash.
- [ ] Submit once.
- [ ] Record Public score as external evidence without changing v3 afterward.