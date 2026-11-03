# V3 Exact CPU Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an exact, auditable cross-job resume mechanism for frozen v3 CPU training, then wire a bounded CPU chunk runner and public-repo GitHub Actions orchestration without changing the scientific experiment.

**Architecture:** First prove a model-agnostic PyTorch checkpoint format that preserves partial gradient accumulation and RNG state. Then add a v3 chunk-planning layer that exposes deterministic cursor boundaries without scoring. Finally add an Actions workflow that chains state artifacts across bounded CPU jobs while leaving the existing GPU scientific run untouched.

**Tech Stack:** Python 3.11, PyTorch, pytest, GitHub Actions artifacts.

**Spec:** `docs/superpowers/specs/2026-11-03-v3-exact-cpu-resume-design.md`

## Global Constraints

- Candidate remains `dct-temporal-fingerprint-v3`.
- Gradient accumulation remains exactly 6 same-prompt microbatches per optimizer step.
- Persist partial gradients when stopping at accumulation positions 1..5.
- No official OOF, firewall prediction, Public-LB tuning, competition-test use, or submission.
- Do not cancel or replace queued GPU run `41580819667`.
- Do not implement six-way distributed gradient merging in this plan.

## Review Focus

- Mid-accumulation resume must preserve existing gradients rather than zeroing them.
- Restoring into a parameter set with missing/extra names or mismatched shapes must fail loudly.
- RNG continuation after restore must match uninterrupted execution.
- Cursor and accumulation position must remain consistent at optimizer-step boundaries.
- Workflow retries must not silently advance from a stale or wrong-lineage state artifact.

---

### Task 1: Exact resumable training-state primitive

**Files:**
- Create: `scripts/exact_cpu_resume_state.py`
- Create: `tests/test_exact_cpu_resume_state.py`

**Interfaces:**
- Produces: `capture_resume_state(...) -> dict`, `restore_resume_state(...) -> dict`, `save_resume_state(path, state)`, `load_resume_state(path) -> dict`.

- [ ] **Step 1: Write failing tests** covering mid-accumulation round-trip, RNG continuity, optimizer/scheduler restoration, gradient preservation, and lineage/shape rejection.
- [ ] **Step 2: Run** `PYTHONPATH=. pytest -q tests/test_exact_cpu_resume_state.py` and verify RED because the module does not exist.
- [ ] **Step 3: Implement minimal state capture/restore** using named trainable parameters and `torch.save`/`torch.load`; include Python/NumPy/Torch RNG, cursor, accumulation position, metadata and per-parameter gradients.
- [ ] **Step 4: Run focused tests** and require PASS.
- [ ] **Step 5: Commit** the primitive and tests.

### Task 2: Uninterrupted-vs-resumed equivalence harness

**Files:**
- Modify: `tests/test_exact_cpu_resume_state.py`
- Create: `scripts/exact_cpu_resume_equivalence.py`

**Interfaces:**
- Consumes: Task 1 state API.
- Produces: deterministic `run_equivalence_probe(...) -> dict` reporting bitwise/equality checks without competition data.

- [ ] **Step 1: Add failing test** that trains a tiny deterministic network across >1 optimizer step, interrupts after a non-boundary microbatch, reconstructs fresh objects, restores, and finishes.
- [ ] **Step 2: Verify RED** for missing equivalence harness.
- [ ] **Step 3: Implement probe** with fixed synthetic inputs, AdamW, scheduler, accumulation=6 and explicit cursor handling.
- [ ] **Step 4: Require exact final parameter, optimizer-state, scheduler-state, cursor, accumulation-position, and next-RNG equality.**
- [ ] **Step 5: Commit** equivalence proof.

### Task 3: Frozen v3 CPU chunk-state contract

**Files:**
- Create: `scripts/dct_v3_cpu_chunk.py`
- Create: `tests/test_dct_v3_cpu_chunk.py`

**Interfaces:**
- Consumes: Task 1 state API and frozen v3 training spec.
- Produces: `chunk_contract()`, `validate_chunk_manifest(payload)`, `next_chunk_boundary(...)`.

- [ ] **Step 1: Add failing tests** pinning candidate name, accumulation=6, zero-scoring scope, monotonic cursor, valid accumulation positions, and stale-lineage rejection.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement only deterministic chunk accounting/manifest validation first; do not run Qwen in unit tests.**
- [ ] **Step 4: Require PASS.**
- [ ] **Step 5: Commit** chunk contract.

### Task 4: Public CPU Actions orchestration

**Files:**
- Create: `.github/workflows/v3-exact-cpu-resume-static.yml`
- Create: `.github/workflows/v3-exact-cpu-chunk.yml`
- Modify: `NOW.md` only on the infra branch if needed to document status; do not merge into scientific v3 until verified.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: static CI plus manual bounded chunk workflow with artifact in/out lineage.

- [ ] **Step 1: Add static workflow** that installs CPU PyTorch and runs focused resume/chunk tests plus equivalence probe.
- [ ] **Step 2: Push and verify static workflow GREEN.**
- [ ] **Step 3: Add manual chunk workflow** with `workflow_dispatch`, bounded microbatch budget, predecessor artifact/run identifiers, strict manifest validation, and successor artifact upload.
- [ ] **Step 4: Add dry-run mode** that exercises state chaining on synthetic model only; no Qwen download.
- [ ] **Step 5: Verify dry-run artifact chaining in Actions.**

### Task 5: Real-Qwen feasibility gate before long CPU chain

**Files:**
- Modify: `scripts/dct_v3_cpu_chunk.py`
- Modify: `.github/workflows/v3-exact-cpu-chunk.yml`
- Test: `tests/test_dct_v3_cpu_chunk.py`

**Interfaces:**
- Consumes: exact resume state and frozen v3 worker semantics.
- Produces: one bounded real-Qwen chunk that saves resumable state without scoring.

- [ ] **Step 1: Implement a single real-Qwen microbatch path** using frozen 24-layer Qwen3-8B v3 settings and exact triplet order.
- [ ] **Step 2: Save state immediately after the microbatch, including partial gradients and cursor.**
- [ ] **Step 3: In a second Actions job, restore the artifact and execute the next microbatch; verify cursor increments and prior gradients are present before backward.**
- [ ] **Step 4: Only if both jobs complete below the hosted timeout, authorize a longer CPU chain. Otherwise record CPU as operationally infeasible without changing v3.**
- [ ] **Step 5: Commit evidence/status.**
