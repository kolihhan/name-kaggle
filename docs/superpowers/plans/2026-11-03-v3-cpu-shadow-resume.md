# V3 CPU Shadow Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-shadow checkpoint/resume primitive that preserves an uninterrupted CPU training trajectory across process/job boundaries, including mid-gradient-accumulation state.

**Architecture:** Keep the frozen CUDA v3 experiment untouched. A small generic checkpoint helper serializes only trainable parameters plus optimizer, scheduler, RNG, partial gradients, progress, and immutable lineage. Real Qwen integration will consume this helper only after the resume semantics are verified independently.

**Tech Stack:** Python, PyTorch, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-v3-cpu-shadow-resume-design.md`

## Global Constraints

- CPU shadow cannot authorize v3 Gate B or submission.
- Do not serialize frozen Qwen base weights.
- Resume inside a six-triplet accumulation group must preserve partial gradients.
- Restore Python, NumPy, and Torch CPU RNG states.
- Reject lineage drift fail-closed.
- Do not implement parallel gradient reduction yet.

## Review Focus

- A checkpoint taken after one microbatch of a multi-microbatch optimizer group must resume bitwise-identically on CPU.
- Frozen parameters must not be serialized as trainable checkpoint payload.
- Changed lineage must fail before state is restored.
- RNG restoration must reproduce Python, NumPy, and Torch streams exactly.
- Unknown/mismatched trainable parameter sets must fail closed.

---

### Task 1: Resume kernel

**Files:**
- Create: `tests/test_cpu_shadow_resume.py`
- Create: `scripts/cpu_shadow_resume.py`

**Interfaces:**
- Produces: `save_shadow_checkpoint(...) -> None`
- Produces: `load_shadow_checkpoint(...) -> dict[str, Any]`

- [ ] Write failing tests for mid-accumulation exact resume, lineage rejection, RNG restore, and trainable-only persistence.
- [ ] Run focused pytest and verify failure because `scripts.cpu_shadow_resume` does not exist.
- [ ] Implement the minimal trainable-only checkpoint kernel.
- [ ] Run focused pytest and require all tests to pass.

### Task 2: Qwen CPU-shadow integration boundary

**Files:**
- Create later: `scripts/dct_v3_cpu_shadow_chunk.py`
- Test later: `tests/test_dct_v3_cpu_shadow_chunk.py`

**Interfaces:**
- Consumes: resume kernel from Task 1.
- Produces: one cleanly bounded CPU chunk that starts from initial state or a prior checkpoint and exits after a declared microbatch count/time budget.

- [ ] First freeze the CPU-specific numerical/runtime contract; do not silently reuse CUDA GradScaler claims.
- [ ] Test exact source split/model revision/training-spec lineage validation.
- [ ] Test progress cursor advancement across partial and completed optimizer groups.
- [ ] Test no official rows / Public-LB / submission authorization.
- [ ] Only then implement real Qwen CPU chunk execution.

### Task 3: Free Actions orchestration

**Files:**
- Create later: `.github/workflows/v3-cpu-shadow-chunk.yml`

**Interfaces:**
- Consumes: Task 2 chunk runner.
- Produces: one bounded job/run with checkpoint artifact handoff.

- [ ] Resolve `PUBLIC_VISIBILITY.md` authorization explicitly before enabling a new workflow.
- [ ] Do not auto-chain unlimited runs.
- [ ] Start with one manually bounded chunk and measure throughput/checkpoint size.
- [ ] Continue only if measured cost is operationally reasonable.

## Current execution boundary

Implement Task 1 now. Tasks 2–3 remain gated on measured CPU feasibility and repository public-visibility policy; this avoids constructing a huge runner chain before the one-triplet probe has produced timing evidence.