# V1 Mechanism Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce post-hoc v1 mechanism diagnostics from immutable completed OOF artifacts without creating a new model-selection loop.

**Architecture:** A focused CPU script loads official train, v1 Gate-B OOF, and Phase14 sparse OOF, validates row alignment, computes fixed descriptive diagnostics, and writes JSON plus Markdown. A branch-scoped public Actions workflow downloads the exact artifact IDs and archives the report.

**Tech Stack:** Python 3.11, pandas, scikit-learn, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-03-v1-mechanism-audit-design.md`

## Global Constraints

- Post-hoc diagnostic only; no model fitting or candidate promotion.
- Immutable artifact IDs: Gate B `13125872633`; Phase14 `13067907516`.
- No Public-LB tuning, alternate folds, routing search, blend search, or submission.
- Do not change the frozen v3 contract or queued v3 scientific run.

## Review Focus

- Reject duplicate/missing/misaligned OOF IDs rather than silently joining rows.
- Keep same-family accounting tied to true/predicted model families.
- Empty agreement-pattern cells must report zero rows safely.
- Within-model length diagnostic must not compare different label mixes.
- Report generation must be deterministic for identical input files.

---

### Task 1: Mechanism analysis core

**Files:**
- Create: `scripts/analyze_v1_mechanism.py`
- Test: `tests/test_analyze_v1_mechanism.py`

**Interfaces:**
- Consumes: train columns `id,text,model`; Gate-B columns `id,model,plain_prediction,centered_prediction`; sparse columns `id,model,prediction`.
- Produces: `analyze_frames(train, gate_b, sparse) -> dict[str, object]` and CLI JSON/Markdown outputs.

- [ ] **Step 1: Write failing tests** for family mapping, alignment rejection, error-family counts, length diagnostics, rescue counts, agreement patterns, and deterministic top-confusion ordering.
- [ ] **Step 2: Run focused test and verify RED** with missing `scripts.analyze_v1_mechanism`.
- [ ] **Step 3: Implement minimal analysis script** with no model fitting or tunable thresholds.
- [ ] **Step 4: Run focused test and full relevant CPU suite; verify GREEN.**
- [ ] **Step 5: Commit.**

### Task 2: Public Actions evidence run

**Files:**
- Create: `.github/workflows/v1-mechanism-audit.yml`

**Interfaces:**
- Consumes: immutable Actions artifacts and Task 1 CLI.
- Produces: uploaded `v1-mechanism-audit` artifact containing JSON and Markdown reports.

- [ ] **Step 1: Add a branch-scoped workflow** for `research/v1-mechanism-audit-2026-11-03` using standard Ubuntu CPU.
- [ ] **Step 2: Download exact artifacts with `gh api`, unzip, run focused tests, run audit, and upload outputs.**
- [ ] **Step 3: Verify workflow success and inspect report values against immutable source metrics.**
- [ ] **Step 4: Commit resulting research summary without changing any candidate contract.**