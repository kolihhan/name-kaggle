# Transfer Firewall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and verify one final unconsumed source-shift firewall without fitting or scoring any model.

**Architecture:** A small pure-data script receives the exact Phase14 conservative subset and the nine already-consumed source names. It reserves every other source, removes prompt-id/key overlap from the training side, and writes immutable accounting JSON. A public GitHub Actions workflow downloads the existing Phase14 artifact and produces the lock.

**Tech Stack:** Python, pandas, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-03-transfer-firewall-design.md`

## Global Constraints

- No model fitting or model predictions.
- No change to the frozen v3 contract or queued run `41580819667`.
- Reserve every unconsumed admissible Phase14 source; no alternate split search.
- Remove `prompt_id` and `prompt_key` overlap from training.
- `model_scores_used_for_selection=false` must be recorded.

## Review Focus

- Missing consumed source must fail closed.
- Shared prompt IDs/keys must be removed from the training side.
- Every unconsumed source must be reserved, not ranked.
- Eligible labels require at least 20 heldout rows and presence after training leakage removal.
- Result must remain diagnostic/selection-free even if coverage is inconvenient.

---

### Task 1: Firewall accounting primitive

**Files:**
- Create: `scripts/freeze_transfer_firewall.py`
- Create: `tests/test_freeze_transfer_firewall.py`

**Interfaces:**
- Consumes: Phase14 conservative subset columns `source,prompt_id,prompt_key,model,text`.
- Produces: `build_firewall_lock(pool, consumed_sources, min_label_rows=20) -> dict`.

- [ ] Write tests for all-unconsumed reservation, prompt-overlap removal, eligibility, and missing-source fail-closed behavior.
- [ ] Run focused test and confirm RED because the module does not exist.
- [ ] Implement the minimal pure-data function.
- [ ] Run focused test and confirm GREEN.

### Task 2: Immutable public Actions execution

**Files:**
- Create: `.github/workflows/freeze-transfer-firewall.yml`

**Interfaces:**
- Consumes: Phase14 artifact `13067907516` and Task 1 function.
- Produces: `transfer_firewall_lock.json` as a workflow artifact.

- [ ] Download artifact `13067907516` through the GitHub API.
- [ ] Run the focused test in Actions.
- [ ] Read `conservative_public_subset.parquet` and build the lock with the exact nine consumed sources.
- [ ] Assert zero prompt overlap and `model_scores_used_for_selection=false`.
- [ ] Upload the lock artifact.

### Task 3: Record verified policy state

**Files:**
- Create after successful run: `artifacts/research/TRANSFER_FIREWALL.md`
- Create after successful run: `artifacts/research/TRANSFER_FIREWALL_LOCK.json`

**Interfaces:**
- Consumes: exact successful Actions artifact.
- Produces: durable source-of-truth accounting and policy interpretation.

- [ ] Verify run conclusion and artifact digest.
- [ ] Record exact source inventory and row/label accounting without changing the selection.
- [ ] Recheck queued v3 run remains untouched.
