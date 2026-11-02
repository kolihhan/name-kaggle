# Public-Safe Repository Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make repository visibility safety explicit and prevent future agents from making this repository public while tracked Competition Data/history or Kaggle public-code-sharing obligations remain unresolved.

**Architecture:** Keep experiment/evaluation state unchanged. Add a repo-wide visibility gate, a local `AGENTS.md` entry point, and a fail-closed guard on the workflow that materializes official Competition Data.

**Tech Stack:** Markdown, GitHub Actions.

**Spec:** User instruction on 2026-11-02 to make the repo public-safe and record the constraint for other agents.

## Global Constraints

- Do not change folds, candidate selection, model configuration, Gate B contract, thresholds, or Kaggle submission policy.
- Treat official competition files as non-public during the competition unless the official rules unambiguously permit redistribution to non-participants.
- Do not make the repository public while official competition files remain in reachable Git history.
- Any public sharing of Competition Code during the competition must satisfy the applicable Kaggle public-sharing rule.

## Review Focus

- `data/train.csv`, `data/test.csv`, and `data/sample_submission.parquet` are currently tracked.
- Historical commits can remain reachable after deleting files from HEAD; history sanitation is a separate destructive step.
- `materialize-official-data.yml` must fail closed if repository visibility is public.
- Other agents must see the gate immediately.
- No scientific/evaluation files should change.

---

### Task 1: Record public visibility gate

**Files:**
- Create: `PUBLIC_VISIBILITY.md`
- Create: `AGENTS.md`
- Modify: `README.md`

- [ ] Record current status as `BLOCKED` with exact blockers and readiness checklist.
- [ ] Add repo-local instructions requiring agents to read the visibility gate before changing visibility or data handling.
- [ ] Surface the blocker near the top of README without changing experiment conclusions.
- [ ] Re-read the files and verify wording is consistent.

### Task 2: Fail-close official-data materialization

**Files:**
- Modify: `.github/workflows/materialize-official-data.yml`

- [ ] Add repository-private state to the job environment.
- [ ] Add an early check that exits before download/commit when the repository is not private.
- [ ] Preserve current behavior for a private repository.
- [ ] Re-read workflow and verify the guard precedes all data materialization.

### Task 3: Final safety verification

- [ ] Confirm no experiment/evaluation files changed.
- [ ] Confirm repository remains `BLOCKED` until tracked data/history sanitation and Kaggle public-code-sharing compliance are completed.
