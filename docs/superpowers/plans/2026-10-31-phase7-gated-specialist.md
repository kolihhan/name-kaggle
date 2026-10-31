# Phase 7 Gated Specialist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate one untuned 8-class GPT/Gemini specialist that can refine only covered-label predictions from the frozen Phase 5b 18-class anchor.

**Architecture:** Re-run the canonical Phase 5b anchor to obtain aligned OOF predictions, then train an independent fold-local 8-class strong-sparse specialist on covered official rows plus the audited 2,503 external rows. The deterministic gate replaces the anchor label only when the anchor already predicts one of the 8 covered labels, which structurally preserves all uncovered-class recall.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy sparse, scikit-learn TF-IDF + LinearSVC, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-31-phase7-gated-specialist-design.md`

## Global Constraints

- Permanent validation: 5-fold `GroupKFold(prompt_id)`, fingerprint `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Canonical Phase 5b anchor pipeline and `LinearSVC(C=1.5, class_weight=balanced)` must remain unchanged.
- Covered labels are exactly the 8 labels in the audited external source.
- External source is exactly 2,503 rows / 430 prompt IDs and has passed the raw identity gate; its prompt bank is reported as 1,000 new prompts and the audit found zero exact normalized-text overlap with official train.
- No threshold, blend-weight, sample-count, class-weight, C, label-set, or gating sweep.
- Score only official OOF validation rows.

## Review Focus

- Anchor predicts uncovered label: ensemble output must equal anchor exactly.
- True uncovered label but anchor predicts covered: specialist may change the wrong covered label, but correctness must remain wrong; uncovered recall must therefore remain invariant.
- Specialist output must always belong to the fixed 8 covered labels.
- OOF row alignment between regenerated anchor, specialist predictions, and official train IDs must be exact.
- External rows must enter specialist training only; they must never enter anchor training or official validation scoring.

---

### Task 1: Lock gating and specialist data contracts

**Files:**
- Create: `tests/test_gated_specialist.py`
- Create: `scripts/run_gated_specialist.py`

**Interfaces:**
- Consumes: `scripts.run_strong_sparse_anchor.run_cv`, `build_fold_features`, `make_svc`, `load_train`, `summarize_oof`; `scripts.run_strong_sparse_external.validate_external`.
- Produces: `gate_predictions(anchor_predictions, specialist_predictions, covered_labels) -> np.ndarray`; `specialist_training_rows(official_fit, external, covered_labels) -> tuple[list[str], list[str]]`.

- [ ] **Step 1: Write failing tests** for: uncovered anchor predictions never change; covered anchor predictions are replaced by specialist; specialist outputs outside covered set fail; specialist training excludes uncovered official labels; exact production covered-label contract is 8 labels; blank/invalid external identity is still rejected.
- [ ] **Step 2: Run** `python -m pytest -q tests/test_gated_specialist.py` and verify failure because `scripts.run_gated_specialist` does not exist.
- [ ] **Step 3: Implement minimal contract helpers** in `scripts/run_gated_specialist.py` without CV/execution code beyond what tests require.
- [ ] **Step 4: Run** `python -m pytest -q tests/test_gated_specialist.py` and verify all contract tests pass.
- [ ] **Step 5: Commit** contract helpers/tests.

### Task 2: Implement fixed 5-fold specialist + gate experiment

**Files:**
- Modify: `scripts/run_gated_specialist.py`
- Modify: `tests/test_gated_specialist.py`

**Interfaces:**
- Consumes: Task 1 helpers and canonical Phase 5b functions.
- Produces: `run_specialist_cv(train, external, anchor_oof, seed=...) -> tuple[dict, pd.DataFrame]`; `build_report(...) -> dict`.

- [ ] **Step 1: Add failing tests** that require official-only anchor row alignment, specialist training on only covered official rows + all 2,503 external rows, ensemble OOF length 2,699, and exact uncovered per-model recall invariance versus anchor.
- [ ] **Step 2: Run focused tests** and verify the new execution/report tests fail.
- [ ] **Step 3: Implement fold loop:** regenerate anchor with canonical `run_cv`; per fold filter official fit to covered labels, append validated external rows, build separate strong-sparse features, fit 8-class SVC, predict all official validation rows, then apply fixed gate.
- [ ] **Step 4: Report** anchor/specialist/ensemble balanced accuracy, fold scores, per-model recall, covered mean recall, uncovered mean recall, changed-row count, and invariance checks.
- [ ] **Step 5: Run focused tests** and verify pass.
- [ ] **Step 6: Commit** execution/report implementation.

### Task 3: Freeze experiment contract and execute once

**Files:**
- Create: `artifacts/phase7/CONTRACT.md`
- Create: `.github/workflows/gated-specialist.yml`

**Interfaces:**
- Consumes: `python -m scripts.run_gated_specialist`.
- Produces: Actions artifact containing `artifacts/phase7/gated_specialist.json` and `artifacts/phase7/gated_specialist_oof.csv`.

- [ ] **Step 1: Write contract** containing the exact covered labels, gate rule, frozen Phase 5b parameters, external identity, permanent fold fingerprint, no-tuning rule, and promotion criterion `ensemble BA > 0.36709` with uncovered recall invariant.
- [ ] **Step 2: Add workflow** with a focused test job and a full job gated by `[phase7-run]` or manual dispatch.
- [ ] **Step 3: Trigger exactly one formal full run** with commit message containing `[phase7-run]` after focused tests are green.
- [ ] **Step 4: Inspect completed run and artifact**; verify 2,699 unique official IDs, no nulls, 18 valid final labels, permanent fold sizes, no prompt split across folds, and uncovered recall invariance.
- [ ] **Step 5: Commit** only the durable result artifacts needed for source-of-truth history.

### Task 4: Update source of truth and verify repository

**Files:**
- Create: `artifacts/phase7/RESULT.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: exact Phase 7 result artifact.
- Produces: durable promote/reject decision and next experiment direction.

- [ ] **Step 1: Document exact result** without post-hoc tuning; if BA <= 0.36709, reject this gating mechanism.
- [ ] **Step 2: Update README** Phase 7 state and next experiment; preserve Phase 5b as canonical CPU anchor unless Phase 7 beats it under the fixed contract.
- [ ] **Step 3: Run full repository verification** `python -m pytest -q` in CI with project + encoder dependencies.
- [ ] **Step 4: Inspect final branch diff** for accidental scope, secrets, or parameter drift.
