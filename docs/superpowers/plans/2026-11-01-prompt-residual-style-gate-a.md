# Prompt-Residual Style Gate A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute only the external source-transfer Gate A for `prompt-residual-style-v1`, without observing any new official-fold score.

**Architecture:** Reuse the exact Phase 14 conservative-subset builder, Phase 15 terms-clean filter, and Phase 3 frozen ModernBERT encoder. A new pure Gate A module builds prompt-residual LDA projections, constructs source holdouts that remove matching `prompt_id` and `prompt_key` rows globally, compares raw frozen embeddings against the projected representation, and emits a contract-shaped JSON result. A GitHub Actions wrapper packages the existing `scripts/` directory into a private Kaggle GPU kernel with the pinned public dataset mounted; no competition source or submission action is included.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, PyTorch, Transformers, GitHub Actions, Kaggle GPU kernel.

**Spec:** `artifacts/candidates/prompt-residual-style-v1/CONTRACT.md`

## Global Constraints

- Use exactly the Phase 15 terms-clean 26,025-row / 11-label corpus reconstructed from the pinned Phase 14 raw pool.
- Encoder is `answerdotai/ModernBERT-base`, frozen, max length 256, attention-mask mean pooling, L2 normalized.
- Gate A must run before any new official grouped OOF.
- Held-out source leakage exclusion uses both `prompt_id` and `prompt_key` across all sources.
- Eligible seed source: at least 100 rows, 20 unique prompt IDs, and 3 labels shared with remaining training rows.
- LDA is exactly `LinearDiscriminantAnalysis(solver="svd")`; no sweep.
- Comparison classifier is exactly `LinearSVC(class_weight="balanced", C=1.0)`.
- Gate passes only when mean source BA delta is positive and projected representation is not worse on more than half of eligible sources; fewer than two eligible sources is inconclusive.
- No official OOF, no Kaggle competition source, and no Kaggle submission in this plan.

## Review Focus

- A source that shares prompt-bank rows under a different source name must not leak through `prompt_id` or `prompt_key`.
- A prompt group with fewer than two distinct retained labels must not enter LDA fitting.
- Held-out labels absent from the remaining training partition must not be scored as if the classifier could learn them.
- A zero residual vector must not produce NaN during normalization.
- Gate status must become `inconclusive` rather than silently changing eligibility when fewer than two sources qualify.

---

### Task 1: Pure prompt-residual and holdout primitives

**Files:**
- Create: `scripts/run_prompt_residual_style_gate_a.py`
- Test: `tests/test_prompt_residual_style_gate_a.py`

**Interfaces:**
- Produces: `build_prompt_residuals(frame: pd.DataFrame, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
- Produces: `build_source_holdout(frame: pd.DataFrame, source: str) -> tuple[np.ndarray, np.ndarray, list[str]]`
- Produces: `eligible_seed_sources(frame: pd.DataFrame) -> list[str]`

- [ ] **Step 1: Write failing tests for prompt residuals**

Test that within-prompt centering is exact, groups with fewer than two distinct labels are excluded from the LDA-fit mask, zero residuals remain finite, and row-count mismatches raise.

- [ ] **Step 2: Run focused test and verify RED**

Run: `python -m pytest -q tests/test_prompt_residual_style_gate_a.py -k 'residual'`

Expected: FAIL because the new module/functions do not exist.

- [ ] **Step 3: Implement `build_prompt_residuals`**

Use `prompt_id` groups, compute group means on the supplied frozen embeddings, subtract, L2-normalize only non-zero vectors, and return the residual matrix plus a boolean fit mask for groups with at least two distinct model labels.

- [ ] **Step 4: Write failing tests for source holdout leakage**

Construct synthetic rows where another source reuses the held-out source's `prompt_id` and where a different `prompt_id` reuses its `prompt_key`. Assert both are excluded from training, held-out rows stay untouched, shared labels are the train/heldout intersection, and train/heldout prompt IDs and prompt keys are disjoint after exclusion.

- [ ] **Step 5: Implement `build_source_holdout` and `eligible_seed_sources`**

Reuse `SEED_SOURCES` from `scripts.run_phase14_public_sparse`. Eligibility is computed before model results and sorted deterministically by source name.

- [ ] **Step 6: Run Task 1 tests**

Run: `python -m pytest -q tests/test_prompt_residual_style_gate_a.py -k 'residual or holdout or eligible'`

Expected: PASS.

### Task 2: Frozen Gate A evaluator

**Files:**
- Modify: `scripts/run_prompt_residual_style_gate_a.py`
- Modify: `tests/test_prompt_residual_style_gate_a.py`

**Interfaces:**
- Consumes: Task 1 primitives.
- Produces: `fit_style_projection(frame: pd.DataFrame, embeddings: np.ndarray) -> LinearDiscriminantAnalysis`
- Produces: `evaluate_source_holdout(frame: pd.DataFrame, embeddings: np.ndarray, source: str) -> dict`
- Produces: `run_gate_a(frame: pd.DataFrame, embeddings: np.ndarray) -> dict`

- [ ] **Step 1: Write failing synthetic evaluator tests**

Use synthetic embeddings with a known within-prompt model-style direction. Assert the report contains raw/projected BA, per-label recalls, row/prompt/key counts, eligible-source list, source deltas, mean delta, worse-source count, and frozen pass/inconclusive decision fields.

- [ ] **Step 2: Run focused evaluator tests and verify RED**

Run: `python -m pytest -q tests/test_prompt_residual_style_gate_a.py -k 'projection or evaluate or gate'`

Expected: FAIL on missing evaluator functions.

- [ ] **Step 3: Implement LDA projection and source evaluation**

Fit LDA only on Task 1's residual fit mask. For each source, fit both comparison `LinearSVC` models on the remaining external training rows restricted to the shared label set; evaluate both on untouched held-out-source rows using balanced accuracy and fixed per-label recall reporting.

- [ ] **Step 4: Implement aggregate Gate A decision**

Unweighted mean source BA; strict positive mean delta; projected representation may be worse on at most `floor(n_sources / 2)` eligible sources; fewer than two eligible sources returns `inconclusive` with `official_oof_authorized=false`.

- [ ] **Step 5: Run all Gate A unit tests**

Run: `python -m pytest -q tests/test_prompt_residual_style_gate_a.py`

Expected: PASS.

### Task 3: Exact-data CLI and artifact contract

**Files:**
- Modify: `scripts/run_prompt_residual_style_gate_a.py`
- Modify: `tests/test_prompt_residual_style_gate_a.py`

**Interfaces:**
- Consumes: existing `build_conservative_subset`, `build_terms_clean_subset`, and `encode_texts`.
- Produces: CLI flags `--raw-pool`, `--output`, `--model`, `--batch-size`, `--max-length`, `--device`.

- [ ] **Step 1: Write failing contract tests**

Assert the CLI-level report records the pinned raw SHA, Phase 14 rows 43,781, Phase 15 rows 26,025, 2,801 prompt IDs, 11 retained labels, exact encoder config, exact LDA/SVC config, `official_rows_scored=0`, `kaggle_submission_authorized=false`, and a three-state Gate A decision (`pass`, `fail`, `inconclusive`).

- [ ] **Step 2: Implement exact reconstruction + embedding runner**

Verify the pinned raw SHA with existing Phase 14 helper, rebuild Phase 14 then Phase 15 using existing functions, encode the 26,025 texts once with the Phase 3 `encode_texts`, run Gate A, and write JSON.

When `--raw-pool` is omitted, search `/kaggle/input/**/pool_v4.parquet` and require exactly one match.

- [ ] **Step 3: Run all Gate A tests and existing encoder smoke tests**

Run: `python -m pytest -q tests/test_prompt_residual_style_gate_a.py tests/test_frozen_encoder_ablation.py tests/test_encoder_integration.py`

Expected: PASS.

### Task 4: Kaggle GPU execution wrapper

**Files:**
- Create: `.github/workflows/prompt-residual-style-gate-a.yml`

**Interfaces:**
- Consumes: `KAGGLE_API_TOKEN`, `nishantdahal/ntl-handwriting-blend`, repository `scripts/`, and `requirements-encoder.txt`.
- Produces: GitHub Actions artifact containing `prompt_residual_style_gate_a.json` and Kaggle kernel logs/output.

- [ ] **Step 1: Add test-only GitHub Actions job**

Install lightweight dependencies and run `tests/test_prompt_residual_style_gate_a.py` before any Kaggle execution.

- [ ] **Step 2: Add private Kaggle GPU job**

Package the repository `scripts/` directory and a tiny kernel entrypoint. Use `machine_shape="NvidiaTeslaT4"`, `enable_internet=true` only to fetch the frozen base encoder, `dataset_sources=["nishantdahal/ntl-handwriting-blend"]`, and **no competition source**.

- [ ] **Step 3: Verify downloaded Gate A artifact structurally**

Assert exact data/config invariants, `official_rows_scored == 0`, `kaggle_submission_authorized is False`, and decision is one of `pass/fail/inconclusive`. Print only compact source-level BA/delta and final Gate A status.

- [ ] **Step 4: Run repository verification before triggering Gate A**

Run the existing `verify-all` workflow or equivalent full pytest suite. Do not trigger Kaggle GPU if repository tests fail.

- [ ] **Step 5: Trigger Gate A once and archive the result**

The first successful contract-compliant Gate A result is consumed evidence. Do not rerun because the result is unfavorable. Infrastructure-only rerun is allowed only for a demonstrated execution failure before a valid result is produced.

### Task 5: Consume Gate A without touching official OOF

**Files:**
- Create after result: `artifacts/candidates/prompt-residual-style-v1/GATE_A_RESULT.md`
- Modify after result: `NOW.md`

**Interfaces:**
- Consumes: frozen Gate A JSON artifact.
- Produces: evidence-classified decision on whether official OOF is authorized.

- [ ] **Step 1: Record exact source-level results and frozen gate decision**

Separate FACT, INFERENCE, and UNKNOWN. Do not reinterpret eligibility or change pass rules.

- [ ] **Step 2: Update `NOW.md`**

If Gate A fails/inconclusive, stop the candidate. If Gate A passes, authorize only a separate Gate B implementation step; do not run Gate B in this plan.

- [ ] **Step 3: Verify no competition submission or new official OOF was created**

Check workflow/kernel metadata and repository submission artifacts before reporting completion.
