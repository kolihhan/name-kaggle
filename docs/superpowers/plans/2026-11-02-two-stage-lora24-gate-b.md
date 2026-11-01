# Two-stage LoRA24 Gate B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement leakage-safe permanent 5-fold official OOF Stage-2 evaluation for `two-stage-lora24-v1`, initialized only from the exact Gate-A PASS checkpoint, without executing Gate B or submitting to Kaggle until the implementation is separately verified and authorized.

**Architecture:** Materialize the immutable Gate-A A1 checkpoint into a private, version-pinned Kaggle dataset with file-level lineage, then run five independent fold workers. Every fold starts from that same checkpoint, trains only on official `fold_id != f` rows for exactly two epochs, and predicts untouched `fold_id == f` rows. A CPU-side evaluator regenerates the frozen Phase-7 row-level OOF, aligns both systems on official IDs, computes the contract diagnostics and paired prompt-group bootstrap, and applies the already-frozen Gate-B decision mechanically.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch, `transformers==4.57.1`, `peft==0.17.1`, safetensors, scikit-learn metrics, Kaggle CLI `2.2.5`, GitHub Actions, private Kaggle datasets/kernels.

**Spec:** `artifacts/candidates/two-stage-lora24-v1/CONTRACT.md`

## Global Constraints

- Candidate: `two-stage-lora24-v1`; Gate A is already frozen PASS and must not be rerun.
- Exact source checkpoint: GitHub Actions run `41333631967`, artifact `13094069155`, artifact ZIP digest `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`.
- Base model: `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Runtime: `transformers==4.57.1`, `peft==0.17.1`.
- Representation stays identical to Gate A: first 24 Qwen layers, pooled layers `[2,4,6,8,12,16,20,24]`, truncate to 447 before EOS, right padding, first-token masking, eight trainable LayerNorms, plain + centered heads, head dropout `0.1`.
- LoRA stays identical to Gate A: `r=16`, alpha `32`, dropout `0.05`, no bias, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- Permanent official fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.
- Official train SHA256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- Stage 2: exactly 2 epochs; LoRA LR `5e-5`; heads/norms LR `5e-4`; weight decay `0.01`; 10% warm-up then cosine; gradient accumulation equivalent to 6 prompt-triplet microbatches per optimizer step; gradient clip `1.0`; seed `20261101`.
- Each fold must load the exact same Stage-1 A1 checkpoint from disk in a fresh worker process before any official-data update. Never carry parameters across folds.
- Stage-2 class weights are computed only from that fold's official fit rows. Validation rows may not affect class weights, training, calibration, early stopping, epoch choice, thresholds, or routing.
- No public Stage-1 rows are mixed back into Stage 2.
- The two epochs replay one deterministic triplet ordering built once with seed `20261101`; the LR schedule spans all optimizer steps across both epochs.
- Primary OOF prediction is the plain head only. Centered head is diagnostic only and cannot affect Gate-B PASS.
- Centered validation predictions use deterministic same-prompt triplets built with seed `20261101`, solely to ensure every validation row gets one secondary diagnostic prediction.
- No Phase-19 shadow fold, new fold salt, hyperparameter sweep, threshold search, blend, Public-LB tuning, competition test inference, or Kaggle competition submission.
- Gate-B PASS is exactly the frozen four-condition rule in the candidate contract; implementation must not weaken it.

## Review Focus

- **Checkpoint substitution:** Gate B must fail if the private Kaggle input does not match the frozen checkpoint lineage, candidate, model revision, labels, or Gate-A implementation invariants.
- **Fold contamination:** no validation prompt ID may appear in fit rows; no validation row may affect class weights, triplet construction, optimizer state, or model selection.
- **Cross-fold state leakage:** each fold worker must start from the same Stage-1 checkpoint rather than from the previous fold's Stage-2 state.
- **Baseline alignment:** regenerated Phase-7 OOF must match frozen Phase-7 aggregate/fold metrics before candidate-vs-Phase7 disagreement/bootstrap is accepted.
- **Bootstrap unit:** paired bootstrap must resample `prompt_id` groups, not individual rows, while scoring candidate and Phase 7 on the same sampled groups.

---

### Task 1: Lock pure Gate-B contracts and decision helpers

**Files:**
- Create: `scripts/run_two_stage_lora24_gate_b.py`
- Create: `tests/test_two_stage_lora24_gate_b.py`

**Interfaces:**
- Consumes: `scripts.run_two_stage_lora24_gate_a.gate_a_training_spec`, `build_prompt_triplets`, `compute_class_weights`, `group_triplets_for_accumulation`; `scripts.run_style_baseline.EXPECTED_TRAIN_SHA256`, `EXPECTED_FOLD_FINGERPRINT`, `fold_fingerprint`.
- Produces: `gate_b_training_spec() -> dict[str, object]`; `validate_gate_a_checkpoint_meta(meta: dict) -> None`; `split_official_fold(train: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]`; `score_oof(...) -> dict`; `paired_prompt_bootstrap_delta(...) -> dict`; `gate_b_decision(...) -> dict`.

- [ ] **Step 1: Write failing tests** asserting exact frozen Stage-2 values, exact Gate-A checkpoint identity/invariants, official train/fold fingerprint checks, 5-fold prompt isolation, fit-only class-weight inputs, and rejection of malformed or substituted checkpoint metadata.
- [ ] **Step 2: Run** `python -m pytest -q tests/test_two_stage_lora24_gate_b.py` and verify RED because the module does not yet exist.
- [ ] **Step 3: Implement minimal pure helpers**. Reuse Gate-A representation/triplet helpers rather than copying them. `split_official_fold` must reject fold IDs outside `0..4`, missing rows, or prompt overlap.
- [ ] **Step 4: Add failing bootstrap/decision tests** using a tiny synthetic prompt-group dataset. Assert exactly `5_000` resamples, seed `20261101`, paired candidate/Phase7 scoring on identical sampled prompt groups, CI lower-bound behavior, `>=3/5` fold rule, and the class-recall safety veto.
- [ ] **Step 5: Implement bootstrap and mechanical decision logic**. The bootstrap sampling unit is unique `prompt_id`; after sampling groups with replacement, include every row belonging to each sampled group, preserving multiplicity.
- [ ] **Step 6: Run focused tests** and require all Gate-B pure tests PASS.
- [ ] **Step 7: Commit** pure Gate-B contracts/tests.

### Task 2: Materialize the exact A1 checkpoint as immutable Kaggle input

**Files:**
- Create: `.github/workflows/two-stage-lora24-gate-b-checkpoint.yml`
- Create: `tests/test_two_stage_lora24_gate_b_checkpoint_workflow.py`
- Create after successful materialization: `artifacts/candidates/two-stage-lora24-v1/GATE_B_CHECKPOINT_LOCK.md`

**Interfaces:**
- Consumes: Actions artifact `13094069155` from run `41333631967`.
- Produces: private Kaggle dataset `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`, pinned to one exact version; `checkpoint_lineage.json`; deterministic per-file SHA256 manifest; lock document containing the resulting dataset version and manifest digest.

- [ ] **Step 1: Write a failing workflow-contract test** requiring the source run/artifact IDs and artifact digest above, no Gate-A training invocation, no `kaggle kernels push`, private dataset creation (no `--public`), and fail-closed behavior if a dataset with conflicting lineage already exists.
- [ ] **Step 2: Run the focused test** and verify RED because the checkpoint-materialization workflow does not exist.
- [ ] **Step 3: Implement the workflow** to fetch artifact metadata from GitHub, assert the immutable artifact digest, download/extract the artifact, assert `adapter/`, `heads.safetensors`, `meta.json`, tokenizer state, and validate `meta.json` through the Task-1 helper.
- [ ] **Step 4: Build `checkpoint_lineage.json`** containing source run, artifact ID/digest, candidate, model revision, Gate-A implementation invariants, and SHA256+size for every checkpoint file. Create a deterministic `a1-checkpoint.tar.gz` plus lineage file for transport.
- [ ] **Step 5: Create one private Kaggle dataset** with `kaggle datasets create -p ...`; Kaggle CLI defaults to private unless `--public` is passed. Never auto-version an existing conflicting dataset.
- [ ] **Step 6: Read back Kaggle dataset status/files**, record the exact dataset version and verify the uploaded transport files are present. Write `GATE_B_CHECKPOINT_LOCK.md` only after this succeeds.
- [ ] **Step 7: Run focused tests and inspect the lock**; verify no Stage-1 scientific rerun occurred.
- [ ] **Step 8: Commit** workflow/test/lock record.

### Task 3: Implement one fresh Stage-2 fold worker

**Files:**
- Create: `scripts/run_two_stage_lora24_gate_b_gpu.py`
- Create: `tests/test_two_stage_lora24_gate_b_gpu_contract.py`

**Interfaces:**
- Consumes: Task-1 spec/helpers; exact checkpoint directory extracted from Task-2 Kaggle dataset; official `data/train.csv`; `artifacts/phase1/folds.csv`.
- Produces: `run_fold_worker(fold: int, ...) -> None` writing `fold_<f>.json` and `fold_<f>_predictions.csv`; internal `load_exact_stage1_checkpoint(...)`; `train_two_stage2_epochs(...)`; plain and centered validation predictions.

- [ ] **Step 1: Write failing source/contract tests** requiring: pinned Qwen revision and library versions; exact Stage-1 checkpoint validation; fresh model/checkpoint load per worker; no public Stage-1 rows; `fold_id != f` only for training; `fold_id == f` only for validation; two epochs; fixed Stage-2 LRs; no early stopping or validation-driven branch.
- [ ] **Step 2: Run focused tests** and verify RED because the GPU runner does not exist.
- [ ] **Step 3: Implement exact checkpoint load**: load pinned 24-layer Qwen base, attach the saved LoRA adapter as trainable, restore eight LayerNorms + plain/centered heads from `heads.safetensors`, restore tokenizer, and assert the saved label order equals official sorted 18 labels.
- [ ] **Step 4: Implement fold-local tokenization/training** using the Gate-A representation helpers. Compute class weights only from fit rows. Build one deterministic fit-triplet list with seed `20261101`; replay it twice. Compute total optimizer steps across both epochs before constructing one 10%-warmup/cosine schedule.
- [ ] **Step 5: Implement plain validation inference** over every untouched validation row and deterministic centered-head diagnostic inference over validation same-prompt triplets. Assert each validation ID receives exactly one plain and one centered prediction.
- [ ] **Step 6: Persist per-fold audit metadata**: train/validation row counts, train/validation prompt counts, zero prompt overlap, triplet count/epoch, optimizer steps, skipped scaler steps, training loss, checkpoint manifest identity, prediction counts, plain/centered fold BA, and no-null/valid-label checks.
- [ ] **Step 7: Run focused tests** and require PASS; compile the GPU module.
- [ ] **Step 8: Commit** fold-worker implementation/tests.

### Task 4: Implement five-fold orchestration and immutable Gate-B evidence

**Files:**
- Modify: `scripts/run_two_stage_lora24_gate_b_gpu.py`
- Modify: `tests/test_two_stage_lora24_gate_b_gpu_contract.py`
- Create: `.github/workflows/two-stage-lora24-gate-b.yml`

**Interfaces:**
- Consumes: Task-2 pinned private checkpoint dataset and Task-3 fold worker.
- Produces: one Kaggle output bundle containing five fold JSONs, five prediction CSVs, combined candidate OOF CSV, and runtime lineage.

- [ ] **Step 1: Add failing orchestration tests** requiring exactly folds `0..4`, one fresh subprocess per fold, no fold skipped/repeated, no `kaggle competitions submit`, and a heavy run only when a dedicated `[gate-b-run]` authorization marker is present.
- [ ] **Step 2: Implement orchestrator**. Embed the exact repo scripts, official `data/train.csv`, and permanent folds file into the private kernel payload; attach only the pinned private checkpoint dataset. Internet may be enabled solely to retrieve the pinned base-model revision.
- [ ] **Step 3: Schedule fold workers** independently. If multiple GPUs are visible, use a bounded worker queue; otherwise run sequentially. Parallelism must not change seeds, checkpoint bytes, fold membership, or outputs.
- [ ] **Step 4: Combine fold predictions** by official `id`; require 2,699 unique official IDs, exactly one prediction per row, all 18 labels valid, permanent fold fingerprint unchanged, and no competition-test rows.
- [ ] **Step 5: Add workflow verification** that downloads output and fails closed on missing fold evidence, wrong checkpoint lineage, fold overlap, duplicate/missing official IDs, or any competition submission path.
- [ ] **Step 6: Upload candidate OOF/evidence as Actions artifacts**. Do not upload or create five post-Stage2 fold checkpoints unless needed for debugging; v1 promotion is based on OOF predictions, not checkpoint cherry-picking.
- [ ] **Step 7: Run focused workflow-contract tests** and require PASS.
- [ ] **Step 8: Commit** orchestration/workflow without the `[gate-b-run]` marker.

### Task 5: Regenerate the frozen Phase-7 row baseline and evaluate Gate B mechanically

**Files:**
- Modify: `scripts/run_two_stage_lora24_gate_b.py`
- Modify: `tests/test_two_stage_lora24_gate_b.py`

**Interfaces:**
- Consumes: combined Gate-B candidate OOF; `scripts.run_gated_specialist` to regenerate Phase-7 OOF; `artifacts/phase7/gated_specialist.json` frozen aggregate reference.
- Produces: `evaluate_gate_b(candidate_oof, train, phase7_oof, phase7_report) -> dict` and final Gate-B report JSON.

- [ ] **Step 1: Add failing baseline-alignment tests** requiring identical `id,prompt_id,model,fold_id` ordering and requiring regenerated Phase-7 overall BA, five fold BAs, and per-class recall to match the frozen Phase-7 JSON within `1e-12` before comparison proceeds.
- [ ] **Step 2: Regenerate Phase-7 OOF** with the existing frozen `run_gated_specialist` pipeline and committed external data; do not infer row predictions from aggregate JSON.
- [ ] **Step 3: Add failing diagnostic tests** for candidate overall BA, Phase-7 BA, delta, five fold deltas, per-class recall deltas, disagreement rate, `candidate_only_correct`, `phase7_only_correct`, and centered-head standalone diagnostics.
- [ ] **Step 4: Implement diagnostics and paired prompt bootstrap** using Task-1 helpers. Record the full bootstrap summary (`n_resamples`, seed, point delta, 2.5%/97.5% quantiles) but not all 5,000 samples unless needed for audit.
- [ ] **Step 5: Apply `gate_b_decision` mechanically**. The report must expose each of the four frozen PASS conditions individually and the final `pass|fail`; no compensating mechanism may be invented post-result. If the class-recall veto fires, v1 fails.
- [ ] **Step 6: Run focused tests** and require PASS.
- [ ] **Step 7: Commit** evaluator implementation/tests.

### Task 6: Verification gate before any scientific Gate-B execution

**Files:**
- Create: `artifacts/candidates/two-stage-lora24-v1/GATE_B_RUN_AUTHORIZATION.md` only after Tasks 1-5 are fully verified.
- Modify: `NOW.md`.

**Interfaces:**
- Consumes: all Task 1-5 code/tests/workflows and Task-2 checkpoint lock.
- Produces: explicit one-shot authorization record; no model metric yet.

- [ ] **Step 1: Run candidate-focused tests** for Gate B pure logic, GPU contract, checkpoint workflow, and Gate-B workflow.
- [ ] **Step 2: Run full repository verification** `python -m pytest -q`; record exact current-head commit and test count.
- [ ] **Step 3: Inspect diff from current pre-Gate-B head** for scientific drift: backbone, LoRA config, pooled layers, max length, Stage-2 epochs/LRs, fold fingerprint, seed, class-weight source, primary head, bootstrap count, PASS thresholds.
- [ ] **Step 4: Create one-shot authorization record** naming the exact source commit, exact checkpoint dataset version/manifest, official train SHA/fold fingerprint, and stating explicitly: one Gate-B OOF execution; no Public-LB tuning; no competition test inference; no Kaggle submission.
- [ ] **Step 5: Update `NOW.md`** to `Gate B implementation ready / authorized but not yet executed` and point to the authorization record.
- [ ] **Step 6: Commit** the authorization record without changing scientific code.

### Task 7: Execute Gate B once, consume once, and close or promote locally

**Files:**
- Create after result: `artifacts/candidates/two-stage-lora24-v1/GATE_B_RESULT.md`
- Modify: `NOW.md`

**Interfaces:**
- Consumes: exact authorized implementation from Task 6.
- Produces: one frozen Gate-B verdict and preserved OOF evidence. It does not submit to Kaggle.

- [ ] **Step 1: Trigger exactly one scientific Gate-B workflow** with `[gate-b-run]`. Do not rerun because the score is disappointing.
- [ ] **Step 2: If CI fails, classify first** whether a valid scientific result exists. Infrastructure-only recovery may attach/download an already-produced result but must not launch another scientific training run unless no valid result was produced and the failure is independently shown to be infrastructure-only.
- [ ] **Step 3: Verify result invariants**: 2,699 official OOF rows, 150 prompt groups, permanent fold fingerprint, five completed fold workers, checkpoint lineage match, zero fold prompt overlap, no test rows, no Public-LB use, no submission.
- [ ] **Step 4: Consume the Gate-B report exactly once** and write `GATE_B_RESULT.md` with candidate/Phase7 overall BA, five fold scores/deltas, bootstrap CI, per-class recall deltas, disagreement/conditional correctness, centered-head diagnostic, every PASS condition, and final verdict.
- [ ] **Step 5: If FAIL**, close `two-stage-lora24-v1`; no class repair, blend, extra epoch, new seed, new fold, or Public-LB rescue inside v1.
- [ ] **Step 6: If PASS**, mark the candidate as locally eligible for a separate promotion/preflight decision only. Do not create a competition submission automatically.
- [ ] **Step 7: Update `NOW.md` and run final repository verification**. Preserve the exact Gate-B OOF/evidence artifact IDs and digests.
