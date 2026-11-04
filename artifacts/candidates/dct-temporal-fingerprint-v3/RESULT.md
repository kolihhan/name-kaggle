# dct-temporal-fingerprint-v3 — Result / Execution Status

Date: 2026-11-03 MYT

## Status

**SCIENTIFIC PAIRED SOURCE-TRANSFER GATE UNRUN / UNKNOWN; GPU TRAINING FEASIBILITY VERIFIED.**

This is **not** a scientific FAIL. The frozen v1-vs-v3 paired source-transfer experiment has not produced any model score.

The original Kaggle execution backend is no longer required for this gate. After Kaggle rejected the frozen kernel because the weekly GPU quota was exhausted, the execution backend was changed to GitHub Actions orchestration with a dedicated self-hosted CUDA runner. The scientific contract is unchanged.

## Current verified execution addendum — 2026-11-04 MYT

- The historical self-hosted run `41580819667` was **CANCELLED** without running the frozen paired v1-v3 experiment. All references below to its queued status are historical, not current.
- Synthetic feasibility on Colab T4: GitHub Actions `41772827966` attempt 2 **SUCCESS**; 11.68 GiB peak reserved, real GPU forward/backward/optimizer, no heldout scoring.
- Real public external-source training **pilot** on Colab T4: `41778479162` **SUCCESS**, 31 train rows, 12 prompt-local microbatches, 2 optimizer updates; checkpoint artifact `13297683118` contains adapter and heads only. **It is not full Stage-1, not exact optimizer-state-resumable, and not source-transfer evaluation.**
- Infrastructure work `scripts/dct_v3_cuda_resume.py` and its three CPU-only equivalence/negative tests added. GitHub Actions `41792568893` **SUCCESS: 27 passed**, including frozen v3 tests and generation/compilation of workers. The helper is not yet integrated with frozen CUDA v1/v3 workers; CUDA resume equivalence remains unverified.
- **PIC sequence:** integrate boundary-safe resume into both frozen workers without recipe changes, prove interrupted-vs-uninterrupted equality on real fixed T4 batches, run the frozen one-epoch v1 and v3 pair with protected optimizer/scaler/RNG state, evaluate the locked 14,506 source-heldout rows once. Apply all original PASS/FAIL gates without changing thresholds. Official OOF is unauthorized until source-transfer PASS.
- Original self-hosted notes below are retained for historical traceability; `NOW.md` contains the active execution policy.

## Frozen experiment identity

- candidate: `dct-temporal-fingerprint-v3`
- contract: `artifacts/candidates/dct-temporal-fingerprint-v3/CONTRACT.md`
- transfer split: `artifacts/candidates/dct-temporal-fingerprint-v3/TRANSFER_SPLIT_LOCK.json`
- pool SHA-256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`
- holdout sources: `alperen`, `hailey`, `willrice`
- heldout rows: `14,506`
- eligible heldout rows: `14,506`
- eligible labels: `17`
- train rows after prompt-bank leakage removal: `14,502`
- prompt-id overlap: `0`
- prompt-key overlap: `0`
- split selection used model scores: `false`
- split artifact: `13200551511`
- split artifact digest: `sha256:6072697bbbcb8d9cd26431c4aa599c48e8de9a0e1ffafe4777f4bd4e55d92080`

## Implementation verification

### Original frozen v3 suite

GitHub Actions run `41569836280` (`V3 Static Tests`): **19 passed in 2.76s**.

Verified:

- materialized v3 Stage-1 worker compiled;
- frozen paired-transfer kernel compiled;
- exact transfer sources were embedded;
- no competition submission command was present.

### Local / self-hosted CUDA execution path

A local CUDA executor was added at `scripts/dct_v3_local_transfer.py` and is covered by `tests/test_dct_v3_local_transfer.py`.

GitHub-hosted CPU verification run `41580519705` completed successfully:

- **24 passed in 3.62s**;
- local CUDA executor imports and compiles;
- generated v3 worker compiles;
- exact pool SHA / model identity / source split / contamination guards are checked;
- the result validator cannot authorize a competition submission.

The executor runs the exact v1 A1 control and v3 A1 treatment sequentially on the same CUDA host, captures aligned heldout predictions, applies the already-frozen `dct_v3_source_eval.py` gate, preserves only the v3 Stage-1 checkpoint, and uses no official competition rows or Public-LB feedback.

## Kaggle infrastructure attempt — historical

Workflow run `41569982605` (`V3 DCT Paired Transfer`) reached the intended Kaggle kernel push. Kaggle rejected the push immediately with:

`Maximum weekly GPU quota of 30.00 hours reached.`

Therefore no scientific model run occurred on Kaggle.

## GitHub Actions self-hosted execution

Workflow: `.github/workflows/v3-self-hosted-gpu-transfer.yml`.

It is restricted to a matching runner:

`[self-hosted, linux, x64, gpu]`

and does not trigger from pull requests. The actual Qwen3-8B LoRA compute is intentionally **not** sent to a GitHub-hosted CPU runner, because doing so would not provide the CUDA environment required by the frozen worker.

Historical queued scientific run (since cancelled):

- run ID: `41580819667`
- head SHA: `2d0a86fffa880709ad25c8ee9153f2b2838617a8`
- job: `transfer`
- status: **queued**

At the time of the original report, a queued job proved workflow parsing/registration was valid. That job was later **CANCELLED**; it is no longer waiting for a runner.

The queued run will:

1. verify CUDA and >=15 GiB VRAM;
2. preserve the host CUDA-enabled PyTorch runtime while pinning `transformers==4.57.1` and `peft==0.17.1`;
3. rerun the frozen v3 focused tests on the GPU host;
4. run exactly one paired v1-vs-v3 external-source transfer experiment;
5. apply the predeclared gate;
6. upload the exact evidence plus v3 Stage-1 checkpoint;
7. perform no competition submission.

## Scientific decision

**UNRUN / UNKNOWN.**

Do not mark v3 PASS or FAIL until run `41580819667` obtains a matching GPU runner, completes the paired experiment, and `dct_v3_source_eval.py` applies the predeclared gate.

## Resume rule

Historical rule (superseded): bring online a dedicated self-hosted Linux x64 CUDA runner with custom label `gpu`, and do not duplicate its queued run. The job has since been cancelled; the active path is Colab T4 with an exact-resumable paired worker, per the addendum above.

Do **not** change:

- DCT modes or pooling weights;
- source holdout or labels;
- Qwen model/revision;
- seed/triplet order;
- LoRA/training settings;
- transfer thresholds;
- official folds;
- Public-LB policy.

If the transfer decision is FAIL, stop v3 permanently. If it is PASS, only then proceed to the separately predeclared official-OOF gate.