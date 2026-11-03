# dct-temporal-fingerprint-v3 — Result / Execution Status

Date: 2026-11-03 MYT

## Status

**WAITING_FOR_SELF_HOSTED_GPU_RUNNER — SCIENTIFIC TRANSFER GATE UNRUN.**

This is **not** a scientific FAIL. The frozen v1-vs-v3 paired source-transfer experiment has not produced any model score.

The original Kaggle execution backend is no longer required for this gate. After Kaggle rejected the frozen kernel because the weekly GPU quota was exhausted, the execution backend was changed to GitHub Actions orchestration with a dedicated self-hosted CUDA runner. The scientific contract is unchanged.

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

Current queued scientific run:

- run ID: `41580819667`
- head SHA: `2d0a86fffa880709ad25c8ee9153f2b2838617a8`
- job: `transfer`
- status: **queued**

A real queued job exists, so workflow parsing/registration is valid. It is waiting for a self-hosted Linux x64 runner carrying the custom `gpu` label.

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

Bring online a dedicated self-hosted Linux x64 CUDA runner with custom label `gpu`. Do not create a replacement scientific run while `41580819667` remains queued.

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