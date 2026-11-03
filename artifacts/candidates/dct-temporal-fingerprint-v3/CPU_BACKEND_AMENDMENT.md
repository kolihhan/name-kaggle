# DCT Temporal Fingerprint v3 — CPU Backend Amendment

Status: **FROZEN BEFORE ANY V3 TRANSFER SCORE**

Date: 2026-11-03 MYT

This amendment authorizes a resource-constrained CPU execution path for the already-frozen `dct-temporal-fingerprint-v3` transfer experiment. It changes execution hardware and job boundaries only. It does not authorize a new candidate, new validation split, new hyperparameter search, or a second model-selection loop.

## Why this amendment exists

The original CUDA transfer run `41580819667` is valid but remains queued because no matching self-hosted GPU runner is online and Kaggle weekly GPU quota is exhausted. No v3 scientific score has been observed.

Public-repository standard GitHub-hosted CPU runners are available without runner-minute cost. Exact cross-job resume has been proven on fresh runners before this amendment was frozen.

## Scientific identity that remains frozen

The CPU path must preserve all of the following:

- candidate: `dct-temporal-fingerprint-v3`;
- control: v1 Stage-1 A1 masked-mean representation;
- treatment: v3 Stage-1 A1 fixed DCT-II modes `[DC0, AC1]`;
- model: `Qwen/Qwen3-8B-Base`;
- model revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`;
- first 24 decoder layers only;
- pooled layers `[2, 4, 6, 8, 12, 16, 20, 24]`;
- max body length 447 plus EOS;
- LoRA `r=16`, `alpha=32`, dropout `0.05`, bias `none`;
- LoRA target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`;
- head dropout `0.1`;
- one exact Stage-1 epoch;
- deterministic same-prompt triplet order;
- gradient accumulation of exactly six triplet microbatches per optimizer step, including the frozen final partial group semantics;
- adapter LR `1e-4`, head LR `1e-3`, AdamW weight decay `0.01`;
- warmup fraction `0.10` and the same LambdaLR cosine schedule;
- gradient clipping `1.0` after unscale and before optimizer step;
- seed `20261101`;
- frozen transfer split `alperen / hailey / willrice` and its zero prompt-id/key overlap contract;
- the existing transfer gate and class-safety veto;
- zero official-row scoring, zero final-firewall scoring, zero Public-LB tuning, zero competition-test use, and no submission authorization during transfer.

## CPU numerical backend

Both paired conditions must use the same CPU backend. Mixing a CUDA-trained control with a CPU-trained treatment, or vice versa, is forbidden.

The CPU implementation must preserve the training precision structure as closely as the platform allows:

- frozen Qwen backbone weights loaded as FP16;
- CPU autocast with `dtype=torch.float16` for the model forward path;
- LoRA trainable parameters stored in FP32, matching the CUDA worker's post-PEFT conversion;
- LayerNorm/readout parameters in FP32;
- `torch.amp.GradScaler("cpu", init_scale=256.0)`;
- `loss / exact_group_size` before scaling/backward;
- scaler state preserved across job boundaries, including mid-accumulation resumes;
- partial scaled gradients preserved across job boundaries;
- unscale, gradient clip, optimizer step, scaler update and scheduler step only at the same optimizer-step boundary as the frozen CUDA recipe.

CPU and CUDA floating-point kernels are not expected to be bitwise identical. This is an execution-backend nuisance, not a license to tune. The transfer comparison remains paired within one backend.

## Cross-job resume contract

A CPU job may stop after any completed microbatch. The successor job must restore, before computing the next microbatch:

- all trainable LoRA parameters;
- v1/v3 norms and plain/centered readout parameters;
- AdamW optimizer state;
- LR scheduler state;
- CPU GradScaler state;
- Python, NumPy and Torch RNG state;
- every non-null partial gradient;
- deterministic microbatch cursor and accumulation position;
- predecessor checkpoint SHA256 and generation lineage.

Frozen base-model weights are reconstructed from the pinned model revision and must not be copied into each rolling checkpoint. A full parameter name/shape/dtype/trainability contract must still be validated on restore.

## Storage policy

Long-running real-Qwen state must not use ordinary Actions artifacts as the primary rolling store because the state is expected to be hundreds of MiB and artifact quota is comparatively small.

The intended rolling store is GitHub Actions cache:

- exact generation-specific cache key;
- successor restores one exact predecessor key;
- successor saves one new generation key only after state validation succeeds;
- old generation cache is deleted after the successor state is safely committed;
- cache miss, ambiguous restore, stale predecessor SHA or lineage mismatch is a hard failure;
- no restore-key prefix fallback is allowed for scientific state.

Small manifests and final infrastructure evidence may still use ordinary artifacts.

## Decision hierarchy — frozen before scores

This CPU path is a fallback execution backend, not an extra validation view.

1. The already-queued CUDA run remains valid and must not be duplicated.
2. CPU v1 and CPU v3 may run in parallel because they are independent paired conditions under the same frozen backend.
3. The **first backend to complete the entire paired frozen transfer experiment** supplies the v3 transfer decision.
4. Once a backend has supplied that decision, a later result from the other backend is replication evidence only and must not reverse PASS/FAIL or trigger rescue tuning.
5. If the first completed paired transfer FAILs, v3 is stopped under the original gate; do not wait for the other backend hoping for a PASS.
6. If the first completed paired transfer PASSes, proceed only through the already-predeclared official-OOF path. The other backend cannot be used to tune or select anything.

## Authorization boundary

This amendment authorizes engineering and execution work needed to establish an exact resumable CPU path. It does **not** authorize launching an unbounded multi-thousand-job chain until a real-Qwen feasibility gate demonstrates that at least one exact microbatch can complete inside the GitHub-hosted job timeout and the rolling checkpoint size/storage path is viable.
