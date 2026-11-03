# V3 TPU Feasibility Design

## Purpose

Determine whether the currently available Kaggle TPU quota can execute the frozen v3 architecture fast enough to replace the unavailable GPU compute path, without consuming any v3 transfer evidence.

## Evidence budget

This work is execution-only. Until the TPU backend is proven viable it must not:

- read or score the frozen transfer holdout;
- read official competition rows for evaluation;
- read the final transfer firewall;
- create Public-LB predictions;
- create a competition submission;
- change the frozen v3 DCT modes, model depth, LoRA rank, sequence length, or source split.

## Gate 0 — accelerator identity

A minimal Kaggle kernel requests `TpuV5E8` and records only runtime identity:

- presence/version of `torch_xla`;
- PyTorch version;
- `torch_xla.device()` result;
- XLA device kind when available;
- basic tensor round-trip on XLA;
- wall-clock start/end.

No Qwen model, competition data, public pool, or labels are loaded.

PASS only if the run is actually backed by an XLA TPU device. A successful Kaggle kernel running on CPU is a FAIL for Gate 0.

This gate is necessary because a recent Kaggle CLI issue reports that pushed TPU kernels may silently receive the standard non-TPU image.

## Gate 1 — exact-architecture one-triplet feasibility

Only after Gate 0 PASS, run one execution-only probe using:

- `Qwen/Qwen3-8B-Base`, pinned revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` or the byte-equivalent pinned Kaggle model source;
- first 24 decoder layers;
- LoRA r16 / alpha32 / dropout0.05 on q/k/v/o/gate/up/down;
- v3 DCT `[DC0, AC1]` pooling on layers `[2,4,6,8,12,16,20,24]`;
- one realistic triplet of 3 rows × (447 body tokens + EOS);
- forward, backward and one optimizer step;
- no scoring.

The probe records compile time, step time, memory/runtime identity and whether the exact architecture completed.

## TPU numerical backend

TPU-native AMP uses BF16 rather than CUDA FP16 and does not require GradScaler. Therefore TPU cannot be described as numerically identical to the CUDA backend.

If Gate 1 PASSes and TPU is promoted to a scientific backend, a separate `TPU_BACKEND_AMENDMENT.md` must be frozen before any transfer prediction. That amendment must require v1 and v3 to use the same TPU/BF16 backend and preserve all non-hardware/non-precision parts of the scientific contract. No CUDA-vs-TPU cherry-picking is allowed.

## Orchestration

GitHub Actions may build and push the small Kaggle kernel and poll/download outputs. Kaggle performs the accelerator compute. The environment probe must use a private kernel and no internet unless required for the probe itself.

## Decision

- Gate 0 FAIL: do not build/run the Qwen TPU probe through the same unsupported path; retain CPU/GPU alternatives.
- Gate 0 PASS + Gate 1 FAIL: TPU is operationally unsuitable for frozen v3; do not tune architecture to rescue it.
- Gate 0 PASS + Gate 1 PASS: freeze the TPU scientific backend amendment, then implement the paired v1/v3 transfer once.
