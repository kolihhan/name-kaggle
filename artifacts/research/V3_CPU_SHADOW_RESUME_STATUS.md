# V3 CPU Shadow Resume Status

Date: 2026-11-03 MYT

Branch: `infra/v3-cpu-shadow-resume`

## Status

**RESUME KERNEL READY; REAL QWEN CPU CHUNK TRAINER NOT YET AUTHORIZED AS SCIENTIFIC EVIDENCE.**

The purpose of this branch is to make slow free-CPU progress possible without mutating or impersonating the frozen CUDA v3 experiment.

## Verified TDD evidence

The exact test file now committed at `tests/test_cpu_shadow_resume.py` was executed locally against the implementation committed at `scripts/cpu_shadow_resume.py`.

RED before implementation:

- pytest collection failed with `ModuleNotFoundError: No module named 'scripts.cpu_shadow_resume'`.

GREEN after implementation/refinement:

- `4 passed`.

The tests verify:

1. a checkpoint taken after one microbatch inside a two-microbatch accumulation group resumes to a **bitwise-identical CPU parameter state** versus uninterrupted execution;
2. a mismatched lineage fingerprint fails closed;
3. Python, NumPy, and Torch CPU RNG streams resume exactly;
4. checkpoints persist only trainable parameters and do not serialize frozen parameters.

## Why the entire Qwen model is not checkpointed

A real v3 CPU job will reload the exact frozen Qwen base snapshot each process. Checkpoints persist only trainable LoRA/readout state plus optimizer/scheduler/RNG/partial-gradient/progress state. Copying frozen base weights into every checkpoint would make cross-job persistence unnecessarily huge.

## Scientific boundary

The frozen v3 worker uses CUDA fp16 autocast + GradScaler. A CPU worker cannot be assumed numerically identical to that CUDA trajectory.

Therefore CPU shadow evidence:

- may support or weaken the DCT mechanism hypothesis;
- may measure feasibility and transfer direction;
- may not authorize v3 Gate B;
- may not authorize submission;
- may not be relabeled as the frozen v3 transfer result.

## Parallel-gradient boundary

Six-way microbatch gradient parallelism is **not implemented**. LoRA dropout and head dropout make naïve independent-worker gradient accumulation non-equivalent to the sequential trajectory unless RNG equivalence is separately proven.

## Current external execution evidence

Existing GitHub run `41581953208` remains the only real exact-architecture CPU backward probe. At the latest check its single worst-case triplet forward/backward step was still in progress.

No second scientific CPU run was launched by this branch.

## Next gate

Before integrating the real Qwen CPU chunk worker:

1. obtain the probe's completed single-triplet timing/error evidence;
2. freeze CPU-specific numerical semantics explicitly;
3. implement one bounded resumable chunk for the existing transfer split;
4. verify checkpoint size and one stop/resume cycle;
5. only then decide whether repeated free-CPU chunks are operationally worth running.
