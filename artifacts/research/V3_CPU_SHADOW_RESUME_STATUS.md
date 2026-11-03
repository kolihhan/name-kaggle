# V3 CPU Shadow Resume Status

Date: 2026-11-03 MYT

Branch: `infra/v3-cpu-shadow-resume`

## Status

**RESUMABLE REAL-QWEN CPU SHADOW LANE IMPLEMENTED; FIRST BOUNDED DCT CHUNK IS RUNNING.**

The purpose of this branch is to make slow free-CPU progress possible without mutating or impersonating the frozen CUDA v3 experiment.

Authoritative frozen CUDA run remains `41580819667`. The CPU shadow lane cannot replace its transfer gate.

## Doctrine decision

CPU was previously treated as an operational dead end because one exact-architecture worst-case triplet was extremely slow. That restriction is now **REOPENED FOR SHADOW EVIDENCE** because the resource constraint changed the practical objective: making slow free progress is preferable to waiting indefinitely for unavailable GPU compute.

The restriction against treating CPU as frozen v3 evidence remains ACTIVE.

### CPU numerical-backend restriction

RESTRICTION: keep the current first shadow chunk on the preregistered CPU-fp16 backend; do not change backend mid-run.

WHY IT EXISTS: changing dtype/backend during an execution would destroy lineage and make the result uninterpretable.

CURRENT EVIDENCE: existing probe `41581953208` and first resumable chunk `41598304633` are both CPU-fp16 lineage.

PIVOT TRIGGER: if fp16 is operationally too slow/times out, run a **no-scoring execution benchmark** comparing CPU fp16 with CPU bfloat16 under the same architecture/input. Promote bfloat16 only if it is materially faster and receives a separate lineage fingerprint.

STATUS: ACTIVE for the current run; bfloat16 is REOPENED as the next execution-backend candidate if the trigger fires.

PyTorch's supported CPU mixed-precision path uses `torch.autocast(device_type='cpu', dtype=torch.bfloat16)`; this is an execution optimization hypothesis, not a model-quality claim.

## Verified TDD evidence

### Resume kernel

`tests/test_cpu_shadow_resume.py` was written before `scripts/cpu_shadow_resume.py`.

RED:

- pytest collection failed with `ModuleNotFoundError: No module named 'scripts.cpu_shadow_resume'`.

GREEN:

- `4 passed` locally.

The tests verify:

1. checkpoint after one microbatch inside a gradient-accumulation group resumes to a **bitwise-identical CPU parameter state** versus uninterrupted toy execution;
2. mismatched lineage fails closed;
3. Python, NumPy, and Torch CPU RNG streams resume exactly;
4. checkpoints persist only trainable parameters and do not serialize frozen parameters.

### CPU chunk contract

`tests/test_dct_v3_cpu_shadow_chunk.py` was also written before the chunk implementation.

Together with resume tests, local verification is:

- `9 passed`;
- `scripts/cpu_shadow_resume.py` compiles;
- `scripts/dct_v3_cpu_shadow_chunk.py` compiles.

These are infrastructure tests only. They do not prove that an 8B CPU microbatch completes within a hosted-runner time budget.

## Real Qwen chunk implementation

`scripts/dct_v3_cpu_shadow_chunk.py` now supports:

- `mean` shadow representation for the v1 control;
- `dct` shadow representation for v3 DC0+AC1;
- exact frozen model revision and Stage-1 training spec;
- exact frozen `alperen / hailey / willrice` transfer split;
- same deterministic prompt-triplet construction and six-triplet gradient accumulation;
- bounded `--max-microbatches` execution;
- trainable-only checkpoint/resume, including partial accumulated gradients;
- immutable lineage fingerprints;
- no official rows, Public LB, or submission authorization.

CPU backend is explicitly labeled:

`cpu_float16_no_autocast_no_gradscaler`

It is not claimed numerically identical to CUDA fp16 autocast + GradScaler.

## Why the entire Qwen model is not checkpointed

Every CPU process reloads the exact frozen Qwen base snapshot. Checkpoints persist only trainable LoRA/readout state plus optimizer/scheduler/RNG/partial-gradient/progress state. Copying frozen base weights into every checkpoint would make cross-job persistence unnecessarily huge.

## Parallel-gradient boundary

Six-way microbatch gradient parallelism is **not implemented**. LoRA dropout and head dropout make naïve independent-worker gradient accumulation non-equivalent to the sequential CPU trajectory unless RNG equivalence is separately proven.

RESTRICTION: no parallel gradient reduction yet.

PIVOT TRIGGER: a dedicated test must prove equivalence under an explicitly frozen per-microbatch RNG scheme before this is allowed.

## GitHub Actions path

A narrow user-authorized public-repo exception is recorded in this branch's `PUBLIC_VISIBILITY.md`.

Workflow:

`.github/workflows/v3-cpu-shadow-chunk.yml`

Safeguards include:

- delete official `data/train.csv`, `data/test.csv`, and `data/sample_submission.parquet` immediately after checkout;
- use only immutable external Phase14 artifact `13067907516`;
- no Kaggle credentials/private competition dataset;
- no automatic unlimited chaining;
- one deliberately bounded chunk;
- checkpoint/result upload after clean completion;
- no submission authorization.

Initial workflow configuration run `41598207466` failed before any job started because `runner.temp` was referenced in job-level `env`, where the `runner` context is unavailable. The workflow was corrected to initialize `HF_HOME` at runtime from `$RUNNER_TEMP`.

Corrected first DCT chunk:

- run: `41598304633`
- representation: `dct`
- max microbatches: `1`
- status at latest check: **IN PROGRESS**
- official-data workspace removal: **SUCCESS**
- no scientific score has been produced yet.

## Existing worst-case CPU probe

Run `41581953208` remains an execution-only feasibility probe for one synthetic worst-case `3 x 448` token triplet. At the latest check its single forward/backward step was still in progress. It has produced no scientific score.

## Decision boundary after the first chunk

When run `41598304633` resolves:

1. inspect whether one real DCT microbatch completed;
2. measure wall time and trainable-only checkpoint size;
3. if it completed cleanly and throughput is tolerable, perform one explicit resume chunk from that checkpoint before scaling further;
4. if fp16 times out or is operationally absurd, do not repeat it indefinitely — trigger the no-scoring CPU-bfloat16 execution benchmark;
5. only after a successful stop/resume cycle decide whether slow free-CPU continuation is worth sustained use.

No CPU result may authorize official OOF or Kaggle submission.