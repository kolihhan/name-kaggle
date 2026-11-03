# V3 CPU Shadow Backend Pivot

Date: 2026-11-03 MYT

## Scope

This document tracks CPU execution/training evidence only. It does not alter the frozen CUDA v3 scientific contract.

## User execution priority

On 2026-11-03 the user explicitly chose **CPU-first training** despite the expected slowness.

Operational consequence:

- continue the CPU lineage first;
- use bounded, resumable chunks with trainable-only checkpoints;
- do not require GPU execution before making CPU training progress;
- keep CPU evidence scientifically isolated from the frozen CUDA v3 transfer gate.

This supersedes the earlier operational recommendation to reject sustained CPU training purely on estimated wall-clock cost.

## FP16 anchor

- run: `41598304633`
- representation: `dct`
- backend: CPU float16, no autocast, no GradScaler
- max microbatches: `1`
- status at latest check: **IN PROGRESS**
- it has not completed one real DCT microbatch yet
- official rows scored: `0`
- Public LB tuning: `false`
- submission authorization: `false`

## BF16 pivot history

The CPU-fp16 restriction was reopened under `RESEARCH_DOCTRINE.md` because the existing worst-case CPU probe and first resumable chunk were operationally slow.

TDD evidence:

- RED run `41601227520`: `scripts.cpu_shadow_backend` absent.
- GREEN run `41601409082`: explicit fp16/bf16 backend contract passed.
- chunk-lineage RED run `41601539263`: backend-aware lineage not yet implemented.
- corrected GREEN run `41601753518`: backend + chunk-lineage static contracts passed; heavy job skipped.

First BF16 benchmark `41602080074` did **not** complete a microbatch. It reached the readout quickly, then failed at the BF16 -> FP32 linear boundary with:

`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`

That run produced no training checkpoint/result and must not be treated as throughput or scientific evidence.

The readout boundary was then fixed by promoting representation features back to FP32 before the frozen FP32 classifier heads. The fix has a dedicated static contract in `tests/test_dct_v3_cpu_shadow_chunk.py`.

Corrected BF16 run:

- run: `41604738508`
- representation: `dct`
- backend: CPU bfloat16 + CPU autocast, no GradScaler
- max microbatches: `1`
- separate lineage fingerprint from FP16
- status at latest check: **IN PROGRESS** in the first real bounded training microbatch
- official rows scored: `0`
- Public LB tuning: `false`
- submission authorization: `false`

## CPU training decision

**STATUS: CPU-FIRST TRAINING ACTIVE.**

WHY:

- the user explicitly prefers making CPU training progress before using GPU;
- the resumable checkpoint path already preserves trainable LoRA/head/norm parameters, optimizer, scheduler, partial gradients, RNG, and progress;
- each chunk can be bounded so a failed or timed-out run does not redefine the scientific candidate.

CURRENT EVIDENCE / COST WARNING:

- frozen transfer training has `14,502` rows;
- prompt triplets contain at most three rows, so a full epoch requires at least `4,834` microbatches;
- both FP16 and corrected BF16 hosted-CPU runs are very slow so far.

This makes full completion potentially expensive in wall-clock time, but it is now an explicit user-priority tradeoff rather than a hard method-class rejection.

## Continuation rule

1. Let corrected BF16 run `41604738508` finish its first real microbatch.
2. Continue from BF16 only if it produces a finite loss and a valid checkpoint.
3. Resume sequentially from that checkpoint in deliberately bounded chunks.
4. Do not mix FP16 and BF16 checkpoints.
5. Do not create an unlimited automatic chain; every continuation must remain bounded and recoverable.
6. Do not score official competition rows, consume Public LB, or submit from the CPU lane.
7. CPU progress does not authorize the frozen CUDA v3 transfer gate or official OOF.

## Pivot trigger

Reconsider the CPU execution method if a completed BF16 microbatch shows an operationally extreme cost or if repeated chunks fail to make checkpointed progress. Any optimization must preserve the frozen candidate and maintain explicit lineage separation.
