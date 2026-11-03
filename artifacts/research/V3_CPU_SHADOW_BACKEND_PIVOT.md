# V3 CPU Shadow Backend Pivot

Date: 2026-11-03 MYT

## Scope

This document tracks execution-backend evidence only. It does not alter the frozen CUDA v3 scientific contract.

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

Corrected BF16 benchmark:

- run: `41604738508`
- representation: `dct`
- backend: CPU bfloat16 + CPU autocast, no GradScaler
- max microbatches: `1`
- separate lineage fingerprint from FP16
- status at latest check: **IN PROGRESS** in the real bounded microbatch
- official rows scored: `0`
- Public LB tuning: `false`
- submission authorization: `false`

## Full-CPU viability decision

**REJECT sustained full Stage-1 CPU training as an operational route.**

WHY:

- frozen transfer training has `14,502` rows;
- `build_prompt_triplets` forms microbatches of at most three rows, so one exact epoch requires at least `ceil(14502 / 3) = 4,834` forward/backward microbatches;
- the corrected BF16 one-microbatch run has already remained in progress for tens of minutes, while the FP16 anchor has taken still longer without completing;
- even an optimistic hypothetical `10 minutes / microbatch` would require about `48,340 minutes = 33.6 days` for a single epoch before setup/reloads/checkpoint overhead.

CURRENT EVIDENCE:

CPU can still provide execution/mechanism diagnostics, but its demonstrated order of magnitude is incompatible with sustained exact Stage-1 training on free hosted CPU.

PIVOT TRIGGER TO REOPEN FULL CPU TRAINING:

Only reopen if a materially different CPU execution method demonstrates **sub-minute** exact microbatches without changing the scientific candidate and without violating lineage/equivalence requirements. No such evidence exists now.

STATUS: **REJECTED for full training; retained only as bounded diagnostic evidence.**

## Decision rule from here

Do not launch additional FP16/BF16 full-training chunks.

Let the already-running corrected BF16 benchmark finish only as a bounded throughput diagnostic. Do not chain it into sustained CPU training.

Do not mix or resume checkpoints across FP16/BF16 lineages.

The real v3 path remains the authoritative exact CUDA execution (`41580819667`) or another exact-compatible GPU runner. CPU evidence cannot authorize the frozen CUDA v3 transfer gate, official OOF, or Kaggle submission.
