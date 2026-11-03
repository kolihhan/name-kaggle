# V3 CPU Shadow Backend Pivot

Date: 2026-11-03 MYT

## Scope

This document tracks execution-backend evidence only. It does not alter the frozen CUDA v3 scientific contract.

## FP16 anchor

- run: `41598304633`
- representation: `dct`
- backend: CPU float16, no autocast, no GradScaler
- max microbatches: `1`
- status at recording: **IN PROGRESS**
- official rows scored: `0`
- Public LB tuning: `false`
- submission authorization: `false`

## BF16 pivot

The CPU-fp16 restriction was reopened under `RESEARCH_DOCTRINE.md` because the existing worst-case CPU probe and first resumable chunk were operationally slow.

TDD evidence:

- RED run `41601227520`: `scripts.cpu_shadow_backend` absent.
- GREEN run `41601409082`: explicit fp16/bf16 backend contract passed.
- chunk-lineage RED run `41601539263`: backend-aware lineage not yet implemented.
- first implementation check `41601666803` failed because the static test runner lacked NumPy, not because of backend logic.
- corrected GREEN run `41601753518`: backend + chunk-lineage static contracts passed; heavy job skipped.

BF16 benchmark:

- run: `41602080074`
- representation: `dct`
- backend: CPU bfloat16 + CPU autocast, no GradScaler
- max microbatches: `1`
- separate lineage fingerprint from FP16
- status at recording: **IN PROGRESS**
- official rows scored: `0`
- Public LB tuning: `false`
- submission authorization: `false`

## Decision rule

Do not mix or resume checkpoints across FP16/BF16 lineages.

Prefer BF16 for sustained CPU-shadow continuation only if the one-microbatch benchmark demonstrates materially better operational throughput while completing cleanly with finite loss and a valid checkpoint.

Neither backend can authorize the frozen CUDA v3 transfer gate, official OOF, or Kaggle submission.

Do not launch another FP16 or BF16 heavy benchmark while these two runs are unresolved.
