# V3 CPU Shadow Backend Pivot

Date: 2026-11-03 MYT

## Scope

This document tracks CPU execution/training evidence only. It does not alter the frozen CUDA v3 scientific contract.

## User execution priority

On 2026-11-03 the user explicitly chose **CPU-first training**.

Operational consequence:

- make useful CPU training progress before requiring GPU;
- keep every CPU lane bounded and checkpointable;
- keep CPU evidence scientifically isolated from the frozen CUDA v3 transfer gate;
- do not score official competition rows, use Public LB, or submit from the CPU lane.

## Exact LoRA CPU evidence

### FP16 anchor

- run: `41598304633`
- representation: `dct`
- backend: CPU float16, no autocast, no GradScaler
- max microbatches: `1`
- official rows scored: `0`

### BF16 history

First BF16 run `41602080074` reached the readout and failed at the BF16 -> FP32 classifier boundary:

`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`

The forward/readout path reached that failure roughly ten seconds after the model had loaded, so the dominant multi-hour cost is not the forward pass.

The readout boundary was fixed by promoting representation features back to FP32 before the FP32 classifier heads.

Corrected BF16 run:

- run: `41604738508`
- representation: `dct`
- backend: CPU bfloat16 + CPU autocast, no GradScaler
- max microbatches: `1`
- status at the 2026-11-03 19:55 MYT check: **IN PROGRESS** in the first real bounded training microbatch after more than three hours
- artifact/checkpoint at that check: **none yet**
- official rows scored: `0`
- Public LB tuning: `false`
- submission authorization: `false`

### Backward root-cause evidence

The separate exact-architecture CPU feasibility probe `41581953208` entered the real backward path and was cancelled after roughly 1h27m without completing. At cancellation the runner reported about `11 GiB` RAM used and `1.5 GiB` swap used on a 4-vCPU / 15-GiB hosted runner.

Together with the earlier BF16 run reaching the classifier readout in seconds, this localizes the operational bottleneck to **backpropagation through the 24-layer Qwen backbone with LoRA**, not model download or ordinary forward inference.

## CPU-first decision after root-cause investigation

**CPU-FIRST remains active, but automatic continuation of the exact LoRA path is stopped.**

WHY:

- one exact LoRA microbatch has demonstrated multi-hour backward cost on the hosted CPU;
- automatically chaining another identical microbatch would spend hours without first changing the execution bottleneck;
- the user asked for CPU training first, not for blind repetition of a path that has not produced one checkpoint.

The one-shot workflow that would have automatically resumed exact BF16 LoRA from `41604738508` was removed in commit `193d3dbcbaae54a96c74de8cb25be629489e6b74`.

## Practical CPU training lane

The next CPU lane should train only the DCT readout path while the Qwen backbone is frozen:

- same pinned truncated Qwen backbone;
- same DCT representation definition and selected layers;
- backbone forward under inference/no-grad;
- trainable LayerNorm/readout heads only;
- training rows only for development; do **not** consume the frozen v3 heldout sources while the exact LoRA transfer gate is unresolved;
- separate lineage and explicit label such as `head-only CPU proxy`; never relabel it as v3 LoRA;
- bounded resumable chunks.

Purpose: make real CPU training progress and test the usefulness/behavior of the DCT representation without paying the prohibitive LoRA backward cost.

## Continuation rule

1. Let already-running exact BF16 `41604738508` finish or time out; do not auto-chain it.
2. If it unexpectedly produces a valid checkpoint, preserve it as exact-LoRA evidence but do not automatically launch another identical chunk.
3. Build and benchmark the frozen-backbone head-only DCT CPU proxy on training rows only.
4. Continue that proxy only if it demonstrates checkpointed progress at practical wall-clock cost.
5. Keep exact LoRA, head-only proxy, FP16, and BF16 checkpoints in separate lineages.
6. Do not use CPU proxy evidence to authorize the frozen CUDA v3 gate, official OOF, or submission.

## Pivot trigger

Reopen sustained exact-LoRA CPU continuation only if an execution change materially reduces backward cost while preserving the candidate and lineage. Otherwise CPU-first progress should come from the explicitly labeled frozen-backbone proxy rather than repeated multi-hour exact-LoRA microbatches.
