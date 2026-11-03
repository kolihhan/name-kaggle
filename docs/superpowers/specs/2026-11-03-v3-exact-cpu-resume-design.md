# V3 Exact CPU Resume Design

## Intent

Provide a zero-GPU execution path for the already-frozen `dct-temporal-fingerprint-v3` experiment by splitting exact training across many free GitHub-hosted CPU jobs without changing the scientific recipe.

## Goals

- Preserve the frozen v3 model, data, triplet order, gradient accumulation, optimizer, scheduler, seed, and scoring contract.
- Allow a job to stop between microbatches and continue in a fresh VM without silently changing the next optimizer step.
- Preserve partial gradient accumulation exactly; a resume after microbatch 1..5 of an accumulation group must not discard those gradients.
- Produce auditable state manifests with lineage and cursor information.
- Keep the existing queued GPU scientific run untouched.

## Non-goals

- No smaller model, QLoRA, fewer layers, shorter sequence length, altered accumulation, altered optimizer, or changed source split.
- No official OOF, firewall prediction, Public-LB tuning, or submission.
- No 6-way distributed gradient merge until serial cross-job equivalence is proven.

## Constraints

- Base branch lineage: `exp/dct-temporal-fingerprint-v3`.
- Candidate remains `dct-temporal-fingerprint-v3`.
- Frozen accumulation remains six same-prompt microbatches per optimizer step.
- GitHub-hosted CPU jobs must checkpoint well before the hosted-job hard timeout.
- State artifacts must contain no competition test predictions or official-row scores.

## Approaches

### A. Checkpoint only at optimizer-step boundaries

Simple, but unusable if one accumulation group cannot reliably finish inside a job. It also wastes completed microbatches when a timeout happens mid-group.

### B. Exact microbatch resume with accumulated-gradient persistence — selected

Persist trainable parameters, optimizer/scheduler state, RNG state, epoch/cursor, accumulation position, and every currently accumulated trainable gradient. On restore, reconstruct the same training state and continue with the next microbatch. This preserves the semantics of six-microbatch gradient accumulation across VM boundaries.

### C. Parallel six-way gradient sharding

Potentially much faster, but adds reduction-order and serialization complexity. Defer until B passes an exact equivalence test.

## Architecture

`scripts/exact_cpu_resume_state.py` owns serialization and restoration of resumable training state. The format is intentionally model-agnostic so it can be proven on a tiny deterministic PyTorch model before being connected to Qwen.

`scripts/dct_v3_cpu_chunk.py` will later wrap the frozen v3 worker and execute a bounded number of microbatches per invocation. It must stop only at an explicit cursor, save state, and never make scientific decisions itself.

A GitHub Actions workflow will download the prior state artifact if present, run one bounded chunk, then upload the successor state. Chunks are infrastructure execution only; scoring occurs only after the full frozen training path completes.

## Resume state contract

The checkpoint must record:

- schema version and candidate lineage;
- model/trainable module state;
- optimizer state;
- scheduler state when present;
- Python, NumPy, and Torch RNG state;
- epoch index;
- deterministic microbatch cursor;
- accumulation position `0..5`;
- gradients for every trainable parameter whose `.grad` is non-null;
- metadata proving no official rows, competition test rows, Public LB, or firewall evidence were consumed.

Restore must reject missing parameters, shape/dtype mismatches, lineage drift, invalid accumulation positions, or contaminated metadata.

## Proof requirement

Before touching the real Qwen worker, a deterministic synthetic test must compare:

1. uninterrupted training across multiple microbatches and optimizer steps;
2. the same sequence interrupted mid-accumulation, serialized, reconstructed in new objects, restored, and continued.

Final trainable parameters, optimizer tensors, scheduler state, cursor, accumulation position, and next RNG draw must match exactly (or bitwise where PyTorch representation allows it). A checkpoint that only matches loss approximately is insufficient.

## Safety

The CPU route must never cancel or replace queued GPU run `41580819667`. If GPU finishes first, CPU infrastructure may be abandoned without scientific consequence. Only a complete frozen-path execution may produce a v3 transfer result.
