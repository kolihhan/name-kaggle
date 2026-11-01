# two-stage-lora24-v1 — Model / Runtime Lock

Status: **FROZEN BEFORE GATE A MODEL EXECUTION**

This file fixes the model identity and the runtime details that directly affect the Gate A LoRA implementation. It does not authorize model execution, official OOF, leaderboard tuning, or submission.

## Base model

- Hugging Face repository: `Qwen/Qwen3-8B-Base`
- pinned revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- expected architecture: `qwen3`
- expected hidden size: `4096`
- upstream model has 36 layers; this candidate loads only the first 24 as frozen in `CONTRACT.md`

The runner must load the pinned revision, not floating `main`.

The pinned Qwen model's **stock final normalization is retained**. The Gate A runner must not replace the model's final norm with `Identity` or otherwise introduce an undocumented representation mutation. The candidate already applies the explicitly frozen trainable LayerNorm to each selected pooled hidden layer.

## Runtime libraries

- `transformers==4.57.1`
- `peft==0.17.1`

The GPU runner asserts these versions at runtime. The Kaggle-provided CUDA/PyTorch build is not replaced; the exact observed PyTorch version is written into the Gate A result JSON.

## Exact one-epoch accumulation

The frozen Stage-1 corpus is consumed exactly once per condition:

- deterministic same-prompt microbatches are built once from the leakage-clean Stage-1 training rows;
- microbatches are grouped into optimizer steps of at most six triplets;
- the final optimizer step may contain fewer than six triplets;
- no triplet may be duplicated merely to fill the final accumulation group;
- the loss inside each optimizer step is averaged over the actual number of triplets in that group.

This is the implementation of the contract's `one epoch` requirement. Randomly padding the epoch to a multiple of six is not admissible.

## Scientific scope

Gate A remains external-only:

- official competition rows scored: `0`
- official OOF runs: `0`
- competition test rows used: `0`
- Public LB tuning: `false`
- Kaggle submission authorized: `false`

A valid Gate A run may only compare the frozen `A0` and `A1` conditions from `CONTRACT.md` on the predeclared public-source holdout.

## Checkpoint lineage

If and only if Gate A returns `PASS`, Gate B must start from the exact `A1` checkpoint produced by that Gate A execution:

- PEFT adapter
- tokenizer files
- trainable per-layer LayerNorm weights
- plain head weights
- centered head weights
- metadata containing this base-model revision, the frozen Stage-1 specification, and the implementation invariants above

Retraining an equivalent-looking Stage-1 checkpoint later is not an admissible substitute.

## Evidence used to lock this file

Before Gate A execution, the public Qwen repository identified the base checkpoint and its Qwen3 configuration, while the audited public LoRA24 implementation established the first-24-layer recipe. The public evidence did not establish an alternate final-normalization mutation, so v1 retains the pinned model's stock behavior rather than inventing one. No Gate A metric, official OOF score, competition test output, or leaderboard score was observed when this lock was written.
