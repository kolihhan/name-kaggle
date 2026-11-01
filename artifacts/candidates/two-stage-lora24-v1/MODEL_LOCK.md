# two-stage-lora24-v1 — Model / Runtime Lock

Status: **FROZEN BEFORE GATE A MODEL EXECUTION**

This file fixes the model identity and the two Python libraries whose behavior is directly used by the Gate A LoRA implementation. It does not authorize model execution, official OOF, leaderboard tuning, or submission.

## Base model

- Hugging Face repository: `Qwen/Qwen3-8B-Base`
- pinned revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- expected architecture: `qwen3`
- expected hidden size: `4096`
- upstream model has 36 layers; this candidate loads only the first 24 as frozen in `CONTRACT.md`

The runner must load the pinned revision, not floating `main`.

## Runtime libraries

- `transformers==4.57.1`
- `peft==0.17.1`

The GPU runner asserts these versions at runtime. The Kaggle-provided CUDA/PyTorch build is not replaced; the exact observed PyTorch version is written into the Gate A result JSON.

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
- metadata containing this base-model revision and the frozen Stage-1 specification

Retraining an equivalent-looking Stage-1 checkpoint later is not an admissible substitute.

## Evidence used to lock this file

Before Gate A execution, the public Qwen repository identified the base checkpoint and its Qwen3 configuration, while the audited public LoRA24 implementation established the first-24-layer recipe. No Gate A metric, official OOF score, competition test output, or leaderboard score was observed when this lock was written.
