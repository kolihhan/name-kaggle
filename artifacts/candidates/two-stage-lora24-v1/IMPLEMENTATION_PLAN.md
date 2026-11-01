# Implementation Plan — two-stage-lora24-v1 Gate A

Contract: `CONTRACT.md`

Scope: **Gate A only. No official train rows, no official OOF, no submission.**

## Task 1 — lock the data/split contract with unit tests

Add `tests/test_two_stage_lora24_gate_a.py` covering:

- deterministic source-hash holdout selects exactly three sources;
- held-out-source rows never enter Stage-1 training;
- `prompt_id` and `prompt_key` leakage is removed from training;
- eligible-label thresholds are applied exactly as frozen;
- class weights are computed from training labels only and normalized to mean 1;
- same-prompt triplet construction never mixes prompts;
- PASS / FAIL / INCONCLUSIVE decision rules match the contract.

Run RED before production implementation.

## Task 2 — implement lightweight contract primitives

Add `scripts/run_two_stage_lora24_gate_a.py` with pure functions for:

- source hashing / selection;
- leakage-clean Stage-1 train vs held-out split;
- eligible label selection;
- class weights;
- deterministic same-prompt triplets;
- balanced-accuracy summaries and frozen Gate-A decision.

Heavy ML imports must remain lazy so the normal test suite stays lightweight.

Run GREEN and the full repo test suite.

## Task 3 — implement the two frozen Stage-1 conditions

In the same runner, implement:

- Qwen3-8B truncated to 24 layers;
- eight pooled hidden layers `[2,4,6,8,12,16,20,24]`;
- first-token-masked mean pooling;
- LayerNorm per pooled layer;
- plain + centered heads;
- A0 frozen backbone / train heads+norms only;
- A1 identical setup plus LoRA r16/a32/dropout .05 on the frozen target-module list;
- one epoch, frozen optimizer schedule and seed;
- class-weighted CE using Stage-1 training labels only.

Output one JSON containing data lineage, split identities/counts, A0/A1 metrics, prompt overlap checks, source/label diagnostics, and the frozen decision.

## Task 4 — execute on Kaggle GPU without official competition data

Add one candidate-specific workflow that:

- downloads/verifies the pinned public pool;
- rebuilds the exact Phase14 conservative subset;
- packages the candidate runner and dependencies;
- attaches the base Qwen3-8B model and public dataset only;
- explicitly attaches **no competition source**;
- runs Gate A once;
- verifies `official_rows_scored == 0` and `kaggle_submission_authorized == false`;
- uploads the JSON evidence.

Do not authorize a rerun because of a bad scientific result. Infra-only reruns are allowed if the model did not produce a valid result.

## Task 5 — consume the result

- PASS: record exact evidence and only then create a separate Gate-B implementation plan.
- FAIL: close `two-stage-lora24-v1`; do not run official OOF.
- INCONCLUSIVE: close v1 as inconclusive; do not mint another source split inside v1.

Update `NOW.md` with the consumed verdict.