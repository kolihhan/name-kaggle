# DCT v3 PASS-Path Readiness Audit

Date: 2026-11-03 MYT

## Verdict

**SCIENTIFIC CONTRACT READY; OFFICIAL-OOF IMPLEMENTATION NOT YET BUILT.**

This is intentional. The v3 contract states that official OOF is unauthorized until the frozen external source-transfer gate passes. The current branch contains the transfer worker/evaluator/execution path, but no v3-specific Gate-B worker or workflow.

## Existing reusable pieces

From v1:

- `scripts/run_two_stage_lora24_gate_b.py` — official fold splitting, frozen Stage-2 spec, evaluator/bootstrap/safety logic.
- `scripts/run_two_stage_lora24_gate_b_gpu.py` — checkpoint loading, fold-local Stage-2 training, OOF prediction mechanics.
- v1 Gate-B workflows — artifact/checkpoint/runtime orchestration patterns.

From v3:

- `scripts/dct_temporal_pool.py` — frozen DC0+AC1 pooling.
- `scripts/materialize_dct_v3_worker.py` — frozen Stage-1 representation transformation.
- `scripts/dct_v3_source_eval.py` — v3 transfer gate only; it must not be repurposed as official OOF evidence.

## Exact implementation boundary after transfer PASS

Only after the frozen v3 transfer result is PASS and its exact Stage-1 checkpoint lineage is preserved:

1. Create a v3 Gate-B GPU worker by adapting the v1 Gate-B worker.
2. Reuse v1 official fold assignments, Stage-2 epochs, LRs, optimizer, seed, class weighting, triplet construction, and fold-local isolation unchanged.
3. Change only representation/readout construction required by frozen v3:
   - DC0 + AC1 per selected layer;
   - same layer-specific LayerNorm reused for both modes;
   - doubled concatenated representation width;
   - plain and centered heads resized to that width.
4. Every fold must initialize from the exact transfer-PASS v3 Stage-1 checkpoint, never a retrained equivalent.
5. Produce row-aligned plain/centered OOF predictions for all 2,699 official rows.
6. Apply the already-frozen v3 Gate-B rules from `CONTRACT.md`:
   - plain BA > v1 `0.5467611235396471`;
   - win at least 3/5 folds;
   - 5,000 prompt-group bootstrap seed `20261101`, lower CI > 0;
   - preserve the v1 class-recall safety veto.
7. No Public LB, firewall evidence, new split, blend, router, DCT-mode change, or short-text post-hoc rule enters v3.

## Files to create only after transfer PASS

- `scripts/run_dct_v3_gate_b.py`
- `scripts/run_dct_v3_gate_b_gpu.py`
- `tests/test_dct_v3_gate_b.py`
- `tests/test_dct_v3_gate_b_gpu_contract.py`
- `.github/workflows/v3-gate-b.yml` or equivalent self-hosted GPU orchestration
- `artifacts/candidates/dct-temporal-fingerprint-v3/GATE_B_RESULT.md` after execution

## Pre-execution checks

Before any official row is scored:

- transfer result must say PASS;
- exact v3 Stage-1 checkpoint artifact/digest must be recorded;
- focused static tests must pass on public GitHub Actions;
- workflow must contain no competition submission command;
- official fold fingerprint must equal the existing frozen mapping;
- no Phase19 shadow fold may be used as fresh evidence.

## Current blocker

Scientific transfer run `41580819667` is still queued for `[self-hosted, linux, x64, gpu]`. Do not create a second v3 scientific transfer execution while it remains queued.
