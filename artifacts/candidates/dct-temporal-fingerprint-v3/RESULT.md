# dct-temporal-fingerprint-v3 — Result / Execution Status

Date: 2026-11-03 MYT

## Status

**BLOCKED_BY_KAGGLE_GPU_QUOTA — SCIENTIFIC TRANSFER GATE UNRUN.**

This is **not** a scientific FAIL. The frozen v1-vs-v3 paired source-transfer experiment has not produced any model score.

## Frozen experiment identity

- candidate: `dct-temporal-fingerprint-v3`
- contract: `artifacts/candidates/dct-temporal-fingerprint-v3/CONTRACT.md`
- transfer split: `artifacts/candidates/dct-temporal-fingerprint-v3/TRANSFER_SPLIT_LOCK.json`
- pool SHA-256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`
- holdout sources: `alperen`, `hailey`, `willrice`
- heldout rows: `14,506`
- eligible heldout rows: `14,506`
- eligible labels: `17`
- train rows after prompt-bank leakage removal: `14,502`
- prompt-id overlap: `0`
- prompt-key overlap: `0`
- split selection used model scores: `false`
- split artifact: `13200551511`
- split artifact digest: `sha256:6072697bbbcb8d9cd26431c4aa599c48e8de9a0e1ffafe4777f4bd4e55d92080`

## Implementation verification

GitHub Actions run `41569836280` (`V3 Static Tests`) completed successfully before the scientific execution attempt.

Focused suite:

- `tests/test_dct_v3_transfer_split.py`
- `tests/test_dct_temporal_pool.py`
- `tests/test_materialize_dct_v3_worker.py`
- `tests/test_dct_v3_source_eval.py`
- `tests/test_build_dct_v3_transfer_kernel.py`

Result: **19 passed in 2.76s**.

Additional static verification:

- materialized v3 Stage-1 worker compiled successfully;
- frozen paired-transfer kernel compiled successfully;
- kernel metadata was verified as private, internet-disabled, T4, with no competition source and no submission command;
- frozen holdout sources were embedded exactly as `['alperen', 'hailey', 'willrice']`.

## Execution attempt

Workflow run: `41569982605` (`V3 DCT Paired Transfer`)

The workflow completed all static checks and reached the one intended Kaggle kernel push. Kaggle rejected the push immediately with:

`Maximum weekly GPU quota of 30.00 hours reached.`

Because the push was rejected:

- no Kaggle v3 scientific kernel execution was created;
- no v1 or v3 Stage-1 training ran;
- no heldout predictions were produced;
- no bootstrap interval was computed;
- no v3 Stage-1 checkpoint exists from this attempt;
- no official competition rows were scored;
- no competition test rows were used;
- no Public LB result was consumed;
- no Kaggle submission occurred.

The following status lookup returned 404 only because the scientific kernel did not exist after the rejected push. It is not an experiment failure.

## Scientific decision

**UNRUN / UNKNOWN.**

Do not mark v3 PASS or FAIL until the exact frozen paired source-transfer execution completes and `dct_v3_source_eval.py` applies the predeclared gate.

## Resume rule

When GPU quota is available again, execute the **same frozen experiment once**:

- same branch/contract;
- same transfer lock;
- same v1 control;
- same two-mode DCT treatment;
- same seed/triplet order/hyperparameters;
- same source gate and bootstrap thresholds.

Do **not** change modes, weights, layers, sources, model, training settings, thresholds, or label subsets in response to this infrastructure block.

Do not rerun the existing GitHub Actions workflow merely to probe quota. Prefer direct authenticated Kaggle execution when available; if orchestration is used later, it must push exactly this frozen kernel once and then attach to that execution rather than retraining on polling/recovery failures.
