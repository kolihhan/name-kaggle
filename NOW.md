# NOW

Updated: 2026-11-03 MYT

## Competition anchor

- **Current submitted champion: `two-stage-lora24-v1` — Public LB `0.520`, submission `57908683`.**
- Gate-B primary plain OOF BA: `0.5467611235396471`.
- Previous Phase-7 Public LB: `0.369`.
- Do not rerun/retrain or resubmit identical v1.
- Public LB is external evidence only, not a tuning loop.

## Rejected v2

Candidate: `prompt-invariant-contrastive-v2`.

Frozen paired source-transfer result:

- v1 BA: `0.41027732627916647`
- v2 BA: `0.4129626992051072`
- delta: `+0.002685372925940732`
- 5,000 prompt-group bootstrap 95% CI: `[-0.0006323911599452988, +0.0069285212171021765]`
- mean source delta: `+0.003746631276755267`
- negative sources: `0 / 3`
- class safety veto: none
- frozen gate decision: **FAIL** because the bootstrap lower bound was not positive.

Do not rescue v2 by tuning contrastive weight/temperature, changing source holdout, or running official OOF.

## Current candidate: DCT temporal fingerprint v3

Branch: `exp/dct-temporal-fingerprint-v3`.

Scientific change only:

- control: exact v1 Stage-1 A1 mean pooling;
- treatment: fixed two-mode length-normalized DCT-II `[DC0(mean), AC1]` at the same pooled layers;
- DCT modes fixed to `[0,1]`;
- all other Stage-1 data/model/LoRA/optimizer/seed/epoch settings remain v1;
- no contrastive-v2, attention pooling, top-k, blend, router, second backbone, or Public-LB tuning.

Frozen contract: `artifacts/candidates/dct-temporal-fingerprint-v3/CONTRACT.md`.

### Fresh source-transfer lock

Selection occurred before any v1-v3 score and explicitly excluded all v1/v2 consumed holdout sources.

- sources: `alperen`, `hailey`, `willrice`
- heldout rows: `14,506`
- eligible rows: `14,506`
- eligible labels: `17`
- training rows after prompt-bank leakage removal: `14,502`
- prompt-id overlap: `0`
- prompt-key overlap: `0`
- model scores used for selection: `false`
- split artifact: `13200551511`
- artifact digest: `sha256:6072697bbbcb8d9cd26431c4aa599c48e8de9a0e1ffafe4777f4bd4e55d92080`

Lock: `artifacts/candidates/dct-temporal-fingerprint-v3/TRANSFER_SPLIT_LOCK.json`.

### Implementation verification

Frozen v3 focused static suite completed successfully:

- GitHub run `41569836280`
- **19 passed in 2.76s**
- materialized v3 worker compiled
- frozen private internet-off T4 kernel compiled
- no competition submission command in kernel

A direct no-Actions runner now exists:

- `scripts/dct_v3_direct_transfer.py`
- `tests/test_dct_v3_direct_transfer.py`
- local TDD result: **5 passed**
- catches Kaggle GPU-quota push failures even when Kaggle CLI returns exit code `0`
- validates exact frozen lineage/result before reporting gate decision
- performs no competition submission

No new GitHub Actions runs were triggered by the direct-runner `[skip ci]` commits.

## Current blocker

**Kaggle weekly GPU quota is exhausted: `Maximum weekly GPU quota of 30.00 hours reached.`**

Attempted scientific workflow: `41569982605`.

The frozen kernel passed static verification, but Kaggle rejected the kernel push before execution. Therefore:

- v3 scientific transfer gate = **UNRUN / UNKNOWN**;
- no v1/v3 Stage-1 training ran in this attempt;
- no v3 predictions or checkpoint were produced;
- no official rows were scored;
- no Public LB was consumed;
- no competition submission occurred.

Exact status record: `artifacts/candidates/dct-temporal-fingerprint-v3/RESULT.md`.

## Execution policy

- Do **not** mark v3 FAIL because of the GPU-quota block.
- Do **not** redesign/tune v3 while blocked.
- Do **not** rerun GitHub Actions to probe quota; Actions should remain minimal/read-only.
- When Kaggle GPU execution is available, run the exact frozen experiment once via:

  `python scripts/dct_v3_direct_transfer.py --work-dir .v3-dct-transfer --repo-root .`

- If transfer FAILs: stop v3 permanently; no official OOF, no Kaggle submission.
- If transfer PASSes: only then implement/run the already-predeclared official five-fold OOF gate. A PASS does not automatically authorize a competition submission.
- Only if both transfer and official OOF gates PASS may v3 proceed to one frozen final-fit/preflight and at most one competition-facing submission.

## Repository visibility

The repository is still public under the user's temporary execution exception. It is **not public-safe by default** because competition files are tracked. The user requested returning it to private after the training/execution work is complete. Current surfaced GitHub connector permissions do not expose repository visibility mutation; do not claim it was changed.

## Next single action

**No further scientific tuning now. Preserve the frozen v3 state. The next executable scientific action is the exact direct v3 paired-transfer run when Kaggle GPU quota permits it.**
