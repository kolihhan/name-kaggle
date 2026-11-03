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

The frozen v3 implementation now has two verified execution layers:

1. Original focused suite: GitHub run `41569836280` — **19 passed in 2.76s**.
2. Local/self-hosted CUDA executor suite: GitHub run `41580519705` — **24 passed in 3.62s**.

Verified in the second run:

- `scripts/dct_v3_local_transfer.py` imports and compiles;
- frozen transfer split / DCT pooling / worker materialization / source evaluator tests pass;
- materialized v3 GPU worker compiles;
- local executor preserves exact pool SHA, model revision, source lock, no official-row scoring, no Public-LB tuning, and no submission authorization.

## Execution backend change — infrastructure only

Kaggle weekly GPU quota is exhausted, so the scientific contract is **not** being changed. Only the execution backend changes.

GitHub-hosted free runners are CPU-only and cannot execute the frozen Qwen3-8B LoRA recipe. The prepared training route is therefore:

- GitHub Actions for orchestration/evidence;
- a dedicated self-hosted Linux x64 CUDA runner with custom label `gpu` for the actual model compute;
- same v1 control and same frozen v3 treatment;
- same external source holdout, seed, triplet order, optimizer, LoRA settings, and gate;
- no Kaggle GPU compute;
- Kaggle CLI is used only to obtain the already-frozen public external pool if it is not cached locally.

Workflow: `.github/workflows/v3-self-hosted-gpu-transfer.yml`.

Local executor: `scripts/dct_v3_local_transfer.py`.

## Current execution state

GitHub Actions run `41580819667` is **QUEUED** with job `transfer`.

Head: `2d0a86fffa880709ad25c8ee9153f2b2838617a8`.

The workflow is valid and has a real queued job. It is waiting for a matching runner:

`[self-hosted, linux, x64, gpu]`

No scientific model score has been produced yet, so:

- v3 transfer gate remains **UNRUN / UNKNOWN**;
- v3 is neither PASS nor FAIL;
- no v3 checkpoint exists from this self-hosted run yet;
- no official rows were scored;
- no Public LB was consumed;
- no competition submission occurred.

## Execution policy

- Do **not** mark v3 FAIL while the Actions job is waiting for compute.
- Do **not** change DCT modes, weights, layers, source lock, model, seed, thresholds, or training parameters.
- Do **not** substitute the GitHub-hosted CPU runner, a smaller model, QLoRA, or another approximation; those would change the frozen experiment.
- Bring online exactly one suitable self-hosted CUDA runner labelled `gpu`; the already-queued run should claim it.
- If transfer FAILs: stop v3 permanently; no official OOF, no Kaggle submission.
- If transfer PASSes: only then run the predeclared official five-fold OOF gate.
- Only if both transfer and official OOF gates PASS may v3 proceed to final fit and at most one competition-facing submission.

## Next single action

**Bring the dedicated Linux CUDA self-hosted GitHub Actions runner online with custom label `gpu`. Do not create a new scientific run. The already-queued run is `41580819667`.**
