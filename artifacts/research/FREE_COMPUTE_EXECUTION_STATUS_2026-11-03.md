# Free Compute Execution Status — 2026-11-03

Scope: infrastructure/resource reconnaissance only. No v3 transfer score, official OOF, final-firewall score, Public-LB tuning, or submission was consumed by this work.

## Decision

Resource priority is frozen as:

1. **Kaggle TPU v5e-8 feasibility** — primary free-compute path.
2. **Original queued self-hosted GPU v3 scientific run** — remains valid and untouched.
3. **Homogeneous ARM GitHub-hosted CPU exact-resume** — emergency engineering fallback only; do not launch a massive ML training chain on Actions.

## Kaggle quota

Read-only `kaggle quota` on 2026-11-03 showed:

- GPU: `31.79h used / 0.00h remaining / 30.00h nominal total`;
- TPU: `0.00h used / 20.00h remaining / 20.00h total`;
- quota refresh: `2026-11-07T00:00:00+00:00`.

No TPU time had been consumed at the last verified status read.

## TPU Gate 0

Branch: `infra/v3-tpu-feasibility`.

Design: `docs/superpowers/specs/2026-11-03-v3-tpu-feasibility-design.md`.

The environment-only probe:

- requests Kaggle `TpuV5E8`;
- is private and offline;
- attaches no competition, dataset, kernel, or model sources;
- imports `torch_xla`, checks the actual XLA runtime device, performs one trivial tensor round-trip;
- writes `environment_probe.json`;
- PASS requires runtime `device_type == "TPU"`;
- loads no Qwen model and performs no scientific scoring.

TDD:

- RED: GitHub run `41595420974` — missing builder module.
- GREEN: GitHub run `41595593150` — static probe builder tests passed.

Kaggle push occurred once through GitHub run `41595649658`.

Important orchestration finding: Kaggle created the actual notebook slug as:

`kolihhan/ntl-v3-tpu-environment-probe`

rather than the requested metadata id suffix `ntl-v3-tpu-env-probe`. The original poll therefore queried the wrong slug; that is an orchestration bug, not a TPU result.

Verified actual status reads:

- around 07:26 UTC: `KernelWorkerStatus.QUEUED`;
- around 07:28 UTC: `KernelWorkerStatus.QUEUED`;
- no output files or logs yet;
- TPU quota remained `0.00h used / 20.00h remaining`.

Therefore Gate 0 is **PENDING / UNRUN**, not PASS and not FAIL.

Do not build or launch the Qwen TPU Gate-1 probe until Gate 0 proves that Kaggle actually allocated a TPU runtime.

## TPU compatibility boundary

Current external documentation indicates that modern Transformers supports TPU execution, but TPU training integrations are still evolving. TPU-native mixed precision is BF16 rather than CUDA FP16.

If Gate 0 PASSes:

- Gate 1 should first use a **single XLA device**, not introduce tensor-parallel LoRA;
- keep the pinned Qwen3-8B model identity, first 24 layers, LoRA recipe, DCT `[DC0, AC1]`, sequence length and optimizer semantics as close as the TPU backend permits;
- run one realistic `3 x 448` triplet forward/backward/optimizer feasibility probe only;
- perform no scoring;
- only after Gate 1 PASS may a separate TPU scientific-backend amendment be frozen;
- any eventual scientific TPU comparison must train **both v1 and v3 on the same TPU/BF16 backend**. Never compare CUDA-trained v1 against TPU-trained v3 for the transfer decision.

## ARM CPU exact-resume evidence

Branch: `infra/v3-exact-cpu-resume`.

Exact state work proved that partial gradient accumulation can be serialized and resumed across jobs, including trainable parameters, AdamW state, LR scheduler, RNG, partial `.grad`, cursor/accumulation position, and CPU GradScaler state.

Key proof runs:

- `41592624451`: exact resume state tests GREEN;
- `41592865376`: uninterrupted vs mid-accumulation-resumed training exactly equal on synthetic dropout/AdamW test;
- `41593562345`: state artifact moved across two distinct fresh GitHub-hosted runners and resumed with `all_equal=true`;
- `41595795865`: homogeneous `ubuntu-24.04-arm` rolling cache chain fully GREEN: producer, fresh-runner resume, and cache cleanup all succeeded.

Thus ARM cross-job resume is **technically viable**.

However, real-Qwen ARM feasibility run `41593868793` began its one worst-case exact-architecture triplet at `2026-11-03T07:05:32Z` and was still in that single forward/backward probe at the latest checked status. This makes ARM CPU operationally unattractive as the primary training route.

Do not launch a thousands-of-jobs training chain on GitHub Actions. Keep Actions for CI, bounded feasibility probes, orchestration, and exact-resume verification.

## Original GPU path

The original frozen scientific run remains untouched:

- workflow run: `41580819667`;
- job: `125734978512`;
- frozen head: `2d0a86fffa880709ad25c8ee9153f2b2838617a8`;
- labels: `[self-hosted, linux, x64, gpu]`;
- latest verified state: `queued` with no runner assigned.

Do not create a second GPU scientific transfer run.

## Next action

Wait only for the already-submitted Kaggle environment probe to leave `QUEUED`. Read the actual runtime identity from `environment_probe.json`.

- Actual TPU -> Gate 0 PASS -> implement/run one zero-scoring Qwen Gate-1 TPU feasibility probe.
- CPU / missing XLA -> Gate 0 FAIL -> do not spend TPU quota on Qwen through the same push path.
- Still queued -> no scientific action; leave all validation evidence untouched.
