# NOW

Updated: 2026-11-03 MYT

## Competition anchor

- Submitted champion: `two-stage-lora24-v1` — Public LB `0.520`, submission `57908683`.
- v1 Gate-B primary plain OOF BA: `0.5467611235396471`.
- Previous Phase-7 Public LB: `0.369`.
- Do not rerun/retrain or resubmit identical v1.
- Public LB is external evidence only, not a tuning loop.

## Research doctrine

`RESEARCH_DOCTRINE.md` is binding on this branch.

Every major restriction must record `WHY / CURRENT EVIDENCE / PIVOT TRIGGER / STATUS`. Do not let a temporary restriction become a permanent method-class ban. Frozen contracts remain binding for already-running candidates.

## Rejected v2

`prompt-invariant-contrastive-v2` is permanently rejected under its frozen gate.

- v1 BA: `0.41027732627916647`
- v2 BA: `0.4129626992051072`
- delta: `+0.002685372925940732`
- 5,000 prompt bootstrap 95% CI: `[-0.0006323911599452988, +0.0069285212171021765]`
- decision: FAIL because the bootstrap lower bound was not positive.

Do not rescue v2 by tuning contrastive weight/temperature, changing its source holdout, or running official OOF.

## Active scientific candidate: DCT temporal fingerprint v3

Branch: `exp/dct-temporal-fingerprint-v3`.

Frozen scientific change only:

- control: exact v1 Stage-1 A1 masked-mean pooling;
- treatment: fixed two-mode length-normalized DCT-II `[DC0(mean), AC1]` at the same pooled layers;
- DCT modes fixed `[0,1]`;
- all other Stage-1 data/model/LoRA/optimizer/seed/epoch settings remain v1;
- no contrastive-v2, attention pooling, top-k, blend, router, second backbone, or Public-LB tuning.

Frozen contract: `artifacts/candidates/dct-temporal-fingerprint-v3/CONTRACT.md`.

Frozen transfer split:

- sources: `alperen`, `hailey`, `willrice`
- heldout rows: `14,506`
- training rows after prompt-bank leakage removal: `14,502`
- eligible labels: `17`
- prompt-id overlap: `0`
- prompt-key overlap: `0`
- model scores used for selection: `false`

Verified implementation:

- static CUDA-oriented suite `41569836280`: 19 passed;
- self-hosted executor suite `41580519705`: 24 passed.

## Authoritative CUDA v3 execution

Scientific run `41580819667`, job `125734978512`, remains QUEUED for labels:

`[self-hosted, linux, x64, gpu]`

Frozen run head: `2d0a86fffa880709ad25c8ee9153f2b2838617a8`.

No CUDA scientific score exists yet:

- v3 transfer gate: UNRUN / UNKNOWN;
- no official rows scored;
- no Public LB consumed;
- no competition submission;
- no replacement CUDA scientific run while `41580819667` remains queued.

`scripts/bootstrap_ephemeral_gpu_runner.sh` is statically verified by GREEN run `41590510236`. It requires Linux x64, CUDA/PyTorch, at least 15 GiB VRAM, a short-lived registration token, and custom label `gpu`.

## CPU shadow — diagnostic only, full training rejected

Branch: `infra/v3-cpu-shadow-resume`.

CPU remains useful only for bounded execution/mechanism diagnostics. It may not replace the frozen CUDA v3 transfer gate.

### FP16 anchor

Run `41598304633`:

- representation: DCT;
- backend: CPU float16, no autocast, no GradScaler;
- max microbatches: 1;
- latest status: still IN PROGRESS in the first real microbatch;
- no official rows, Public LB, or submission authorization.

Existing worst-case CPU feasibility probe `41581953208` is also still in progress and remains execution-only evidence.

### BF16 pivot

First BF16 run `41602080074` was not a training result. It reached the readout quickly, then failed at the BF16 -> FP32 linear boundary with a dtype mismatch and produced no checkpoint/result.

The readout boundary was fixed by promoting representation features to FP32 before the FP32 classifier heads. The fix is covered by `test_head_features_cross_autocast_boundary_as_float32` in `tests/test_dct_v3_cpu_shadow_chunk.py`.

Corrected BF16 benchmark:

- run `41604738508`;
- representation: DCT;
- backend: CPU bfloat16 + CPU autocast, no GradScaler;
- max microbatches: 1;
- separate lineage from FP16;
- latest status: IN PROGRESS in the real bounded microbatch;
- no official rows, Public LB, or submission authorization.

### Full-CPU viability decision

Sustained exact Stage-1 CPU training is REJECTED as an operational route.

WHY:

- Stage-1 transfer training has 14,502 rows;
- `build_prompt_triplets` creates microbatches of at most three rows;
- therefore one exact epoch requires at least `ceil(14502 / 3) = 4,834` forward/backward microbatches;
- the corrected BF16 benchmark has already required tens of minutes without finishing one microbatch, while FP16 is slower;
- even an optimistic hypothetical 10 minutes per microbatch implies about 33.6 days for one epoch before setup/reload/checkpoint overhead.

CURRENT EVIDENCE:

CPU can diagnose execution and mechanism behavior but is orders of magnitude too slow for sustained exact Stage-1 training on free hosted runners.

PIVOT TRIGGER TO REOPEN:

Only reopen full CPU training if a materially different exact-compatible execution method demonstrates sub-minute microbatches without changing the scientific candidate or violating equivalence/lineage requirements.

STATUS: REJECTED for full training; bounded diagnostics only.

Do not chain the existing FP16/BF16 runs into sustained CPU training.

## Parallel gradient restriction

Do not split one six-triplet accumulation group across independent runners yet. LoRA dropout + head dropout make naive parallel reduction trajectory-changing unless RNG equivalence is explicitly proven.

Pivot trigger: a dedicated equivalence test under a frozen per-microbatch RNG scheme.

## Validation-budget firewall after v3

Independent branch: `research/v1-mechanism-audit-2026-11-03`.

Frozen firewall run `41590073872`, artifact `13209726705`, digest `sha256:7de14265ac8490511b134a11832ab427d1e15a75077376b3d4e305a58ec2f190`.

Firewall sources:

`abhinav, dophuchung, kaplinskiy, keshav, prvsiyan, prvsiyan_v1, seyit, zbigniew`

Accounting:

- rows: 15,206;
- eligible rows: 14,013;
- eligible labels: 17/18;
- missing eligible label: `claude-fable-5`;
- leakage-clean development rows: 10,359;
- prompt-id/key overlap: 0.

Do not reselect firewall sources to recover the eighteenth label. The firewall is not retrofitted into v3.

## Fallback research — not active v4

If v3 fails, current strongest fallback hypothesis is centered-to-plain privileged-information distillation.

Mechanism audit:

- centered rescues 353 plain errors;
- centered breaks 350 plain-correct rows;
- centered is complementary, not universally stronger;
- same-prompt centering is privileged context unavailable at test inference.

Research primitive TDD:

- RED `41591144258`;
- GREEN `41591267999`.

This is not v4 yet and has consumed no firewall prediction, official OOF, or new Public-LB observation.

## v3 PASS path

Only authoritative exact CUDA transfer PASS may authorize official OOF.

Then:

1. preserve the exact passing v3 Stage-1 checkpoint lineage;
2. adapt the v1 Gate-B worker only for frozen DC0+AC1 representation;
3. keep v1 folds, Stage-2 epochs/LRs/optimizer/seed/class weighting/triplet isolation unchanged;
4. run five-fold official OOF once;
5. require plain BA > `0.5467611235396471`, at least 3/5 fold wins, positive 5,000-group-bootstrap lower bound, and class-recall safety veto;
6. only then final fit + at most one submission.

CPU shadow results cannot authorize this gate.

## Current execution policy

- Primary execution problem is now GPU availability, not model design.
- Keep authoritative CUDA run `41580819667` queued; do not duplicate it.
- The corrected BF16 run `41604738508` may finish only as a bounded throughput diagnostic; do not chain it.
- Do not launch additional FP16/BF16 full-training chunks.
- Do not change DCT modes/layers/source lock/model/seed from CPU diagnostics.
- Do not consume official OOF or Public LB while transfer remains unresolved.
- If authoritative CUDA v3 FAILs: stop v3 and reconsider architecture under `RESEARCH_DOCTRINE.md` before activating v4.
- If authoritative CUDA v3 PASSes: execute the predeclared official OOF gate.

## Next action

**Acquire/provide one exact-compatible GPU runner (Linux x64, CUDA/PyTorch, >=15 GiB VRAM, label `gpu`) so queued authoritative run `41580819667` can execute. Do not spend more effort trying to turn free hosted CPU into the full training path.**
