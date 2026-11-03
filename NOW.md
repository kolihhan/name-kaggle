# NOW

Updated: 2026-11-03 MYT

## Competition anchor

- **Submitted champion:** `two-stage-lora24-v1` — Public LB `0.520`, submission `57908683`.
- v1 Gate-B primary plain OOF BA: `0.5467611235396471`.
- Previous Phase-7 Public LB: `0.369`.
- Do not rerun/retrain or resubmit identical v1.
- Public LB is external evidence only, not a tuning loop.

## Research doctrine

`RESEARCH_DOCTRINE.md` is now present on this active branch and `AGENTS.md` requires it before major research decisions.

Durable rule:

> Every major restriction must have `WHY / CURRENT EVIDENCE / PIVOT TRIGGER / STATUS`.

Do not let a temporary restriction become a permanent method-class ban. Frozen contracts remain binding for already-running candidates.

## Rejected v2

`prompt-invariant-contrastive-v2` is permanently rejected under its frozen gate.

- v1 BA: `0.41027732627916647`
- v2 BA: `0.4129626992051072`
- delta: `+0.002685372925940732`
- 5,000 prompt bootstrap 95% CI: `[-0.0006323911599452988, +0.0069285212171021765]`
- mean source delta: `+0.003746631276755267`
- negative sources: `0 / 3`
- decision: **FAIL** because bootstrap lower bound was not positive.

Do not rescue v2 by tuning its contrastive weight/temperature, changing its source holdout, or running official OOF.

## Active scientific candidate: DCT temporal fingerprint v3

Branch: `exp/dct-temporal-fingerprint-v3`.

Frozen scientific change only:

- control: exact v1 Stage-1 A1 masked-mean pooling;
- treatment: fixed two-mode length-normalized DCT-II `[DC0(mean), AC1]` at the same pooled layers;
- DCT modes fixed `[0,1]`;
- all other Stage-1 data/model/LoRA/optimizer/seed/epoch settings remain v1;
- no contrastive-v2, attention pooling, top-k, blend, router, second backbone, or Public-LB tuning.

Frozen contract: `artifacts/candidates/dct-temporal-fingerprint-v3/CONTRACT.md`.

### Frozen fresh transfer split

- sources: `alperen`, `hailey`, `willrice`
- heldout rows: `14,506`
- eligible rows: `14,506`
- eligible labels: `17`
- training rows after prompt-bank leakage removal: `14,502`
- prompt-id overlap: `0`
- prompt-key overlap: `0`
- model scores used for selection: `false`
- split artifact: `13200551511`
- digest: `sha256:6072697bbbcb8d9cd26431c4aa599c48e8de9a0e1ffafe4777f4bd4e55d92080`

Lock: `artifacts/candidates/dct-temporal-fingerprint-v3/TRANSFER_SPLIT_LOCK.json`.

### Verified implementation

- run `41569836280`: **19 passed**.
- self-hosted executor run `41580519705`: **24 passed**.
- frozen transfer split, DCT pooling, worker materialization, local executor and source evaluator are verified.

## Authoritative CUDA v3 execution

Scientific run `41580819667`, job `125734978512`, remains **QUEUED**.

Frozen run head: `2d0a86fffa880709ad25c8ee9153f2b2838617a8`.

It waits for:

`[self-hosted, linux, x64, gpu]`

No CUDA scientific score exists yet:

- v3 transfer gate: **UNRUN / UNKNOWN**;
- no official rows scored;
- no Public LB consumed;
- no competition submission;
- no replacement CUDA run is allowed while `41580819667` remains queued.

The helper `scripts/bootstrap_ephemeral_gpu_runner.sh` is statically verified by GREEN run `41590510236`. It requires Linux x64, CUDA/PyTorch, >=15 GiB VRAM, a short-lived registration token, and custom label `gpu`.

## CPU resource restriction — REOPENED as shadow evidence

Previous restriction:

`CPU is operationally non-viable; do not use it.`

### Updated doctrine record

**RESTRICTION:** CPU results may not replace the frozen CUDA v3 gate.

**WHY:** frozen v3 uses CUDA fp16 autocast + GradScaler. CPU changes numerical execution and cannot be assumed trajectory-identical.

**CURRENT EVIDENCE:** existing exact-architecture CPU probe `41581953208` shows a worst-case `3 x 448` triplet is extremely slow; it is still execution evidence only.

**PIVOT TRIGGER:** resource scarcity makes slow free progress preferable to indefinite waiting, provided CPU is explicitly isolated as a shadow lineage and does not consume official OOF/Public LB.

**STATUS:** CPU method class **REOPENED FOR SHADOW EVIDENCE**; restriction against treating it as authoritative v3 remains **ACTIVE**.

### Resumable CPU shadow lane

Branch:

`infra/v3-cpu-shadow-resume`

Key files:

- `docs/superpowers/specs/2026-11-03-v3-cpu-shadow-resume-design.md`
- `docs/superpowers/plans/2026-11-03-v3-cpu-shadow-resume.md`
- `scripts/cpu_shadow_resume.py`
- `scripts/dct_v3_cpu_shadow_chunk.py`
- `tests/test_cpu_shadow_resume.py`
- `tests/test_dct_v3_cpu_shadow_chunk.py`
- `.github/workflows/v3-cpu-shadow-chunk.yml`
- `artifacts/research/V3_CPU_SHADOW_RESUME_STATUS.md`

Verified locally before the heavy run:

- resume kernel RED first: module missing;
- resume/chunk suite GREEN: **9 passed**;
- both scripts compile;
- trainable-only checkpointing preserves optimizer/scheduler/RNG/partial gradients/progress;
- a toy interrupted CPU trajectory resumes bitwise-identically to uninterrupted CPU execution.

The CPU shadow checkpoint intentionally does **not** copy frozen Qwen base weights.

### Current first real CPU shadow chunk

Run: `41598304633`.

Frozen for this run:

- representation: `dct`;
- max microbatches: `1`;
- backend: `cpu_float16_no_autocast_no_gradscaler`;
- same external Phase14 safe subset;
- same v3 transfer split/triplet construction;
- no official rows;
- no Public LB;
- no submission authorization.

Latest verified progress:

- runner acquired: success;
- official competition files removed from runner workspace: success;
- pinned CPU runtime installed: success;
- 9 CPU shadow infrastructure tests: success;
- immutable Phase14 external subset downloaded: success;
- pinned truncated Qwen snapshot download: in progress at last check.

Do not launch another CPU shadow chunk until this first one resolves.

### CPU bfloat16 pivot

CPU fp16 is not a permanent doctrine restriction.

If the current fp16 probe/chunk times out or is operationally absurd, the next smallest experiment is a **no-scoring fp16-vs-bfloat16 execution benchmark** under the same architecture/input. CPU bfloat16 gets a separate lineage and may be promoted for shadow execution only if materially faster.

Do not switch backend in the middle of run `41598304633`.

### Parallel gradient restriction

Do **not** yet split one six-triplet accumulation group across independent runners. LoRA dropout + head dropout mean naïve parallel workers may consume different RNG streams and change gradients.

Pivot trigger: prove equivalence under an explicitly frozen per-microbatch RNG scheme before parallel reduction is allowed.

## Validation-budget firewall after v3

Independent branch:

`research/v1-mechanism-audit-2026-11-03`

Frozen firewall run `41590073872`, artifact `13209726705`, digest `sha256:7de14265ac8490511b134a11832ab427d1e15a75077376b3d4e305a58ec2f190`.

Firewall sources:

- `abhinav`
- `dophuchung`
- `kaplinskiy`
- `keshav`
- `prvsiyan`
- `prvsiyan_v1`
- `seyit`
- `zbigniew`

Accounting:

- rows: `15,206`
- eligible rows: `14,013`
- eligible labels: `17 / 18`
- missing eligible label: `claude-fable-5`
- leakage-clean development rows: `10,359`
- prompt-id overlap: `0`
- prompt-key overlap: `0`

Do not reselect firewall sources to recover the eighteenth label. The firewall is not retrofitted into v3.

## Fallback research — not active v4

If v3 fails, current strongest fallback hypothesis is **centered-to-plain privileged-information distillation**.

Mechanism audit:

- centered rescues `353` plain errors;
- centered breaks `350` plain-correct rows;
- therefore centered is complementary, not a universally stronger teacher;
- same-prompt centering is privileged context unavailable at test inference.

Research branch contains the dossier and selective imperfect-teacher distillation primitive. TDD:

- RED `41591144258`;
- GREEN `41591267999`.

This is **not v4 yet** and has consumed no firewall prediction, official OOF, or new Public-LB observation.

## v3 PASS path

Only authoritative CUDA transfer PASS may authorize the predeclared v3 official OOF path.

Then:

1. preserve exact passing v3 Stage-1 checkpoint lineage;
2. adapt v1 Gate-B worker only for frozen DC0+AC1 representation;
3. keep v1 folds, Stage-2 epochs/LRs/optimizer/seed/class weighting/triplet isolation unchanged;
4. run five-fold official OOF once;
5. require plain BA > `0.5467611235396471`, >=3/5 fold wins, positive 5,000-group-bootstrap lower bound, and class-recall safety veto;
6. only then final fit + at most one submission.

CPU shadow PASS cannot authorize this gate.

## Current execution policy

- Keep authoritative CUDA run `41580819667` queued; do not duplicate it.
- Let first CPU shadow run `41598304633` resolve; do not duplicate it.
- CPU shadow is supporting evidence only.
- Do not change DCT modes/layers/source lock/model/seed from shadow results.
- If CPU-fp16 is too slow, benchmark bfloat16 without scoring before changing shadow backend.
- Do not parallelize gradient microbatches until RNG equivalence is proven.
- Do not consume official OOF or Public LB while transfer remains unresolved.
- If authoritative CUDA v3 FAILs: stop v3; then reconsider architecture under `RESEARCH_DOCTRINE.md` before activating v4.
- If authoritative CUDA v3 PASSes: execute the predeclared official OOF gate.

## Next action

**Wait only on the two already-running compute lanes: authoritative CUDA run `41580819667` for a matching GPU, and bounded CPU-shadow run `41598304633` for its first real microbatch. Do not launch duplicates. Inspect and act on whichever produces evidence first.**
