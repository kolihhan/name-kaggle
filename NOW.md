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

The frozen v3 implementation has two verified execution layers:

1. Original focused suite: GitHub run `41569836280` — **19 passed**.
2. Local/self-hosted CUDA executor suite: GitHub run `41580519705` — **24 passed**.

Verified:

- `scripts/dct_v3_local_transfer.py` imports and compiles;
- frozen transfer split / DCT pooling / worker materialization / source evaluator tests pass;
- materialized v3 GPU worker compiles;
- local executor preserves exact pool SHA, model revision, source lock, no official-row scoring, no Public-LB tuning, and no submission authorization.

## Current scientific execution state

GitHub Actions run `41580819667` is still **QUEUED** with job `transfer`.

Frozen run head: `2d0a86fffa880709ad25c8ee9153f2b2838617a8`.

It is waiting for a matching runner:

`[self-hosted, linux, x64, gpu]`

No scientific score exists yet:

- v3 transfer gate: **UNRUN / UNKNOWN**;
- v3 is neither PASS nor FAIL;
- no v3 Stage-1 checkpoint from this run exists yet;
- no official rows were scored;
- no Public LB was consumed;
- no competition submission occurred.

Do not create a second v3 scientific transfer execution while this run is queued.

## GPU runner path — infrastructure only

Kaggle weekly GPU quota is exhausted. The scientific contract remains unchanged; only execution infrastructure is being prepared.

Workflow: `.github/workflows/v3-self-hosted-gpu-transfer.yml`.

A one-job public-repo-safe helper now exists:

`scripts/bootstrap_ephemeral_gpu_runner.sh`

It requires:

- Linux x86_64;
- working CUDA/PyTorch;
- at least 15 GiB VRAM;
- a short-lived GitHub runner registration token supplied only through `RUNNER_TOKEN`;
- custom label `gpu`.

It registers the runner with `--ephemeral`, so the runner accepts one matching job and then deregisters. Do not keep a long-lived self-hosted runner attached to this public repository.

Static bootstrap verification followed TDD:

- RED: run `41590447418` before the helper existed;
- GREEN: run `41590510236`, bootstrap contract passed.

This infrastructure work does not alter the queued scientific run or its frozen head.

## Free GitHub-hosted CPU path

Run `41581953208` is an execution-only feasibility probe. It performs **one exact-architecture CPU backward step**, not a model-selection experiment.

At the latest verified check, that single backward probe had remained in progress for more than 90 minutes. Therefore GitHub-hosted free CPU is treated as **operationally non-viable for the full paired v1-vs-v3 transfer experiment**, even if the single-step probe eventually finishes.

Policy:

- do not launch a CPU scientific approximation;
- do not shrink the model, switch to QLoRA, reduce layers, or otherwise change the frozen v3 experiment to fit free CPU;
- retain the CPU probe only as infrastructure feasibility evidence.

## Validation-budget firewall after v3

Independent research branch:

`research/v1-mechanism-audit-2026-11-03`

A final source-shift firewall was frozen and verified by GitHub run `41590073872`, evidence artifact `13209726705`, digest `sha256:7de14265ac8490511b134a11832ab427d1e15a75077376b3d4e305a58ec2f190`.

The firewall is every remaining admissible Phase14 source not already consumed as fresh v1/v2/v3 transfer evidence:

- `abhinav`
- `dophuchung`
- `kaplinskiy`
- `keshav`
- `prvsiyan`
- `prvsiyan_v1`
- `seyit`
- `zbigniew`

Accounting:

- firewall rows: `15,206`;
- eligible firewall rows: `14,013`;
- eligible labels: `17 / 18`;
- missing eligible label: `claude-fable-5`;
- leakage-clean development rows: `10,359`;
- prompt-id overlap after cleanup: `0`;
- prompt-key overlap after cleanup: `0`.

Do **not** now change firewall source membership to recover the eighteenth label. That would be post-hoc validation design.

This firewall does not modify v3 and must not replace v3's already-frozen `alperen / hailey / willrice` transfer gate.

## Fallback research — not an active candidate

If v3 fails, the strongest currently supported fallback hypothesis is **centered-to-plain privileged-information distillation**.

Rationale from the verified v1 mechanism audit:

- centered view rescues `353` plain errors;
- centered view breaks `350` plain-correct rows;
- centered is therefore complementary, not a universally stronger teacher;
- same-prompt centering is privileged context unavailable at competition inference.

Research-only work on branch `research/v1-mechanism-audit-2026-11-03` now contains:

- `artifacts/research/PRIVILEGED_CENTERED_DISTILLATION_DOSSIER.md`;
- `scripts/research_privileged_distillation_loss.py`;
- synthetic-only tests for selective imperfect-teacher distillation.

TDD verification:

- RED: run `41591144258` (`ModuleNotFoundError` before implementation);
- GREEN: run `41591267999`, all synthetic tests passed.

This is **not v4 yet**. It has not consumed:

- final transfer firewall predictions;
- official OOF;
- Kaggle Public LB;
- any new scientific training run.

Do not activate/freeze v4 while v3 remains unresolved.

## v3 PASS path

The scientific contract is ready, but v3-specific official-OOF implementation is intentionally not built yet.

Only after transfer PASS:

1. preserve the exact passing v3 Stage-1 checkpoint lineage;
2. adapt the existing v1 Gate-B worker to frozen DC0+AC1 representation;
3. keep v1 official folds, Stage-2 epochs/LRs/optimizer/seed/class weighting/triplet isolation unchanged;
4. run the predeclared five-fold official OOF gate once;
5. require plain BA > `0.5467611235396471`, at least 3/5 fold wins, positive 5,000-group-bootstrap lower bound, and the class-recall safety veto;
6. only if that passes may final fit and at most one Kaggle submission occur.

Do not pre-run official OOF while transfer is unresolved.

## Execution policy

- Do **not** mark v3 FAIL while the queued Actions job is waiting for compute.
- Do **not** change DCT modes, weights, layers, source lock, model, seed, thresholds, or training parameters.
- Do **not** substitute a smaller/cheaper approximation and call it v3.
- Do **not** create a replacement scientific transfer run while `41580819667` remains queued.
- If transfer FAILs: stop v3 permanently; no official OOF, no Kaggle submission.
- If transfer PASSes: execute the predeclared v3 official OOF path.
- If v3 FAILs and a v4 is later activated, freeze its complete contract before touching the final firewall.

## Next single action

**Bring exactly one suitable ephemeral Linux CUDA self-hosted GitHub Actions runner online with custom label `gpu`. The already-queued scientific run `41580819667` should claim it. Do not create a new scientific run.**
