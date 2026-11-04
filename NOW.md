# NOW

Updated: 2026-11-04 MYT

## Kaggle 0.622 retry outcome — 2026-11-04 MYT

**CONFIRMED GPU QUOTA BLOCK; NO NEW KAGGLE SUBMISSION.** The user explicitly requested a fresh retry. GitHub Actions run [41823017835](https://github.com/kolihhan/name-kaggle/actions/runs/41823017835) failed at `kaggle kernels push` with Kaggle's explicit response: `Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.`. The account's maximum is **30.00 hours per week**; remaining quota at this attempt was insufficient for the GPU notebook.

- This retry corrected a genuine `kernel-metadata.json` title/slug mismatch, put original upstream attribution in a Markdown cell rather than executable Python, and set the documented `machine_shape: NvidiaTeslaT4` for T4 x2. Static preflight verified that executable code hash still matches the published scored V1 source, both dataset/model source lists and the output contract remain unchanged, and no new model training is introduced. Relevant commits: `44e738cdffd1fe206502a54943c70cc36a2de6a9`, `dbe1a5d54b38df19cae51c1a855e3cc60c015819`, `98eb2fc4d8f575ae090879df80e073ce44ad7ce5`.
- Earlier run [41822974665](https://github.com/kolihhan/name-kaggle/actions/runs/41822974665) failed only on a transient static-test newline-join bug; final run `41823017835` passed the corrected source SHA/preflight gate. The actual Kaggle push was rejected due to the **confirmed quota**.
- The existing Kaggle submissions remained **7**, including five redundant previously completed v1 rows (Public `0.520`); **there is no 0.622 submission or new Public score** from this retry.
- **DO NOT RETRY before GPU quota renews.** Do not mislabel this as a model/Notebook inference failure, silently change hardware, or attempt another training approach. Use the frozen Kaggle prefit inference-only bundle after eligibility returns; verify quota and existing submission duplicates first.

## User-requested Kaggle prefit submission attempt — 2026-11-04 MYT

**NO NEW SUBMISSION CREATED. KAGGLE KERNEL PUSH BLOCKED.** User authorized submitting a Kaggle Notebook to run on Kaggle rather than self-training. Chosen path was frozen published Cosmorin 0.622 V1 inference-only code, **not** a new training job.

- Added `scripts/build_public_0622_inference_kernel.py`, verifying upstream V1 raw notebook SHA `e9819e87b1b60231388aff0db1d0ef312b707c6c6b9911a054603333f6bb067b` and executable code SHA `1595f543caf314fc766a79dca313961af4d73f2a5cbea1ec2b4724aea5c4289b`, preserving author attribution. It does not include new neural model training.
- Added `.github/workflows/submit-public-0622-inference.yml`, a once-only Kaggle submission guard: existing submissions, exact source, public sample execution, parquet validation, and only then final Kaggle submission. It fails closed if Kaggle refuses a GPU notebook.
- First GitHub Actions run [41821527795](https://github.com/kolihhan/name-kaggle/actions/runs/41821527795) and second diagnostic run [41821744868](https://github.com/kolihhan/name-kaggle/actions/runs/41821744868) both failed at **`kaggle kernels push`**, with Kaggle CLI reporting `400 Client Error: Bad Request`. The response did not expose a confirmed detailed cause in sanitized job logs. Earlier Kaggle GPU quota probe on this date had found `Maximum weekly GPU quota of 30.00 hours reached`, so GPU quota remains a plausible but **unconfirmed** explanation for this particular HTTP 400.
- Verified **7 existing official submissions** (Phase7, Phase15, 5 duplicate v1 submissions), latest v1 Public **0.520**; no new submission, score, model training, inference or Kaggle kernel version from these attempts.
- **STOP:** Do not auto-retry 400 pushes, change Kaggle metadata or resource settings blindly, bypass the GPU restriction, or claim 0.622 for our account. Preserve pinned builder/workflow for a future targeted diagnosis or quota reset. Any later attempt must verify the specific 400 cause and current exact dataset versions before submitting. Keep user preference: avoid self-training.

## 0.622 version-pinned preflight — 2026-11-04 MYT

**STATIC PASS, END-TO-END RUNTIME UNKNOWN — NO TRAINING OR SUBMISSION.** Read the scored Kaggle Notebook V1 and current V3 directly using the public Kaggle API: executable code SHA-256 is identical (`1595f543caf314fc766a79dca313961af4d73f2a5cbea1ec2b4724aea5c4289b`), but Kaggle Docker image digests differ. The existing SVC cache's public manifest matches its code-pinned cache hash and source version; both LoRA adapter configs, three model metas and Kaggle dataset license labels were checked using **small JSON files only**. Dataset metadata says `CC-BY-SA-4.0` for both hosted datasets; HF Qwen model cards say `apache-2.0`. Current dataset version numbers and full weight-byte checksums remain **UNKNOWN**.

The full ~4,000-row hidden test triggers pairwise clustering and 300-round Sinkhorn absent from the three-row public sample, plus sequential two-14B+DeBERTa inference. The Notebook requires pinned NumPy/SciPy/scikit-learn/joblib versions, GPU T4×2 and <=9h; these full-run requirements are not benchmarked. **Do not promote 0.622 to a new Kaggle submission without the independent environment, integrity and runtime gates.** Evidence: [`artifacts/research/public-submission-reuse-audit-2026-11-04.md`](artifacts/research/public-submission-reuse-audit-2026-11-04.md), GitHub Actions `41818385187` (source V1/V3), `41818435010` and `41818716399` (public metadata/small manifests). No model weights, official data or private credentials were downloaded into the public GitHub artifacts.

## Public-submission source audit — 2026-11-04 MYT

**DONE: READ-ONLY AUDIT; NO TRAINING, INFERENCE OR SUBMISSION.** Report: [`artifacts/research/public-submission-reuse-audit-2026-11-04.md`](artifacts/research/public-submission-reuse-audit-2026-11-04.md). Verified public-notebook source pull: Actions `41810333036`; verified authenticated dataset metadata enumeration: Actions `41809970435`.

**Prioritize reproducibility of Cosmorin's historic 0.622 prefit SVC + 2x14B LoRA + DeBERTa inference pipeline** (2 dataset sources, 2 base models) using exact published versions and rights checks. No independent score gain or hidden runtime is established. The newer Cosmorin 0.636 code-only guide references 32 fixed inputs, uses CPU SVC fitting, and reports 2002s for 3-row sample, not hidden execution. The current Lightning 0.666 source uses a hardcoded 3-row preview (21-second UI is not real ensemble runtime); its scored version V47 precedes newer source edits. **DO NOT retrain, resume v3, tune on Public LB, or submit without renewed explicit authorization.** Public dataset filenames readable via authenticated API do not verify weights' bytes, licensing, or final runtime.

## User-directed strategy change — 2026-11-04 MYT

**STATUS: SELF-TRAINING PAUSED; READ-ONLY SUBMISSION REVIEW.** The user explicitly requested that the project stop pursuing self-trained v3/full 8B/27B experiments for now and look at actual Kaggle submission results instead.

- Do not launch/relaunch Colab, Kaggle GPU, self-hosted, or CPU training jobs; do not add another model submission merely to compare Public scores.
- Existing Colab v3 resume-equivalence run [41794900419](https://github.com/kolihhan/name-kaggle/actions/runs/41794900419) ended `cancelled` at 2026-11-04T08:44:34Z; **no CUDA-resume PASS or v3 source-transfer result** was obtained.
- Existing verified submissions: Phase 7 Public `0.369` (#57838845); Phase 15 Public `0.357` (#57861337); two-stage LoRA v1 Public `0.520` (#57908683). Private LB unavailable. The v1 baseline remains unchanged.
- Kaggle's live competition overview states that Public LB uses about 40% of the test, with about 60% reserved for final Private LB; Public scores are not a model-selection oracle.
- Public Kaggle notebook pages currently display stronger third-party methods (for example, `0.622`, `0.642`, `0.666`), but the availability, licensing, full checkpoint/input accessibility, clean reproduction, and independence of those methods are **not yet verified**. Their displayed notebook interactive runtime is not necessarily the hidden re-run runtime.
- **Next task:** read-only audit of published high-scoring Kaggle notebooks and existing competition submission records. Prioritize reproducible, permissible inference-only reuse of public pretrained checkpoints plus fixed, non-tuned ensemble design; evaluate runtime/data/source integrity before any future submission decision.
- Existing frozen v3 contract and past negative evidence are retained for reproducibility, not treated as an obligation to keep training. Reopening GPU training requires a fresh explicit user request.

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

## Authoritative CUDA v3 execution — awaiting usable GPU, not scored

- Self-hosted CUDA run `41580819667` was **CANCELLED** after remaining queued without a matching GPU host. It produced no paired v1/v3 training or score.
- Frozen source, model, evaluation contracts and holdouts remain unchanged.
- V3 source-transfer decision remains **UNRUN / UNKNOWN**, not FAIL.
- No official OOF, Public LB retuning, or v3 Kaggle submission has been authorized.

### Available execution routes (verified 2026-11-04)

- **Google Colab Free T4:** User-observed Tesla T4, **14.56 GiB** addressable, torch `2.11.0+cu130`. The frozen executor's `>=15 GiB` preflight rejects it; that is a conservative infrastructure gate, not a demonstrated OOM.
- `notebooks/dct_v3_colab_t4_probe.ipynb`: pinned synthetic T4 LoRA + DCT forward/backward feasibility probe; **EXECUTED SUCCESSFULLY** via GitHub Actions run `41772827966` (attempt 2), 14.563 GiB device memory, 11.68 GiB peak reserved, 4.08-second synthetic update. No official or heldout rows scored; NOT v3 transfer evidence.
- `notebooks/dct_v3_colab_gpu.ipynb`: pinned full paired-transfer launcher for sufficiently large CUDA GPU. **Full frozen v1-v3 source-transfer run still UNRUN**; the separate two-update real-data training pilot `41778479162` is not that experiment.
- **Kaggle T4**: private, no-data quota-only probe `41767348246` re-ran from GitHub Actions. Push failed at `2026-11-04T03:28:46Z`: `Maximum weekly GPU quota of 30.00 hours reached`. GPU kernel did not launch. The updated workflow now fails closed instead of polling a missing kernel.
- **Hugging Face Jobs**: connected Free account's L4 attempt returned `402 Payment Required`; not a usable no-cost GPU route.
- The agent-local runtime has no CUDA GPU, but the connected GitHub Actions workflow **can operate Colab T4 through authenticated official CLI**: synthetic run `41772827966` and real-data pilot `41778479162` both verified; quota and full-run lifetime remain unguaranteed.

GPU *single-batch* LoRA feasibility is **verified**; full-epoch viability and frozen paired v1-v3 source-transfer remain **UNKNOWN/UNRUN**. Do not claim a v3 scientific transfer score without explicit paired-training logs and heldout evaluation evidence.

## CPU shadow — stopped, separate non-scientific lineage

- The previously checkpointed CPU BF16 DCT lineage successfully preserved state through **microbatch #4** (run `41636327118`); no optimizer step had been reached (accumulation=6).
- BF16 microbatch #5 (run `41644695149`) timed out after **330 minutes**, shell exit `124`. Previous checkpoint artifact was preserved, but #5 did **not** produce verified training progress.
- The CPU continuation trigger was removed; latest CPU workflow `41673536800` was `success` only because the training chunk was **skipped**.
- No sustained CPU training. Do not convert CPU shadow partial gradients or loss to scientific transfer PASS/FAIL.
- Previous FP16/BF16 diagnostic summaries elsewhere in this branch are historical and do not authorize further CPU chunks.

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

- Primary blocker: availability of an actual GPU executor with enough tested memory and runtime to preserve the **exact** frozen 24-layer LoRA v1-v3 contract.
- Do not retrigger CPU continuation; previous #5 hit a 330-minute timeout.
- Kaggle GPU is quota-blocked as of latest `41767348246`; do not launch full kernel while this remains true.
- Synthetic T4 memory probe is only a hardware diagnostic; it is **not** v3 transfer evidence.
- Official Google Colab CLI `0.7.5` bridge was added at `.github/workflows/v3-colab-t4-probe.yml` under explicit user authorization, using a single bounded synthetic T4 probe, sparse checkout, no official rows, and forced Colab cleanup.
- First validation run `41772721301` failed at workflow parsing (job-level runner context); fixed in commit `20435de64b63e56f8d5957574c754d78b5a57ccf`.
- Initial `41772827966` attempt was blocked by missing user OAuth; the user configured `COLAB_ADC_JSON` and **the same run, attempt 2, SUCCESSFULLY allocated Colab T4**, finished one synthetic forward/backward/optimizer, peaked at 11.68 GiB reserved, and returned `synthetic_single_triplet_step_seconds=4.08`. No scoring or submission.
- New separate training pilot: `scripts/dct_v3_colab_t4_train_pilot.py`, invoked via `.github/workflows/v3-colab-t4-real-train.yml`. It loads the immutable external Phase14 safe subset artifact `13067907516`, verifies the frozen 3-source holdout split, trains **only on 31 real external-source training rows** in 12 prompt-local triplet microbatches, and performs **2 genuine AdamW optimizer updates** on FP16 Qwen3-8B (24 layers) + DCT [0,1] + LoRA r16/norms/heads.
- First full-data pilot attempt `41777780491`: remote training COMPLETED 2 optimizer steps but Colab CLI default execution timeout prevented checkpoint recovery. This is a tooling timeout, not training failure.
- **Verified successful pilot `41778479162`**: completed 2/2 optimizer updates, `head_parameter_delta_l2=0.0255487971`, `loss_mean=5.7616079` (train-only; not predictive quality), peak reserved 11.043 GiB, no official data, no heldout predictions, no Public LB. CLI was given `--timeout 1200`, and both `real_train_result.json` and `real_train_checkpoint.zip` were downloaded, validated, and retained as Actions artifact `13297683118` (~114MB compressed). Remote Colab T4 session cleanly stopped.
- This pilot checkpoint contains adapter + norm/head weights only; **it is NOT an exact-resumable full epoch/AdamW state**, NOT the frozen paired v1-v3 transfer, and NOT authority for official OOF or submission. Do not overinterpret train-loss changes or repeatedly rerun the pilot.
- The Colab CLI/Google user OAuth secret now works. Protect `COLAB_ADC_JSON`, never expose it in logs/history; the consent-screen Testing setting may affect credential lifetime.
- Never weaken the frozen source lock, architecture, DCT modes, training steps, split, seed, or evaluation gates to fit a smaller GPU and then call the result v3.
- Do not consume official OOF or Public LB before exact paired source-transfer PASS.

## PIC checkpoint/research execution review — 2026-11-04 MYT

### Evidence and decision
- **VERIFIED:** Real external-data T4 pilot `41778479162` completed 2 true optimizer updates and preserved trainable-only adapter/heads. It tested training mechanics, **not** source-transfer or predictive quality. Full v1-v3 paired source-transfer remains `UNRUN / UNKNOWN`.
- **VERIFIED:** Added `scripts/dct_v3_cuda_resume.py` (atomic optimizer-boundary save/load for trainable parameters, optimizer, scheduler, GradScaler, Python/NumPy/torch CPU and all CUDA RNG states, strict checkpoint fingerprint/cursor validation, fail-closed mid-accumulation); `tests/test_dct_v3_cuda_resume.py` proves bitwise CPU interruption/resume equivalence with dropout and AdamW and checks wrong lineage/unsafe cursor rejection.
- **VERIFIED:** Run `41792568893` — `V3 Static Tests` — completed **27 passed**, including resume primitives, and compiled frozen v3 worker/kernel. This test suite does **not** verify CUDA RNG restoration or a real resumed 24-layer Qwen run.
- **UNKNOWN:** GPU resume equivalence across separate Colab sessions, Colab free-T4 quota/lifetime for full runs, genuine source-transfer advantage for DCT, or resulting Kaggle performance.
- **REJECTED:** Treating 31 train rows, 2 updates, training loss or a successful CI as evidence that v3 beats v1. The saved pilot adapter is **not** an exact-resumable v3 Stage-1 checkpoint.

### Current PIC priority (smallest discriminating next work)
1. Wire the already-tested optimizer-boundary checkpoint primitive into the **existing frozen v1 and materialized v3 workers**, preserving all training semantics, prompt triplet order and every frozen configuration field, not building a parallel experimental training recipe.
2. Freeze a verifiable resume fingerprint binding the model revision + truncated shard manifest, source-code revision, raw public pool SHA, Phase14 subset/split-lock digest, label order, v1/v3 condition, DCT modes, triplet order, training spec, PyTorch/transformers/PEFT runtime and exact optimizer-step count. Reject foreign/incompatible checkpoints.
3. Test uninterrupted vs interrupted-and-resumed training on the **same** small fixed real-data microbatches on Colab T4; compare trainable parameters, optimizer/scheduler/scaler cursor, per-microbatch loss and RNG state without looking at heldout predictions. CPU resume 27-test CI does not substitute for this.
4. After CUDA equivalence PASS, execute both full **one-epoch** Stage-1 conditions using the exact frozen 14,502 training rows, recover checkpoints after bounded runs, and score the predeclared three-source 14,506-row holdout **once** only when both conditions finish.
5. Apply frozen paired-transfer gate as-is: pooled BA increase, positive 5,000-prompt bootstrap lower CI, positive mean source delta, at most one negative source and no per-label recall safety veto. **FAIL -> reject v3 without retuning**; **PASS -> official five-fold OOF**, then only if that gate passes proceed to final fit/submission per contract.

### Resource and research guardrails
- Do not relaunch the 2-step pilot or the 330-minute CPU shadow. Reuse the successful T4 feasibility and existing CPU-resume implementation as infrastructure evidence.
- Colab free T4 is **not** an unlimited GPU; stop cleanly and preserve optimizer-boundary checkpoints before session limits. Do not open several competing GPU sessions. No additional Kaggle GPU run while weekly quota status is unverified.
- No Public-LB tuning, no adaptive holdout source reselection, no v3 mode/layer/loss sweeps. Source-transfer is a frozen external-data diagnostic, not a guarantee of hidden-Kaggle generalization.
- Research question: whether an added temporal AC1 component captures generator identity that survives source shift, versus noise/corpus-format artifacts. This mechanism is **hypothesis**, not yet evidence. If transfer fails, write per-source/label error analysis as allowed by the frozen result report, and only then reconsider the model family or centered-to-plain distillation under a *new* predeclared contract.

- **Pre-result mechanism confound (2026-11-04):** DCT `[DC0, AC1]` also doubles the Stage-1 classifier input width and consequently increases head parameter count versus v1 mean-only pooling. A future paired-transfer PASS would establish improvement by the *frozen composite treatment*, not isolate AC1's generator-fingerprint contribution from added head capacity. Keep the v3 contract unchanged. Any capacity-matched placebo/ablation must be separately preregistered and judged with independent evidence, not used to rescue/reselect v3 against its already-locked holdout.
- **Version pin audit:** The live Colab optimizer-resume proof pins `d9bf4f0f88d396d32348e94033c07759c244ec6d`; Git blob SHAs of its shared Stage-1 worker, CUDA resume helper, v3 materializer, and frozen split builder match those on this branch as of `10a6919a18641a885d4876ab07fbbc835dfa108d`. Later commits changed proof/CI/tests/docs, not those four worker inputs. A PASS would still be *only* resume-equivalence evidence, never source-transfer evidence.

- Do not declare CUDA resume or paired transfer complete just because a state-serialization unit test passed.

## PIC implementation ledger — 2026-11-04 (resume worker integration)

- **IMPLEMENTED:** Opt-in `optimizer_boundary_path` / `checkpoint_every_steps` / `stop_after_optimizer_steps` parameters on the **shared frozen v1 A1 CUDA worker**, and the exact generated v3 worker inherits them. Defaults remain disabled; no science configuration, source selection, head pooling, DCT coefficient or evaluation threshold was changed.
- **IMPLEMENTED:** Checkpoints carry strict runtime/worker source/safe-parquet/split index/triplet order/label/model revision/manifest/training-spec fingerprint, trainable weights, AdamW, scheduler, GradScaler, CPU/CUDA RNG, optimizer cursor and training counters. Clean optimizer boundary is enforced. Pausing cannot read heldout predictions.
- **VERIFIED CPU/STATIC:** GitHub Actions `41794531008` 30 passed and follow-up `41795292904` **31 passed** including new worker default/resume contract, interrupted-vs-uninterrupted CPU dropout/AdamW equality and training counter restoration; frozen worker materialization/kernel compilation passed.
- **RUNNING/NOT YET VERIFIED:** GitHub Actions `41794900419`, a single Colab T4 invocation comparing fixed real external-data frozen-v3-worker uninterrupted two optimizer steps against interrupted one step + resume one step. No official competition data or source-holdout prediction is used; result must be checked before labeling CUDA resume PASS.
- **NOT DONE:** CUDA bitwise equivalence, full v1/v3 one-epoch training, frozen paired source-transfer and official OOF/submission. Do not infer model quality from optimizer state or training-loss outputs.

## Next action

Real-data T4 LoRA training feasibility and checkpoint artifact are now **VERIFIED** by run `41778479162`. Stop further bounded pilot reruns: there is no value in repeating two optimizer steps. The open scientific question is still the original *frozen paired v1 vs v3 full source-transfer* experiment. Before any costly full attempt, build an exact-resumable CUDA training/checkpoint mechanism that preserves optimizer/scheduler/RNG/grad-accumulation state and survives Colab session limits; keep evaluation firewall and split unchanged. GPU availability, remaining free quota, full-run duration, and the effect of DCT on heldout BA remain **UNKNOWN**.

Visibility policy (user reaffirmed 2026-11-04): Keep using the PUBLIC repository and GitHub Actions / Colab workflows; do not impose a mandatory switch back to PRIVATE after routine training. This does **not** prove Git history was sanitized: official competition data exposure in reachable history was previously recorded, and cleanup is unverified. Do not introduce new official competition files or secrets; do not rewrite history without separate authorization. See `PUBLIC_VISIBILITY.md`.
