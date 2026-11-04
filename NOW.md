# NOW

Updated: 2026-11-04 MYT

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
- `notebooks/dct_v3_colab_t4_probe.ipynb`: pinned, synthetic-only T4 LoRA + DCT forward/backward memory feasibility probe. **Created but NOT executed on a real T4**. It scores zero official or heldout rows and cannot decide v3 scientific transfer.
- `notebooks/dct_v3_colab_gpu.ipynb`: pinned full paired-transfer launcher for sufficiently large CUDA GPU, but **no live Colab execution was observed**.
- **Kaggle T4**: private, no-data quota-only probe `41767348246` re-ran from GitHub Actions. Push failed at `2026-11-04T03:28:46Z`: `Maximum weekly GPU quota of 30.00 hours reached`. GPU kernel did not launch. The updated workflow now fails closed instead of polling a missing kernel.
- **Hugging Face Jobs**: connected Free account's L4 attempt returned `402 Payment Required`; not a usable no-cost GPU route.
- A local/container GPU is not available from the current agent runtime; the user's live Colab session cannot be operated through the connected tools.

GPU feasibility remains a separate execution question from the frozen v1-v3 scientific comparison. Do not claim a real T4 step or v3 transfer score without explicit execution logs/artifacts.

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
- Corrected run `41772827966` executed the GitHub Actions job and stopped at its credential preflight: secret `COLAB_ADC_JSON` was absent; **no Colab runtime was requested, no GPU allocated, and no training/scoring occurred**.
- The only remaining blocker for the Colab bridge is one-time Google **user OAuth ADC** authorization supplied privately to GitHub Actions as the repository secret `COLAB_ADC_JSON`. Never commit the credential, upload it as an artifact, or paste it into chat. Do not repeat the same no-credential run.
- Never weaken the frozen source lock, architecture, DCT modes, training steps, split, seed, or evaluation gates to fit a smaller GPU and then call the result v3.
- Do not consume official OOF or Public LB before exact paired source-transfer PASS.

## Next action

For the official GitHub→Colab CLI path, resolve the one-time missing `COLAB_ADC_JSON` GitHub Actions secret using Google user OAuth ADC; then run `.github/workflows/v3-colab-t4-probe.yml` exactly once to obtain the synthetic T4 result. Without that credential, do not retry this path. Alternative: run the existing Colab notebook interactively in the user's already-authenticated browser session. Do not claim it ran unless verified. If T4 OOMs, seek GPU with >=24 GiB and adequate runtime. If Kaggle GPU quota recovers, run only a bounded no-official-data hardware probe first; defer full paired scientific training until resource + checkpoint viability are established.