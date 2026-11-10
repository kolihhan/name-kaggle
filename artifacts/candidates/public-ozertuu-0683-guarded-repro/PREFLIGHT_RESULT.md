# Ozertuu 0.683-title candidate — guarded source preflight

Date: 2026-11-11 MYT. **SOURCE GATE PASS; ONE-OFF KAGGLE SUBMISSION ACCEPTED; SCORE PENDING.**

## Verified execution
- Audited author-public Notebook: `ozertuu/0-683-0-682-qwen3-8-27b-gsq-15-mix-0-52`; author title claim 0.683 is NOT our score.
- Kaggle upstream Notebook status `COMPLETE`, source SHA256 `14d48eb60e4201cd749e0db7f6b9fae89796d4cbb10a4f2f7006300f267535c6`.
- Offline T4 metadata; original competition source, 20 dataset, 8 Notebook, 6 model refs.
- Original `submission.parquet` listed in public preview, not proof of hidden-scale runtime.
- [Read-only source/execution audit](https://github.com/kolihhan/name-kaggle/actions/runs/42872002790) SUCCESS.
- [Guarded reproduction source preflight](https://github.com/kolihhan/name-kaggle/actions/runs/42872819394) SUCCESS:
  - Original SHA verified: `14d48eb60e4201cd749e0db7f6b9fae89796d4cbb10a4f2f7006300f267535c6`.
  - Generated private Notebook SHA: `49d4fdce69b4586afff6d240969ffe443f6537a5f810803e758313bb62b63574`.
  - Three fail-closed training guards inserted and verified: `gauss.fit`, Atlas training, core-worker `--train-final`.
  - Did not run a Kaggle model, train, submit, commit raw public source, download official text, or consume GPU.
- One-shot own Kaggle private reproduction and submission **initiated** in [Action #42873288053](https://github.com/kolihhan/name-kaggle/actions/runs/42873288053), which must be inspected for an accepted official Submission ID. Workflow START is not submission success.

## Uncertainty / restraint
- Upstream's actual numeric Public Score **not independently available** through `kernels list`, which omits numeric score fields. Treat 0.683 as title claim.
- Upstream-specific SPDX license not independently established by Kaggle CLI; no claim of Apache-2. Kaggle official documentation permits forking public Notebooks; own fork remains private and retains source credit.
- Actual hidden rerun (~4,000 passages, <=9h) is untested until the Kaggle scorer completes. If supervised Gaussian fallback is required, frozen no-training guards force failure rather than silently train.
- No independent scientific/model novelty claimed. Baseline remains 0.668 (58022340/58045266) until a real higher official score exists.
- Historical public Git competition-data incident still unresolved, but repo policy permits guarded no-data Actions while public.

## Official one-shot promotion (2026-11-11 02:17 MYT)

[GitHub Actions #42873288053](https://github.com/kolihhan/name-kaggle/actions/runs/42873288053) **COMPLETED SUCCESS**, job `130029849162`. This is Kaggle-authenticated evidence; action success alone is NOT the competition score.

- `NTL_OZ_0683_OWN_KERNEL_VERSION1_PUSHED_WITH_NO_TRAINING_GUARDS`: private own Notebook V1 successfully uploaded.
- Preview progressed QUEUED → COMPLETE, and `submission.parquet` was listed.
- Kaggle `competitions submit` returned **official submission ref `58210835`**.
- Confirmed official ledger row: `status=SubmissionStatus.PENDING`, `publicScore=""`, `privateScore=""`; description `Frozen public Ozertuu GSQ15 0.683 claim guarded no-training reproduction 20261111`.
- No new supervised training, no repeated submit, no external data uploaded to public GitHub, no unauthorized change to existing frozen OOF.
- Until an authenticated nonempty `publicScore` appears, independently verified best remains **0.668**. The public source's "0.683" remains a **title claim** for this exact reproduction.

**Status propagation:** The prior "one-shot initiated" text is superseded by the official submission receipt above. The Kaggle hidden-scale inference is PENDING, not verified complete. Do not resubmit the same candidate.
