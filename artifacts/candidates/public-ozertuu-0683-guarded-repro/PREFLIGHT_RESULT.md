# Ozertuu 0.683-title candidate — guarded source preflight

Date: 2026-11-11 MYT. **Source gate PASS. Hidden-score outcome not yet known.**

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
