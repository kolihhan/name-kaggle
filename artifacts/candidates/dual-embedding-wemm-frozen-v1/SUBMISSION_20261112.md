# Current execution checkpoint — 2026-11-12 13:34 MYT

- **FAILED:** `58244483` remains an unscored hidden runtime failure, freshly authenticated in [Action 43149001011](https://github.com/kolihhan/name-kaggle/actions/runs/43149001011).
- **FAILED:** CUDA bootstrap v3 ended ERROR before Q14 load, with no output. [Recovered evidence 43162254900](https://github.com/kolihhan/name-kaggle/actions/runs/43162254900) rejects lazy initialization as a sufficient fix; underlying visibility/build/mapping cause remains unknown.
- **DONE (CPU only):** builder no longer allows the precomputed 3-row fixture to stand in for real inference. Three focused regression tests pass. Existing Kaggle versions and model/scientific configuration were not changed.
- **NOT EXECUTED:** new GPU run or official submission. Expected advantage versus 0.679 is unmeasured; real full-pipeline and increasing-size inference have not passed. Submission quota is available (5), so quota is not the reason to abstain.
- Session cost: 2 read-only Actions; no new model download, training, GPU run or submission.

The records below remain historical provenance.

---

# Terminal official result — WeMM hidden rerun FAILED (verified 2026-11-12 01:44 MYT)

**SUPERSEDES the original PENDING entry below.** Exact submission `58244483` returned `Status: COMPLETE` **with `Error: Your notebook hit an unhandled error while rerunning your code`**, empty Public and Private Scores. `COMPLETE` here is **not a successful scored submission**. Independently authenticated [read-only exact-details workflow #43071058170](https://github.com/kolihhan/name-kaggle/actions/runs/43071058170), job `130698716843`, succeeded. Prior [read-only status workflow #43070833655](https://github.com/kolihhan/name-kaggle/actions/runs/43070833655) saw COMPLETE with blank score, prompting this investigation. The owned private Notebook's **three-example preview was COMPLETE**, but Kaggle's **hidden test rerun failed**; the exact exception traceback is not available from this official generic error category. No new submission, model fit, GPU run or Kaggle quota was consumed by these checks. **Keep champion `58210835` = 0.679 Public BA.**

**Decision:** Do not repeat the same Notebook or speculate that WeMM's representation is worse/better; this is a deployment failure, not an accuracy measurement. Audit hidden-size behavior, memory/shape/checkpoints and runtime source without using hidden labels, then only consider a separately frozen defensive revision after credible reproducible diagnosis and new one-shot authorization.

---

# Dual Embedding + WeMM — verified Kaggle submission

**Date:** 2026-11-12 00:38 MYT (2026-11-11 16:38 UTC).

## Official result at acceptance

- **Kaggle competition submission ref: `58244483`**, description `WEMM frozen dual embedding exploratory no-training 20261112`.
- Official Kaggle ledger: **`SubmissionStatus.PENDING`**, `publicScore` blank, `privateScore` blank, date `2026-11-11 16:38:25.427000 UTC`.
- Independent read-only recheck [#43060338073](https://github.com/kolihhan/name-kaggle/actions/runs/43060338073) **SUCCESS** at 16:38:43 UTC: exact same ref/date/status and private preview Notebook `kolihhan/ntl-dual-embedding-wemm-frozen-v1` `KernelWorkerStatus.COMPLETE`.
- One-shot Kaggle push + preview + official entry [#43060048662](https://github.com/kolihhan/name-kaggle/actions/runs/43060048662) **SUCCESS**; preflight job `130659812086`, submit job `130659953429`. Private version 1 created only once; 3-example output `submission.parquet` confirmed. Preview COMPLETE is **not** official hidden rerun scored.
- **Champion still `58210835` Public BA `0.679` COMPLETE**, not superseded by this pending candidate. Previous `58233227` MGT-v1 still PENDING.

## Evidence-based reasoning for the experiment

Historical Phase7 -> Phase15 gains in repeatedly examined grouped OOF did **not** transfer to Kaggle Public. The incumbent's 347 external-row diagnostic had **71/87 errors within provider family** and 12/18 supported true classes; this does not prove a clean 18-class transfer ceiling. Public #1 `0.765` is verified on the authenticated leaderboard snapshot, but its model code has not been verified available, so it cannot be reproduced from score alone.

Frozen candidate adds a source-visible Tencent WeMM 4B embedding dependency alongside Qwen3 Embedding and other prefit public experts, testing a different representation against the heavily shared current public pipelines. It is not an independently trained model. **Comparisons are confounded by multiple source pipeline differences; a Public BA gain alone would not establish that WeMM caused it.**

## Source integrity and no-training ablation

- Original public Notebook: `ozertuu/llm-dual-embedding-lora-qwen3-8-27b-gsq` (Ertuğrul Özer, retain source attribution).
- Exact public executable code SHA256: `a65e0115aaa56e3ce64391e9911fed7d3521c7ace3a1e990d92cc4f1b8a03a82`, 43 code cells. Public Notebook container SHA256 as pulled: `8708ba43653f262f1be3fdec423ffcb463ea07affc765e69fab7c633a09af1a1`. Earlier container hash differed but code digest did not; exact model/source manifest and training-call inventory checked.
- Preexisting three training guards preserved for Gaussian, Atlas and core-worker supervised training. Additional `sibling_style.py` function explicitly returns its input logits unchanged, removing **two** LogisticRegression fitting callsites from the executable output. Original source had six `fit` / `fit_transform` sites: three unsupervised TF-IDF prompt-group transforms (retained), one prohibited Gaussian fitting site (guarded), two prohibited sibling supervised style sites (removed).
- Inference-only public-source ablation, **not exact upstream reproduction** and not proven to improve accuracy.
- Authenticated fixed source/guard no-GPU preflight [#43059889716](https://github.com/kolihhan/name-kaggle/actions/runs/43059889716) **SUCCESS**. Built private Notebook SHA256 `96db3dd6ed4cfb620e2567c7c578bd13cc886b618e4b35ab820bf5e3b5b72cc1`.
- Private Kaggle Notebook; internet disabled; T4 runtime; no new supervised training, no official Competition Data in GitHub runners/artifacts, no new public weights or provider secrets committed.

## Interpretation and next action

This was **one user-authorized exploratory competition submission**, notwithstanding the still **BLOCKED** clean fresh 18-label same-row independent holdout. It is an experiment, not a scientifically promoted model. Do not tune WeMM/sibling gates or resubmit based on Public LB.

When the **authenticated Kaggle** ledger transitions `58244483` to `COMPLETE` with a numeric `publicScore`, compare only to champion `0.679`. Keep independent transfer claims and Private BA as unknown. If scoring fails, inspect official error rather than duplicate/retrain.

Evidence ledger: [read-only recheck #43060338073](https://github.com/kolihhan/name-kaggle/actions/runs/43060338073) and [one-shot execution #43060048662](https://github.com/kolihhan/name-kaggle/actions/runs/43060048662).
