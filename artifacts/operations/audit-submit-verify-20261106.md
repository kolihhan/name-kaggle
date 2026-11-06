# Audit → Submit → Verify — Name That LLM (2026-11-06)

Execution mode chosen by user: **no own ML experiments or new training; only audit existing published notebooks, submit an unchanged version, verify official Kaggle submission/results**. The publicly authored notebooks may perform fit/model computations when Kaggle reruns them for scoring; we did not create or run a custom training experiment.

## Verified official My Submissions, before new submission

Read-only authenticated GitHub Actions [42161796795](https://github.com/kolihhan/name-kaggle/actions/runs/42161796795), snapshot **2026-11-06T04:19:27Z**, showed **11 submissions**.

| Ref | Published approach | Official public score | Status |
|---|---|---:|---|
| `58022340` | Ozertuu exact V3 (author's public notebook) | **0.668** | COMPLETE |
| `58008463` | Lightning V47 (author's public notebook) | 0.666 | COMPLETE |
| `57908683` and exact retries | independently built two-stage LoRA24 | 0.520 | COMPLETE |
| `57983359`, `57985641` | Cosmorin V1 reproductions | blank | COMPLETE, not a scored success |

The 0.668 was verified from **both** the official submissions table and `kaggle competitions submission 58022340`; it is NOT the 0.672 suggested in an author title.

## Ranked public notebook audit

- [Source inspection run 42161923001](https://github.com/kolihhan/name-kaggle/actions/runs/42161923001) read `kernels status`, pulled published notebook source only and inspected notebook metadata without inference.
- Candidate selected: `ozertuu/0-672-dual-svc-residual-refinement-apex` (author title **0.672+**, NOT our Kaggle score).
- Public latest status `KernelWorkerStatus.COMPLETE`. Exact notebook code SHA256 `3cc78ce05de7304b1240b3b406f00909690e2183efc04b0ea76e267677c70dbd`; 9 code cells, competition source `name-that-llm`, offline GPU, 20 datasets / 8 notebook sources / 8 model sources; code includes `submission.parquet`. A public complete notebook run is not evidence of successful full hidden rerun.
- Audit found `.fit(` calls and strings associated with gradient training; **do not promise zero computation when Kaggle reruns this externally authored notebook**. We did **not** fine-tune, train models or run a local scientific experiment.
- [Exact version pin run 42162097106](https://github.com/kolihhan/name-kaggle/actions/runs/42162097106) independently downloaded `/1`; SHA256 matched latest publicly inspected source, with all metadata assertions true. The other public slugs were surveyed but not submitted.
- Source is different from earlier Ozertuu V3 (source SHA256 `05367a7f8efd41b2f4959505bc7e2e07a5978b85ee426c08bb5969c60945e2c2`). Different SHA does not mean independent scientific novelty.

## Actual official submission (one candidate, one acceptance)

[Submission workflow 42162219817](https://github.com/kolihhan/name-kaggle/actions/runs/42162219817) verified the exact immutable V1 source/metadata, checked **11 prior submissions and previous champion ref 58022340/0.668** to prevent accidental duplicate/concurrent submission, then called:

`kaggle competitions submit name-that-llm -k ozertuu/0-672-dual-svc-residual-refinement-apex -v 1 -f submission.parquet -m "Reproduction public Ozertuu Dual-SVC Residual Apex exact V1 20261106 no independent training"`

Kaggle returned and My Submissions independently confirmed **ref `58045266`**, accepted **2026-11-06 04:23:43 UTC**. First verified status: `SubmissionStatus.PENDING`, blank Public and Private score. Do NOT report as scored yet. DO NOT submit the same code again; version/description locked. No new custom kernel, training run, Public-LB tuning or local ML experiment occurred.

**Next verification:** query `kaggle competitions submission 58045266` and the authenticated My Submissions table read-only. On COMPLETE, report official Public Score against **0.668** and any Kaggle exception detail; no automatic retuning or duplicate submission. If score is blank, treat COMPLETE as not necessarily scoring success. This is an operational benchmark reproduction and is NOT independent original research.


## Immediate official scoring recheck

[GitHub Actions 42162414001](https://github.com/kolihhan/name-kaggle/actions/runs/42162414001) **SUCCESS** (read-only), checked 2026-11-06 04:25:37 UTC. Authenticated `kaggle competitions submission 58045266` and `competitions submissions name-that-llm` independently agree: `SubmissionStatus.PENDING`, blank public and private scores. This verifies submission acceptance and current status, **not hidden execution success or score**. No duplicate submission was made; best scored remains 0.668 until a new officially scored entry appears.

## Recheck — 2026-11-06 12:55 MYT

[GitHub Actions 42165348166](https://github.com/kolihhan/name-kaggle/actions/runs/42165348166) SUCCESS. Authenticated Kaggle query at 2026-11-06T04:55:49Z reports exact submission `58045266` remains `SubmissionStatus.PENDING`, blank Public/Private scores. The existing 0.668 (`58022340`) remains the latest confirmed scored best. This was a read-only recheck; no new submission or experiment. Do not resubmit this exact V1. On terminal status compare against 0.668 or inspect runtime failure if terminal unscored.
