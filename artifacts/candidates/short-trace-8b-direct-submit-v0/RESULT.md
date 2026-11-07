# Short-TRACE 8B Direct Official Code Submission — execution record

Date: 2026-11-07  
**Latest verified:** 8/8 CPU checks **PASS**, offline dependency **PUBLISHED**, private Kaggle competition Notebook v1 **PUSHED**, **no confirmed official competition submission ref or score**. This is an execution-status checkpoint, not a scientific result.

## Completed
- Frozen [contract](CONTRACT.md) before scoring.
- Raw external GPU pilot was dropped in favor of a direct official hidden rerun by current user request. No local/Colab/GitHub Actions GPU inference has run in this path. Kaggle official scoring still needs an accelerator to run the frozen 8B evaluator.
- Public PyPI `bitsandbytes==0.50.2` SHA256 `55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283` vendor wheel is a [Kaggle public dataset](https://www.kaggle.com/datasets/kolihhan/ntl-short-trace-offline-bnb-0502); [Actions 42376629851](https://github.com/kolihhan/name-kaggle/actions/runs/42376629851) succeeded. No model data or official test content in dataset.
- Implementation `scripts/run_short_trace_direct_submit.py` and builder `scripts/build_short_trace_direct_submit.py`. Preflight 8 CPU unit tests PASS and Kaggle offline wheel input visibility PASS in [Actions 42376851502](https://github.com/kolihhan/name-kaggle/actions/runs/42376851502).
- Exact private code-competition Notebook [`kolihhan/ntl-short-trace-8b-direct-18class`](https://www.kaggle.com/code/kolihhan/ntl-short-trace-8b-direct-18class), **version 1**, pushed successfully. Contains competition source `name-that-llm`, 8B model input, offline runtime/wheel input, preview-only schema fixture and hidden 18-class real-rank prediction path that outputs `submission.parquet`.
- An *immediate* official code submission request for v1 returned Kaggle **HTTP 400 CreateCodeSubmission**, before any accepted ref. The reason is not verified; version availability is a plausible explanation, not an established diagnosis.
- [Same-version finalize workflow 42376975873](https://github.com/kolihhan/name-kaggle/actions/runs/42376975873) was started to reconcile submission history and submit **only if** Kaggle reports v1 notebook preview `COMPLETE` and its `submission.parquet` is available. This does not create any new Notebook/GPU experiment version.

## Not established
- Kaggle official acceptance **unknown**, no submission ref; hidden rerun **not confirmed**, no Public BA or H/S/T outcome.
- 8B runtime viability on hidden 4k rows unverified. The prior research-only NF4 Notebook failed early because bitsandbytes was absent; that is why the official package now includes a vendor wheel.
- Official-train grouped OOF is descriptive on repeatedly used official rows; a Public score cannot validate that T beats S or certify source-transfer.

## No tuning
No control/bin/seed/classifier changes or PubLB tuning. No submission of a dummy fixture as scientific results.
