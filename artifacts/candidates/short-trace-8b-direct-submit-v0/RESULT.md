# Short-TRACE 8B Direct Official Code Submission — execution record

Date: 2026-11-07  
**Latest verified 2026-11-07 15:19 UTC:** Submission **58093864 COMPLETE, Public Score 0.093**. 8B Short-TRACE frozen rank-transition **standalone candidate FAILED** relative to verified 0.668 best own submitted reproduction. Scientific H-vs-S-vs-T mechanism **INCONCLUSIVE**, because official Kaggle score observes only T, and earlier pilot H/S/T GPU arm did not complete.

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

## Finalize — accepted submission (2026-11-07 13:48 UTC)

- [GitHub Actions run 42376975873, rerun attempt 2](https://github.com/kolihhan/name-kaggle/actions/runs/42376975873) completed SUCCESS. The same frozen Notebook v1 status was `KernelWorkerStatus.COMPLETE`; its real preview output file `submission.parquet` was present. We used the exact same version and no new code/GPU pilot run.
- The CLI returned **`Submission ref: 58093864`** and `3 submissions remaining today`. No additional Short-TRACE official submission made.
- **Do not claim** the hidden scoring execution, its actual H/S/T metrics, or a Public score until submission ref 58093864 resolves. See read-only status workflow `42378696638`.
- From the same authenticated historical Kaggle submissions list on [Actions 42378571735](https://github.com/kolihhan/name-kaggle/actions/runs/42378571735), the submitted V47-length40 baseline is `0.663`, published Lightning V47 is `0.666`, and Ozertuu dual-SVC reproductions are `0.668` Public BA. They are *different submitted systems*, not the Short-TRACE score.

## Official scoring recheck — 2026-11-07 13:49:28 UTC

- [Authenticated read-only Kaggle CLI action 42378696638](https://github.com/kolihhan/name-kaggle/actions/runs/42378696638) returned `Submission Ref: 58093864`, `Status: PENDING`, empty Public Score and empty Private Score. The scoring rerun is **not verified complete**; no claims about final BA, GPU/runtime success, or true-vs-shuffled improvement.
- The next action is check only this submission ref for `COMPLETE`/`ERROR`; do not tune or resubmit based on the score.

## Official scoring verdict — 2026-11-07 15:19 UTC

**Authenticated first-party result:** [GitHub Actions run 42385772029](https://github.com/kolihhan/name-kaggle/actions/runs/42385772029) reads Kaggle `competitions submission 58093864` and competitions submission history without resubmission or GPU. It returned `Submission Ref: 58093864`, `Status: COMPLETE`, `Public Score: 0.093`, blank Private Score. This is an actually scored submission, not an exception/pending task.

| Exact submitted system | Official Public BA |
|---|---:|
| Short-TRACE 8B T-only standalone, 58093864 | **0.093** |
| Ozertuu Dual-SVC reproduced, 58045266 / 58022340 | **0.668** |
| Lightning V47 reproduced, 58008463 | **0.666** |
| V47 length40, 58078151 | **0.663** |

**Disposition: REJECT direct T-only standalone model; do not promote to champion or escalate to 14B, and do not patch/tune the candidate with this Public score.**

### What was and was not tested

- **Tested:** The exact frozen private 8B standalone submission path produced a valid score of 0.093. CPU tests had passed but did not establish predictive quality.
- **Not tested:** A paired **H baseline vs S shuffled transition vs T real transition** on an independent source holdout. The external H/S/T pilot was blocked by a missing bitsandbytes package, and official hidden scoring only tested the T classifier. Therefore **cannot claim true transitions are worthless, or inferior/superior to shuffled**, only that this particular T-only representation and readout was not competitive.
- **Known method limitation from source code:** No semantic/lexical feature view, no external multi-source training, no multi-evaluator ensemble; only nine fixed rank frequency bins and 81 adjacent rank-bin transition frequencies from a single frozen Qwen3-8B plus a linear logistic readout. These are plausible *mechanistic reasons* for weak discrimination, **not verified causal attribution** without ablation. Scores near chance may also reflect domain/protocol shift; the Kaggle score alone cannot separate those hypotheses.
- **Validity caveat:** The public code-competition preview contains a schema-only prediction for tiny fixture test. An official `COMPLETE` + score does not expose hidden row-level predictions, rerun trace or per-class confusion here. Do not claim the hidden path was fully logged or that failure can be traced to a particular component solely from the BA.
- **Process error:** The direct competition deployment was allowed by user direction but skipped the originally planned mechanistic H/S/T gate. A single competition score is *not* an adequate substitute for that gate and was insufficient to diagnose transition causality. No further Kaggle submissions should be used to test minor ablations of this frozen failed standalone model.

### Research decision

Archive `short-trace-8b-direct-submit-v0` as **negative standalone evidence**. Existing best own verified submitted Public BA remains `0.668`, reproduced from an external/public candidate, not original research. For a genuinely new attribution mechanism, predeclare a separate evaluation where marginal-only H, shuffled S and true T share one controlled dataset split and one frozen evaluator, plus source-shift check; validate *incremental/residual complementarity* to strong lexical or semantic anchor **off-leaderboard** before any competition-facing submit. A larger evaluator alone cannot be inferred to repair the lost discriminative content. If unable to run that independent evaluation, stop this lane rather than present a 14B escalation as supported.
