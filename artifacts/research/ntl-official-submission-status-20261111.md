# Name That LLM — authenticated submission status reconciliation

## Latest official result update — 10:28 MYT (supersedes 02:22 snapshot)

[Authenticated read-only Kaggle check Actions #42874538415 attempt 2](https://github.com/kolihhan/name-kaggle/actions/runs/42874538415), job `130279696484`, **SUCCESS** at 2026-11-11 02:28:39 UTC:

| ID | Status | Official Public BA |
|---|---|---:|
| `58210835` | **COMPLETE** | **0.679** |
| `58167751` | PENDING | — |
| `58022340` | COMPLETE | 0.668 |
| `58045266` | COMPLETE | 0.668 |

Kaggle's exact-submission endpoint independently returned `Status: COMPLETE`, `Public Score: 0.679` for `58210835`. **Our new authenticated champion is 0.679, +0.011 vs 0.668.** Old PENDING status below refers only to the earlier historical snapshot and must not be repeated as current truth. Full outcome: [OFFICIAL_SCORE_20261111.md](../candidates/public-ozertuu-0683-guarded-repro/OFFICIAL_SCORE_20261111.md). **No Private Score published.**

---

Verification time: **2026-11-11 02:22 MYT (2026-11-10 18:22 UTC)**.

## Canonical Kaggle evidence

- [GitHub Actions status #42874225818](https://github.com/kolihhan/name-kaggle/actions/runs/42874225818): Kaggle `competitions submissions -v -q` and private own Notebook status, job **SUCCESS**.
- [GitHub Actions exact-submission status #42874538415](https://github.com/kolihhan/name-kaggle/actions/runs/42874538415): re-checked the same ledger **and** `kaggle competitions submission <REF>` individually, job **SUCCESS**. This removes uncertainty about parsing the aggregated CSV.
- Both workflows use GitHub Actions secret `KAGGLE_API_TOKEN` and emit strictly filtered public-safe status fields, without downloading Competition Data, training, GPU, or new submission.

| Our actual submission ID | Candidate description | Actual Kaggle status | Public Score | Evidence type |
| --- | --- | --- | --- | --- |
| `58022340` | Reproduced Ozertuu Dual-SVC V3 | `COMPLETE` | **0.668** | Official authenticated ledger |
| `58045266` | Reproduced Ozertuu Dual-SVC Residual Apex | `COMPLETE` | **0.668** | Official authenticated ledger |
| `58167751` | Reproduced public Lightning 0.681 title | `PENDING` | empty | Official ledger + exact detail |
| `58210835` | Reproduced public Ozertuu GSQ15 0.683 title, guarded no-training | `PENDING` | empty | Official ledger + exact detail |

The private Notebook `kolihhan/ntl-ozertuu-0683-guarded-no-train` has `KernelWorkerStatus.COMPLETE` (preview), **not** a completed hidden scoring run. The source-author public title 0.683 and Lightning's 0.681 are **not own official Public BA**.

## Competition interpretation

1. **Current best scored Public Balanced Accuracy remains 0.668.** Neither pending candidate can yet be compared with it.
2. The new `58210835` was formally accepted at **2026-11-10 18:17:32 UTC**, submission proved by [one-shot execution #42873288053](https://github.com/kolihhan/name-kaggle/actions/runs/42873288053). No duplicate should be submitted.
3. The older `58167751` was accepted at **2026-11-09 14:40:53 UTC** and is still pending at 2026-11-10 18:22 UTC. This is a long delay; the exact endpoint did **not** return a failure reason. Do not infer hidden rerun success, failure, stuck status, or platform outage.
4. Source-preflight guards on `58210835` remain fail-closed against supervised Gaussian fit, Atlas train mode and core `--train-final`. The future scorer may fail if these paths are needed; do not relax them for a higher score.
5. No new Kaggle submission, training, model quality experiment, or score change was performed in this reconciliation. The new read-only workflow is `.github/workflows/ntl-scored-58210835-status-readonly.yml`.

## Decision

Until Kaggle returns a terminal state: preserve existing frozen artifacts, **do not resubmit/retrain/tune**, and avoid treating Notebook preview `COMPLETE` as Public Score. When a real score appears, compare only the actual Public Score with 0.668, then separately judge whether a stronger result is an **external reproduction** or an independent advance.

Kaggle's documented command `kaggle competitions submission <REF>` exposes status and failed-submission errors where applicable: https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md.
