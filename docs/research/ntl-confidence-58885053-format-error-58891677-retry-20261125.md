# NTL official confidence fusion — format error and controlled retry (2026-11-25)

## Authoritative result: submission 58885053 DID NOT SCORE

Read-only GitHub Actions [#45629876691](https://github.com/kolihhan/name-kaggle/actions/runs/45629876691), [#45630038648](https://github.com/kolihhan/name-kaggle/actions/runs/45630038648), [#45630513392](https://github.com/kolihhan/name-kaggle/actions/runs/45630513392) authenticated to the owner's Kaggle account.

- Kaggle `kaggle competitions submission 58885053` shows `Status: COMPLETE`, `Error: Your notebook generated a submission file with incorrect format...`, `Public Score:` empty.
- SDK `get_submission(58885053)`: `totalBytes=0` (no usable submitted artifact). Successful baseline submission `58755304` 0.718: `totalBytes=60069`.
- Source notebook `kolihhan/ntl-frozen-confidence-gsq-fusion-20261125` preview worker `COMPLETE`, no logged exception; preview artifact `submission.parquet` is 3 rows, columns id/model, no nulls, unique IDs. Preview is designed to run `FAST_COMMIT=True` when local preview dataset has fewer than 50 test rows and `KAGGLE_IS_COMPETITION_RERUN` is absent. **This is not sufficient to show the hidden rerun took the preview path.**
- No accessible hidden rerun logs or hidden output to attribute the zero-byte failure to a specific line. An unconfirmed operational hypothesis is that submitting version 1 while source notebook `QUEUED` caused incomplete scoring/replay. Alternative causes: hidden rerun exception, data-shape or probability validation failure, output missing at scorer deadline. Do not claim a proven root cause or Kaggle backend bug.

## Single controlled retry accepted: submission 58891677

**Exactly the same immutable kernel, version 1, with no code, model, parameter or metadata changes.** Only the source-worker state at submit time differs: now `KernelWorkerStatus.COMPLETE`.

- [GitHub Actions #45630734602](https://github.com/kolihhan/name-kaggle/actions/runs/45630734602) `SUCCESS`.
- Frozen source validation passed: exactly 58 original notebook cells, one changed `infer.py` source, 9 model mounts, 26 dataset mounts, 10 kernel mounts.
- Authenticated pre-submit checks: old error `totalBytes=0`, source worker `COMPLETE`, official slots `allowed_now=4`, no other active scoring submissions, GPU remaining `17.03h`.
- Kaggle CLI returned `Submission ref: 58891677`, authenticated ledger showed `SubmissionStatus.PENDING` at 2026-11-25 14:08:23 UTC.
- Marker: `ntl-confidence-fusion-v36-after-complete-one-retry-20261125`; **do not submit a third time automatically**.
- Official champion remains 0.718 until authenticated scored BA >0.718. Given same-run expert stage on the spent 252-panel decreased 147→145 correct, this is a bounded official scoring bet, not scientific proof of a gain.
- If retry also produces `totalBytes=0`, investigate hidden rerun execution/FAST_COMMIT signals and source output artifact contract before considering any further submit.
