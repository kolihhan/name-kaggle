# NTL official fusion failures: CPU forensic closeout (2026-11-26 MYT)

## Operating decision: candidate RETIRED
**Do not submit again, do not push a Kaggle notebook, and do not start a GPU.** Owner explicitly prohibited further submission of this candidate. Preserve Public Best **0.718** (official submission **58755304**).

Three submission/retry workflows are disabled (`workflow_dispatch` only plus job `if: false`):
- `.github/workflows/ntl-confidence-fusion-official-submit-20261125.yml`
- `.github/workflows/ntl-confidence-official-submit-existing-20261125.yml`
- `.github/workflows/ntl-58885053-after-complete-one-retry-20261125.yml`

## Verified official Kaggle receipts
- **58885053**: `COMPLETE`, no Public score, generic **Submission Format Error**, SDK `totalBytes=0`.
- **58891677**: `COMPLETE`, same error, no Public score, SDK `totalBytes=0`, even though source notebook was `COMPLETE` before submission.
- **58755304**: `COMPLETE`, **0.718**, SDK `totalBytes=60069`.
- Kaggle's official [Code Competition debugging guide](https://www.kaggle.com/code-competition-debugging) explicitly states the failure categories are intentionally generalized and hidden-run debug details are withheld. `totalBytes=0` in the score receipt is **not a hidden runtime trace**: don't infer the exact stage or claim a proven root cause.
- Read-only receipts: [#45650308213](https://github.com/kolihhan/name-kaggle/actions/runs/45650308213) and [#45630513392](https://github.com/kolihhan/name-kaggle/actions/runs/45630513392).

## New evidence from CPU-only investigation
Source forensic: [#45657934059](https://github.com/kolihhan/name-kaggle/actions/runs/45657934059); output-writer trace: [#45658094568](https://github.com/kolihhan/name-kaggle/actions/runs/45658094568).

1. The failing source's `infer.py` **writes a stage output before any modified GSQ code**. In the saved module: `write_pred` at line 488 constructs `id,model`, asserts legal labels, unique IDs and row parity, then calls `to_parquet(a.out,index=False)` at line 491; `P=blend(LP)` and `write(P,'blend without ATLAS')` occur at lines 500–501. New fusion `_ntl_fuse(P,Q27)` is only reached at line 624. Within the GSQ stage, subsequent writes happen after the fusion.
2. The entire GSQ branch is under `try/except Exception`; a `ValueError` from probability normalization would be caught and logged as `GSQ SKIPPED`. **Such an exception alone cannot establish why the official rerun failed scoring**.
3. Exact failed-source metadata: 58 notebook cells, 9 model mounts, 26 dataset mounts, 10 kernel mounts; GPU enabled, internet disabled, competition source `name-that-llm`. Source cached notebook SHA `4205c0a2048a2be775f6cb92dc5aab8a6697609d40a276079542fb4ee8c021a5`. These counts match the pinned source contract, but exact mount *set identity against the owner's successful 0.718 clone* could not be directly compared because its old owner kernel slug was no longer accessible by the read-only CLI. Do not claim exact metadata parity.
4. Full-size **synthetic CPU** validations of `fuse()` passed for N = 3, 252, 4000, 4500, 6500, including normalization, legal shape, nonnegative scores, and unchanged ineligible rows. At N=4000 the measured fusion time was ~0.032 seconds; N=6500 ~0.008 seconds. These are synthetic random model probabilities, **not real hidden-input behavior**. They rule out simple row-count-dimensional failure in this isolated function, not Kaggle rerun or full pipeline failures.
5. The private notebook visible execution generates a 3-row preview with `FAST_COMMIT=True`, deliberately without running full inference. Its completion **does not validate the hidden dataset path**. The prior worker queue theory was falsified by the completed-worker second failure.

## Remaining hypotheses, NOT verified
- Hidden rerun runs a different or incomplete code path, or has output file/row/id/label mismatch despite early write.
- Inference stages upstream/downstream of GSQ fail or exceed resource constraints for hidden input.
- Kaggle's rerun artifact extraction, notebook-version or environment contract differs from the working preview.
- Genuine `infer.py` fusion change may trigger a hidden-only state, but the CPU synthetic passing and caught GSQ errors do not support a specific failure mechanism.

No access to hidden scorer stderr, hidden test IDs/shape, or rerun artifact bytes; available evidence does **not isolate root cause**. Do not label this as fixed or pass another Kaggle submission to search for a signal. Further work would require independent reproducible failing evidence, extra Kaggle staff-provided diagnostics, or owner authorization for a separately scoped fully-offline CPU harness.

## Result for research
The confidence-conditioned fusion itself also failed its own strict evidence gate on the historically spent labelled 252-row panel: same-run GSQ 147→145 correct, 4 rescues and 6 regressions. Full pipeline 152/252 compared to a historical, not same-run, 151/252. **Do not promote or continue tuning this fusion lane**. No new Kaggle submits, GPU or training occurred in this forensic closeout.
