# NTL confidence fusion — STOP (2026-11-25)

**Owner instruction: no more Kaggle submissions or GPU for this candidate.** Retain official best **0.718** (#58755304).

## Kaggle evidence
Read-only authenticated [receipt audit](https://github.com/kolihhan/name-kaggle/actions/runs/45650308213):
- #58885053: COMPLETE, generic submission format error, empty score, `totalBytes=0`.
- #58891677: COMPLETE, identical format error, empty score, `totalBytes=0`.
- #58755304: COMPLETE, Public BA 0.718, `totalBytes=60069`.

The **same immutable Version 1** was submitted first while preview worker QUEUED, then after worker COMPLETE. Queue timing is disproven as a sufficient cause. The preview worker's 3-row `submission.parquet` is deliberate `FAST_COMMIT` behavior. The hidden rerun's output, stderr and row counts were **not available**, so the exact failure boundary is **unproven**. Kaggle's zero-byte metadata does not independently identify which stage failed.

## Superpowers systematic-debugging check
The only original code cell changed was `infer.py` GSQ fusion. Source [audit](https://github.com/kolihhan/name-kaggle/actions/runs/45650651932) confirms `_ntl_fuse(P,Q27)` is **inside try/except**, so a probability-validation exception alone would be caught and logged `GSQ SKIPPED` rather than necessarily aborting the entire scorer. Full-size hidden runtime, downstream stages, and output-contract failures remain possibilities, not established causes.

On the *spent* 252-row external panel, same-run GSQ 147→145 correct (4 rescues, 6 regressions). Candidate final 152/252 vs historical, nonpaired 151/252. No evidence for promotion.

## Permanent candidate safety gate
Verified these three GitHub Actions have **no push trigger**, are manual-only, and have job-level `if: false`:
- `.github/workflows/ntl-confidence-fusion-official-submit-20261125.yml`
- `.github/workflows/ntl-confidence-official-submit-existing-20261125.yml`
- `.github/workflows/ntl-58885053-after-complete-one-retry-20261125.yml`

**Status: RETIRED. No new submissions, notebook pushes, or GPU jobs were run during this investigation.** Any future work requires owner authorization plus actual full-size failure evidence first, not another blind retry.
