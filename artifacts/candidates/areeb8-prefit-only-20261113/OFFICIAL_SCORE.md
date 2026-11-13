# Areeb8 no-fit real-path — authenticated official Kaggle score

Verified date: 2026-11-13 14:24:25 UTC.

## Official result
- Competition: Name That LLM, 18-class Balanced Accuracy.
- Official Kaggle submission **58295802**, status **SubmissionStatus.COMPLETE**, **Public BA 0.683**, Private BA unknown.
- Independently authenticated [read-only exact receipt Action 43440969485](https://github.com/kolihhan/name-kaggle/actions/runs/43440969485). Prior best 0.679 (refs 58210835 and 58233227), gain **+0.004 BA** (+0.4 percentage points).
- This external Public score is not independent full-18 held-out matched-pair evidence, nor a guarantee of future Private performance.

## Frozen provenance
- Public source areebbb/0-684-i-m-better, pinned original source SHA256 5d053f49005df0b2511e07bfb154faf53f22bca130761ac4a17b61aa5939cb00.
- Own inference-only source SHA256 178587738bf3474e78a7701d78254455192a0d16c97815075fa2560e3b695902, private Kaggle kernel kolihhan/ntl-areeb8-prefit-only-nov13 v1.
- [Original Kaggle GPU full-path gate + guarded one-shot official submission #43363342756](https://github.com/kolihhan/name-kaggle/actions/runs/43363342756). Real 3-row private GPU output: valid schema and 3 distinct predicted labels; no fit guard triggered.
- Removed original fake FAST_COMMIT shortcut, disabled supervised Gaussian flags, and fail-closed guarded sklearn LinearSVC/LogisticRegression and Gaussian fit. Frozen public prefit weights preserved; no new supervised fitting or neural training.
- The successful official numeric score establishes that this exact candidate completed its hidden scorer rerun. It does NOT guarantee another candidate's hidden runtime.
- This is an attributed public-source pretrained ensemble with execution/safety corrections, not an original model trained by the account.

## Next milestone: 0.700
- Need +0.017 BA, without using 0.683 Public LB to select thresholds, weights, models, or supervised fit at test time.
- Missing evidence: full 18-class same-row independent benchmark, per-class rescue/regression, source/protocol and served-identity disjointness, all future candidate 9-hour runtime.
- Previously spent diagnostics imply same-provider sibling confusion but have not established that it is the residual error distribution of the current 0.683 candidate.
- Freeze a materially different no-train sibling-evidence hypothesis before a new external evaluation; do not duplicate submission 58295802 or refit the Kaggle notebook in place.
