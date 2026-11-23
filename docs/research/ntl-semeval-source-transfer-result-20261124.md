# NTL SemEval 2024 top-solution transfer — Real-source diagnostic result

Date: 2026-11-24 MYT. **Terminal success:** [GitHub Actions #45228449276](https://github.com/kolihhan/name-kaggle/actions/runs/45228449276). Frozen preregistration [contract](ntl-semeval-source-transfer-contract-20261124.md), code [script](../../scripts/ntl_semeval_source_transfer_20261124.py). The first attempt [#45228160865](https://github.com/kolihhan/name-kaggle/actions/runs/45228160865) failed before measurement because actual C1/C2 schemas contain `label`, not `model`; fixed only the prior-documented alias and reran without changing samples, metrics, parameters or gate.

## Data and scope

- Authenticated Kaggle official historical TRAIN: 2,699 real texts, 18 declared source classes, exactly 149–150 records per class. Training class max/min ratio: **150/149 = 1.0067114**.
- External C1 publisher source `beroli/ntl-benchmark-c1-generated-panels`: 360 responses across 30 distinct prompt groups, original `passages.csv` SHA256 `8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177`.
- External C2 publisher source `beroli/ntl-benchmark-c2-generated-panels`: 470 responses across 50 prompt groups, original `passages.csv` SHA256 `bf25ef1e953d5c37b177f27b841a04098ae9b35f45fc43fa11036d91e8831e02`.
- Comparison: only the 12 shared official source classes. Official subset 1,800 responses across 150 prompts, C1 360 across 30, C2 470 across 50. Both C1/C2 come from the same publisher, **actual provider-served versions not independently verified**.
- Every feature defined before looking at results. Per-response feature minus its prompt group mean of the other models, followed by per-declared-class feature means. Spearman rank correlation across **12 class centroids** between each corpus. This is **not a model classifier**, no OOF Balanced Accuracy, no current 0.715 same-row predictions, no model training or Kaggle submit.
- Frozen descriptive criterion: a feature requires both official vs C1 and official vs C2 rho >= 0.35. At least 3 of 5 features required to pass.

## Results (authentic source files, actual code executed)

| Predeclared prompt-controlled fingerprint | Official vs C1 rho | Official vs C2 rho | C1 vs C2 rho | Meets frozen 0.35 both? |
| --- | ---: | ---: | ---: | :---: |
| Word count | +0.111888 | +0.335664 | +0.132867 | NO |
| Mean word length | +0.314685 | +0.629371 | +0.629371 | NO |
| Punctuation / 100 words | **+0.839161** | **+0.909091** | +0.776224 | YES |
| Digit characters / 100 words | +0.132867 | **−0.398601** | −0.531469 | NO |
| Function words / 100 words | **+0.559441** | **+0.657343** | +0.706294 | YES |

**Frozen stability criterion: 2/5; required 3/5 — FAIL.**

## Interpretation

1. **SemEval-2024 AISPACE class weighting not justified by official NTL label-frequency imbalance alone.** Existing 18-label official TRAIN is already essentially balanced, so inverse-count weighting would differ by at most 0.67% under this simplest definition. This **does not** disprove class-specific loss improvements on a different or externally augmented distribution.
2. **Naive pooling of C1/C2 is not prevalidated.** The 12-source relative rank order changes for word count, mean word length (C1), and digit features; two simple style cues persist under this descriptive criterion, but this cannot establish that adding C1/C2 would harm or help a trained classifier.
3. These rank correlations are across just 12 labels, not independent sample-level accuracy estimates; no bootstrap interval or reliable p-value was predeclared or measured. The two external panels share a publisher and source-model identity is not independently certified. Do not infer causality, robust 18-class generalization or current 0.715 gain.
4. **Next scientific test:** candidate-source protocol/served-model integrity and whether a *previously unused* credible full18 panel can give same-row true-labeled scorer predictions. Avoid further feature selection on these repeatedly inspected data; avoid new training and avoid old Hungarian, prompt-matched surface feature tweaks.

## Execution / privacy

- GitHub Action retrieved original official competition data and external CSVs into temporary runner files, checked pinned SHA and 18/12 label contracts and self-test. All temporary data and logs scrubbed.
- No data rows, model predictions, test labels, private output or original texts published; script logs numeric aggregates only.
- No GPU, new fitted model, paid API call, Kaggle private kernel, competition submission or change to frozen 0.715 Champion.
- **Official owner Public BA remains 0.715 submission #58680531.** No verified new score.

**Decisions:** PARK naive 12-label external-source pooling and SemEval frequency-based class weighting as standalone score-facing ideas; REOPEN only if cleaner source/channel evidence or a materially new well-supported hypothesis becomes available. This is not evidence against trainable encoder models in general.
