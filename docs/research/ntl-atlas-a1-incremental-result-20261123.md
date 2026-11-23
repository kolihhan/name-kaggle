# NTL ATLAS A1 — Frozen Global53 + One Fixed ATLAS Margin (2026-11-23)

**SUCCESS**: [GitHub Actions #45174502026](https://github.com/kolihhan/name-kaggle/actions/runs/45174502026). Original attempt [#45173649443](https://github.com/kolihhan/name-kaggle/actions/runs/45173649443) failed **before scoring** due to `requests.exceptions.ChunkedEncodingError: Response ended prematurely` while downloading the original Kaggle public-cache ZIP. Retry on transport only, with unchanged SHA256 source pins, passed: one interrupted attempt then valid pinned source recovered. No model-score tuning.

## Frozen test

- Existing official TRAIN 2699, but only **300** true GLM5.3 / GLM5.3-Flash labelled rows and **150** prompt groups used for this binary diagnosis.
- 5 frozen prompt-disjoint folds, identical logistic head `StandardScaler + LogisticRegression(C=1, class_weight=balanced)`.
- Historical baseline **53** published global / paired features. Only new feature is one original ATLAS fixed-temperature (1,2,4,8) product-readout difference. No new feature search, model forward, paid inference, foundation-model training or Kaggle submission.
- All original publisher v1/v2 source hashes pinned, token IDs/offsets aligned. Exact historical 0.9433333 baseline predictions independently reproduced.

## Real results

| Arm | Binary BA | Correct / 300 | Recall Full | Recall Flash |
| --- | ---: | ---: | ---: | ---: |
| ATLAS readout A0 | 0.91333333 | 274 | 0.8733333 | 0.9533333 |
| Frozen original Global53 | **0.94333333** | **283** | 0.9400000 | 0.9466667 |
| Global53 + **ONE** fixed ATLAS product margin | **0.94666667** | **284** | 0.9466667 | 0.9466667 |

- Paired vs frozen Global53: **2 rescues, 1 regression**, net **+1/300** or **+0.00333334 BA**, 3 changed labels.
- 150-prompt cluster bootstrap conditional 95% interval **[-0.0066667, +0.0134167]** BA, crosses zero.
- Per-outer-fold correct (baseline→augmented) 58→59, 57→57, 56→55, 55→55, 57→58.
- ATLAS product margin correlates with pre-existing pair mean logprob difference (**Pearson 0.930719**) and pair token-win fraction (**0.918735**).
- The 15 Global53-only correct examples' true-label-signed Global53 median margin: **+1.809955**; six ATLAS-only correct examples' ATLAS median signed margin: **+0.023209**.
- Prior independent 13-feature token-fusion nested result on this same spent subset was **0.9233333**, below Global53. [Run #45169232586](https://github.com/kolihhan/name-kaggle/actions/runs/45169232586).

## Decision

**PARK** this line on these previously consumed GLM rows. +1 correct is an exploratory positive point estimate, not reliable generalization. Do not select new temperatures, thresholds, extra tokens or a discordance rule after seeing these results. A full learned ATLAS source-conditioned model was not evaluated. Full 18-way champion's real matched predictions and clean independent 18-label transfer remain unavailable: there is **no evidence of Kaggle 18-class score gain**.

**Official Kaggle Public champion remains 0.715. No new competition submission.**

Source: [script](../../scripts/ntl_atlas_incremental_a1_20261123.py) · [workflow](../../.github/workflows/ntl-atlas-incremental-a1-20261123.yml).
