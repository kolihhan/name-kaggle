# Global 53 + Token 13 Fusion — Nested Prompt-Grouped OOF Result (2026-11-23)

**Run succeeded:** [GitHub Actions #45169232586](https://github.com/kolihhan/name-kaggle/actions/runs/45169232586). All publisher cache SHA checks passed; the earlier first attempt stopped on an implementation issue (short non-GLM sequences were unnecessarily sent to token extraction), which was fixed before this successful run.

## Frozen question

Can the existing strong 53 global likelihood statistics be improved by learning to combine them with 13 real token-level evidence features, including ATLAS argmax-prod margin, robust token-difference distribution and four token-position segment means, with zero new model calls?

## Data and evaluation

- 2,699 official TRAIN IDs covered by the already-published GLM-5.3 / GLM-5.3-Flash token caches; source versions 1 and 2 verified against all eight previously pinned SHA256 hashes.
- 300 real GLM/Flash-labelled responses, 150 prompt groups.
- Frozen outer 5-fold prompt-disjoint OOF. Inner 4-fold StratifiedGroupKFold used **only inside each outer training split** to choose L2 C and token block scaling. Scaling is fit inside folds.
- No new model inference, paid API, QLoRA, GPU or Kaggle submission.
- This is reused official TRAIN data and a two-class GLM-only test, not an independent external transfer or an 18-class estimate.

## Actual results

| Method | Balanced accuracy | Correct / 300 |
|---|---:|---:|
| **C0: Global 53 (fixed C=1)** | **94.33%** | **283** |
| C1: Global 53 + Token 13 (fixed C=1) | 92.00% | 276 |
| C2: Token 13 only (fixed C=1) | 92.67% | 278 |
| C3: Global 53 + Token 13 (nested prompt CV) | 92.33% | 277 |

The 94.33% baseline reproduced exactly.

### Paired same-row results vs C0

| Fusion | Rescues | Regressions | Net correct | BA change | Prompt-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| Fixed concat | 1 | 8 | −7 | −2.33 pp | [−4.33, −0.67] pp |
| **Nested fusion** | **1** | **7** | **−6** | **−2.00 pp** | **[−4.00, −0.33] pp** |
| Token only | 9 | 14 | −5 | −1.67 pp | [−4.67, +1.67] pp |

Nested fusion loses to global-only in **4 of 5 folds** (two folds tie, one fold improves, two folds decline; more precisely deltas: 0, 0, −8.33, −3.33, +1.67 percentage points). Inner selection picked the global-only baseline in one outer fold and a 0.5 token-block scale in the other four. Despite that nested selection, held-out token fusion still underperforms.

## Verdict

**Reject this 13-feature fusion for the current GLM scorer.** Token evidence contains some complementary cases (token-only rescues 9 global errors), but it also introduces more regressions. With the existing global feature set already at 94.33% on this cohort, these token summaries do not provide enough clean conditional signal; the nested model's prompt-bootstrap interval is entirely below zero on this reused cohort.

Do not tune thresholds or keep changing the 13-feature block against these same 300 rows. The next meaningful evidence would require a genuinely independent labelled source/model cohort or verified paired predictions for the full 18-class 0.715 path. No claim of Public leaderboard gain is supported.

[Experiment code](../../scripts/ntl_global53_token_fusion_nestedcv_20261123.py) · [Workflow](../../.github/workflows/ntl-global53-token-fusion-20261123.yml) · [Run #45169232586](https://github.com/kolihhan/name-kaggle/actions/runs/45169232586)
