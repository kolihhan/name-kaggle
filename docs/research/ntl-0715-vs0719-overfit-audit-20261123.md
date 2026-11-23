# NTL Public 0.715 vs headline 0.719 — overfit audit, 2026-11-23

**VERDICT:** **UNPROVEN either way.** The apparent 0.004 Public BA difference is too small to promote a purported 0.719 improvement or to assert Public-LB overfitting without matched independent truth. No new training/inference/submission was performed.

## Actual read-only execution

- [Actions #45137623140](https://github.com/kolihhan/name-kaggle/actions/runs/45137623140) — SUCCESS; live authenticated Kaggle metadata; local deterministic sensitivity math.
- 27 returned entries on owner's Kaggle submission ledger: **25 COMPLETE / 1 ERROR / 1 PENDING**, **21 valid numeric public scores**, maximum **0.715**. Some entries have no numeric score; do not confuse COMPLETE with scored.
- Kaggle public leaderboard --show succeeded but returned only **23 lines** and zero matching named author identifiers; this **cannot** establish others' scores, entries or absence. Official public score 0.719 for listed *titles* is **NOT verified by this Kaggle API query**.
- Previous [read-only highscore source audit #45136005059](https://github.com/kolihhan/name-kaggle/actions/runs/45136005059) found own scored 0.715 code SHA identical to public Estopadilla V5 and multiple 0.719-title derivatives sharing GLM/Fable resources. [Decision-scope audit #45136282372](https://github.com/kolihhan/name-kaggle/actions/runs/45136282372) found common `fable_return`/GLM prefix code in Dennis, Areeb, Hcluo and Ranjeet. This is **not** independent replication of accuracy.
- Kaggle [official leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard): roughly **40%** of test for Public score. Kaggle [official data page](https://www.kaggle.com/competitions/name-that-llm/data): hidden rerun about **4,000** passages. Thus Public has about **1,600**, **not** 4,000. Test balance by class unknown.

## Why a +0.004 Public BA is weak evidence

Balanced Accuracy is mean of 18 per-class recalls. **Only if** 1600 public samples were evenly distributed over labels, a +0.004 BA is equivalent to roughly **6.4 net additional correct examples**; it is not the true rescue count when classes are unbalanced. It is not possible to deduce number of changed predictions, model agreement, per-class gains or confidence from these two aggregate scores alone.

Under a **clearly hypothetical**, symmetric matched-pair sign null, using a six-net-improvement threshold:

| Hypothetical number of changed predictions | Two-sided null probability of at least 6 net |
|---:|---:|
| 8 | 0.07031 |
| 10 | 0.10938 |
| 20 | 0.26318 |
| 40 | 0.42959 |
| 80 | 0.57643 |
| 160 | 0.69276 |

**This table is sensitivity illustration, NOT Kaggle ground-truth p-values.** Real classes unequal, submissions may change different subsets, outputs share prompts, and Public-score candidate selection breaks single-trial assumptions. Do not apply this test to 0.715/0.719 as a real confidence interval.

One source-verifiable sign of score selection: on public [Sanpier Notebook](https://www.kaggle.com/code/sanpier/0-709-320-token-glm-likelihood-qwen-ensemble) latest page shows **13 versions, best 0.717 at V3**; the latest current code and requested V3 have different SHA256 as confirmed in our authenticated pulls. This is evidence of **multiple version trials/score visibility**, not proof of train-data or Public-LB overfit. The 0.719 titles of Areeb/Hcluo cannot be upgraded to authenticated score claims from notebook names alone.

## Actionable science decision

1. **Freeze** own **0.715**; no promotion, no force relabeling, no tuning user-prefix/GLM/Fable thresholds against the same Public 40%.
2. **Require matched evidence:** same class labels, same rows, both complete end-to-end final predictions, no data leakage from those examples into models or parameter fitting. Report 18-way BA and class-level recalls, changed rows, rescues/regressions, prompt-group bootstrap CI, per-source drift and compute cost. A previously used cohort may be a locked historical diagnostic, but not call it a new independent test.
3. **If only unlabeled hidden outputs exist**, report disagreement and stage provenance; do **not** claim a gain, loss or overfit diagnosis.
4. **No requirement to acquire perfect full-18 new data before code research**; but no claim of generalizable 18-class score difference without independent full-18 truth. A source-disjoint subset can falsify the exact subset mechanism only.
5. Major research effort should focus on independent information not already present in strong LoRA/GLM/Fable inference, not more tiny pretrained models, QLoRA or Public-targeted heuristics.

**Latest verified owner Public best: 0.715.** **New real competitive-score result: NONE.**
