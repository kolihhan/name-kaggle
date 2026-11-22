# NTL — Distributed aligned GLM token evidence: real negative result

Date: 2026-11-22. Status: **REJECT C1**.

## Actual completed run

[GitHub Actions 45016238278](https://github.com/kolihhan/name-kaggle/actions/runs/45016238278) SUCCESS, authentic official train (2,699 rows; 150 prompts). Public publisher cache was recovered from pinned notebook version numbers 1 (800 rows) and 2 (1,899 rows). Per-row IDs were parsed with `pickletools` without executing third-party object-array pickle; full/flash token IDs, offsets and prefix counts matched. The source version file checksums appear in the run log; raw data were deleted.

**Mechanism hypothesis**: bursts of model-specific token likelihood evidence spread across disjoint 8-token windows might provide information beyond global likelihood statistics.

**Frozen comparison** (same 300 GLM-labelled train passages, five prompt-disjoint historical folds; LogisticRegression C=1 fixed):

| Arm | GLM binary balanced accuracy |
| --- | ---: |
| C0: native published global feature statistics + global pairwise stats | **0.9433333** |
| C1: C0 + fixed sparse windowed aligned log-prob disagreement | **0.9266667** |
| C2: C0 + same window features after within-half Flash score permutation | **0.9266667** |

C1 against C0: **1 rescue, 6 regressions, net −5**, BA delta **−0.0166666**; fold accuracy differences [−0.033333, 0, −0.033333, −0.016667, 0]. Prompt-cluster paired bootstrap 95% CI [−0.03667, 0]. C1 did not beat the matched C2 negative control. Predeclared mechanism gate FAILED.

## Limits

This is a historical, repeatedly reused official TRAIN prompt-held-out development experiment, **not** independent unseen transfer. C0 is a newly fitted 2-class diagnostic classifier, not the public 0.715/18-way champion. In particular no official score gain, cross-source evidence, or top-2 champion rescue can be claimed from this study.

## Decision

**STOP** distributed token-disagreement as a claimed score-improving mechanism. Do not retune window width, percentile, folds or hyperparameters against this spent cohort. Retain 0.715 as the last official best. Future NTL work should target new information absent from the current global likelihood panel (other family sibling choices, fresh provenance-audited generation or exact champion error-headroom), not another GLM-only prompt/feature tweak.

Sources:
- [Original mechanism code](../../../scripts/ntl_glm_distributed_evidence_20261122.py)
- [Version-safe cache merge](../../../scripts/ntl_merge_public_glm_caches_20261122.py)
- [Executed CPU workflow](../../../.github/workflows/ntl-distributed-glm-evidence-cpu-once-20261122.yml)
