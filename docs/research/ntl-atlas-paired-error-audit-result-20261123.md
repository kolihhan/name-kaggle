# ATLAS A0 paired-error attribution — completed result (2026-11-23)

**Execution:** [GitHub Actions #45170089900](https://github.com/kolihhan/name-kaggle/actions/runs/45170089900) SUCCESS.  
**Script:** [ntl_atlas_paired_error_audit_20261123.py](../../scripts/ntl_atlas_paired_error_audit_20261123.py)  
**Frozen contract:** [ntl-atlas-paired-error-audit-20261123.md](ntl-atlas-paired-error-audit-20261123.md)

## Reproduction and paired errors

The audit exactly reproduced the prior result on the same 300 genuine GLM-5.3 / GLM-5.3-Flash labelled official TRAIN rows, 150 prompt groups, and five prompt-held-out folds:

| Scorer | BA |
|---|---:|
| C_global, existing 53-feature supervised control | **0.943333** |
| C_ATLAS_prod, exact fixed-T argmax-prod readout | **0.913333** |

Of the 300 predictions, 279 agreed. The 21 disagreements were:

- **15 regressions:** C_global correct, ATLAS wrong (12 GLM-5.3, 3 Flash).
- **6 rescues:** C_global wrong, ATLAS correct (2 GLM-5.3, 4 Flash).
- Net ATLAS change: **−9 correct / −3.00pp BA**.

## Finding 1 — ATLAS margin is mostly redundant with existing pairwise features

ATLAS product margin association on the same 300 rows:

| Existing evidence | Pearson r | Spearman ρ |
|---|---:|---:|
| C_global OOF decision margin | 0.939 | 0.935 |
| Pair mean log-likelihood difference | 0.931 | **0.965** |
| Token win fraction | 0.919 | 0.918 |
| Mean token sign | 0.919 | 0.918 |
| Pair p90–p10 spread | −0.190 | −0.180 |
| Pair score correlation | 0.042 | 0.046 |

This is the central result: the ATLAS scalar is highly aligned with evidence already represented by C_global's mean-difference and token-win summaries. It is not showing strong independent movement on this cohort.

## Finding 2 — C_global's 15 correct calls use distribution evidence ATLAS compresses away

For the 15 C_global-correct / ATLAS-wrong rows:

- C_global's true-label-signed margin had median **+1.810** (10th percentile +0.238). These were not all marginal C_global decisions.
- ATLAS's true-label-signed margin had median **−0.0205**.
- The strongest average C_global contributions supporting the true label included Flash top-5, pair score correlation, GLM-5.3 lower likelihood quantile, Flash surprisal, and Flash log-rank.
- The largest opposing contributions included GLM-5.3 likelihood q25, top-1, and mean likelihood.

Interpretation: the stronger model is using a broader distribution/style profile across both sources; the bounded ATLAS readout does not preserve enough of that information.

## Finding 3 — the six ATLAS rescues are weak-margin and too few to establish complementarity

For the six C_global-wrong / ATLAS-correct rows:

- C_global's true-label-signed margin median was **−1.476**.
- ATLAS's true-label-signed margin median was only **+0.0232**.
- Plain mean-loglikelihood's true-label-signed margin median was **+0.1304**.
- The top supporting C_global contributions were GLM-5.3 surprisal, top-1, likelihood standard deviation, and pair mean difference; the largest opposing terms included Flash surprisal and Flash minimum likelihood.

The six rescues are real paired corrections, but ATLAS's own margins are small and the group is too small to demonstrate a stable new signal.

## Independent cross-check — learned fusion also failed

A separate nested prompt-grouped experiment added 13 token features, including the ATLAS product margin, to the same 53-feature global block. [Actions #45169232586](https://github.com/kolihhan/name-kaggle/actions/runs/45169232586) returned:

- Global-only: **0.943333 BA**.
- Nested global + token: **0.923333 BA** (−2.00pp).
- Paired: **1 rescue / 7 regressions**.
- Prompt-bootstrap 95% interval for the delta: **[−4.00pp, −0.33pp]**.
- Promotion gate failed.

This is not a test of the full trained ATLAS paper. It is evidence against adding this cached GLM token-evidence block to the existing global model on the repeatedly consumed 300-row subset.

## Decision

**Park ATLAS A0 and stop token-readout / token-feature fusion work on this GLM subset.** Do not sweep temperatures, tune thresholds, or fit a correction on the 21 discordant cases. The evidence says ATLAS A0 mostly repackages already-used token comparisons, while C_global retains additional cross-source distribution information.

Limits:
- The 300 rows are repeatedly consumed official TRAIN data, not prospective transfer.
- This is GLM-vs-Flash binary mechanism evidence, not an 18-class result.
- Feature contributions describe fold-model arithmetic; they do not establish causal feature importance.
- Full ATLAS source-conditioned training was not reproduced or falsified.

No new model forward, paid API, QLoRA, GPU, or Kaggle submission. **Official 18-way Public champion remains 0.715.**
