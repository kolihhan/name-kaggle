# PB115 token-aligned evidence — actual CPU result, 2026-11-25

[Completed authenticated diagnostic](https://github.com/kolihhan/name-kaggle/actions/runs/45557688248), job 139013603770. No new fitted classifier, GPU inference, submission, or raw data publication. Downloaded the public LumiBee PB115 feature package and official train in an ephemeral runner, verified one-to-one coverage and finite feature values. The 2699 rows / 150 prompt groups are historically spent development data.

## Question and limitations
Does token-level GLM evidence improve ranking where the old scalar likelihood fails, or largely repeat the same evidence? Controls: signed full-minus-Flash mean likelihood (sibling), maximum of the two mean likelihoods (GLM vs other). These are scalar controls, NOT the full historical 54-D classifier or actual 0.718 ensemble. No fitted decision boundary. This is a descriptive rank comparison, not a candidate accuracy experiment, OOF, fresh transfer, or leaderboard score. Both directions were reported; best orientations below are post-hoc summaries, not selected prediction rules. Cross-family pairs are clustered by 150 prompts, not 4798 independent trials. No causal claim about v66's +0.004.

## Real results
- Sibling: 150 matched pairs, scalar rank 0.98, 3 wrong. Token-level features have at most 3 baseline-miss opportunities in this cohort, insufficient to establish a reliable incremental gain.
- GLM vs non-GLM: 4798 same-prompt pairs, scalar rank 0.797416, 972 wrong. 0/61 individual token features exceed scalar rank even with post-hoc orientation.

| Feature | Post-hoc sign | Sibling rank | Rank on baseline misses |
|---|---:|---:|---:|
| p:dlp_tstat | - | 0.993333 | 0.666667 |
| p:dlp_pos | - | 0.993333 | 0.666667 |
| p:dlp_q25 | - | 0.993333 | 0.666667 |
| p:dlp_big_neg | + | 0.986667 | 1 |
| p:dlrk_tstat | + | 0.986667 | 0.333333 |
| p:dlp_mean | - | 0.98 | 0 |

| Feature | Post-hoc sign | GLM-vs-other rank | Rank on baseline misses | Rank on baseline wins |
|---|---:|---:|---:|---:|
| glm53:lp_tail32 | + | 0.737182 | 0.4393 | 0.812859 |
| flash:lp_tail32 | + | 0.724469 | 0.345679 | 0.8207 |
| p:both_t1 | + | 0.716861 | 0.249486 | 0.835599 |
| glm53:logrk_hi_ent | - | 0.705294 | 0.335391 | 0.799268 |
| glm53:lp_hi_ent | + | 0.702376 | 0.353909 | 0.790904 |
| glm53:lp_head32 | + | 0.698833 | 0.367284 | 0.783063 |

The sibling token ranks show strong signal, but old scalar rank already 98%; the highest-ranked cross-family feature is below scalar control. Some token features distinguish old scalar failures, but this does not establish independent incremental signal after the real 54-D head or final 18-class ensemble. No weights, thresholds, or feature subset tuned on these observations.

## Decision
Keep 0.718 intact. Do not blindly port PB115, train LR/LGB, or claim that all token features are ineffective. Focus the next distinct mechanism on independent generator-specific evidence for cross-family attribution, or a frozen inference-only conditional complement to the exact champion, rather than more sibling calibration. Require actual 18-class paired incumbent predictions for any rescue/regression claim. Existing official train data are diagnostic only. No official submission triggered. 

## Reproduction
Source hashes: `{"flash_features.csv":"35c2b383fb45d4797091d401f7e9d951051d68dbed8cb8eb6fd2c66ee5394f36","glm53_features.csv":"882b695788b06fdd5717bcab1cce663ea26628cecc6dcca237c3255f92a9817b","paired_token_features.parquet":"55860b37a488c5a94533524973666b25b2474df1649a86ba11f1c62f1d61210a","spec.json":"8b9513abdd1c07497a7a11daa3fbaa0c551bf22f120b51a974e8129124521488"}`. Full 61-feature directional results are in the linked job log. Code: `.github/workflows/ntl-pb115-token-evidence-20261125.yml`. 
