# NTL — real source-locked host-match proxy feasibility (2026-11-20)

## Status: COMPLETED, CPU only; NOT an accuracy gain

- [GitHub Actions run #44710955318](https://github.com/kolihhan/name-kaggle/actions/runs/44710955318): terminal **SUCCESS**. Script: [source-controlled metadata gate](../../../scripts/ntl_hostmatched_source_stratified_feasibility_20261120.py).
- Exact Open Pool SHA256: 7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492; no pool drift.
- Reconstructed the prior PC-CL **8,347** external-source training rows, held-source validation **1,800** rows with 18 labels (NOT an untouched new evaluation). Same source, prompt, flags, length, publisher-license-metadata and served-model-publisher-claim filters.
- Constructed **2,095** top-weight examples versus **2,095** fixed-random examples, **identical number per (publisher source, label) stratum**. No model training or new GPU. This isolates source-label composition from within-stratum weight selection, but NOT authentic serving model identity, license, or the hidden-test domain.

| Actual fixed-matched CPU metric | Source-locked high match_weight | Same source-label random |
| --- | ---: | ---: |
| Rows | 2,095 | 2,095 |
| Median match_weight | **1.8301** | **0.8722** |
| 75th percentile | 2.5382 | 1.4050 |
| Numeric distinct weights | 942 | 1,350 |

**Concentration:** Between-publisher source variance explains **0.013801** of observed match_weight variance in the eligible training set (descriptive ANOVA eta-squared, not proof of source independence). The weight carries substantial within-publisher variation.

**Predeclared strict sample-sufficiency gate FAILED** for a uniform 18-class high-match quarter-sample: 11 of 18 classes have **fewer than 100 high-match selected rows**; 7 have **fewer than 2 represented training publisher sources**. The seven are: claude-opus-4-6, claude-opus-5, gemini-3.1-pro-preview, gemini-3.6-flash, gpt-5.5, gpt-6-astra, qwen3.8-max. This is a problem with THIS 25% subsample, NOT proof full-data weighting is impossible.

Weak7 selected row counts: Gemini 3.1 **78**, 3.6 **80**, 3.7 **142**, 3.8 **105**, GLM 5.3 **175**, GLM 5.3 Flash **182**, Kimi K3 **129**. Counts are not permission to train or proof those examples match competition serving conditions.

## Decision and scope of inference

1. **Positive feasibility**: equal-budget, exact source- and label-controlled match proxy separation is possible. This answers the first prerequisite that the earlier 160-step PC-CL experiment never tested.
2. **Negative readiness**: do not promote this small top-quarter set as a new 18-label training baseline, claim a 0.702 complement, or launch QLoRA merely because its weight median is higher. There is no scored BA, verified API identity, original-source output-right clearance, or independent test transfer.
3. **Prospective replacement candidate, NOT executed**: keep the **full** 8,347-row / full-18 class source-label mix (avoid dropping weak labels) and compare fixed uniform training exposures against fixed within-(source,label) match-weighted exposures. Freeze exposure count, class/source composition, prompt-disjoint validation and one model/optimizer before GPU. Report paired BA18 per-source/per-class and rescue/regression, including weak labels. Do not rerun the spent 1,800 rows as "fresh" validation, nor claim an improvement over 0.702 without matched strong-model evidence or authenticated official scoring.
4. **Data eligibility gate**: verify original publisher data rights and provider-generated-output policies for each included source before training; dataset license metadata and publisher-reported model identity do NOT satisfy this. If rights cannot be established, use verified admissible sources or authorized fresh generation, not a silent override.
5. Maintain the exact official best **0.702** (#58514818). No Kaggle submission, new GPU, or repeated quantile/seed tuning on spent cohort.

The successful Action verifies unit test, upstream SHA guard, public aggregate-only export and ephemeral downloaded-text cleanup. This report documents data feasibility, NOT a model gain.
