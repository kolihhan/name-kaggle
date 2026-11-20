# 2026-11-20 — NTL sparse-prompt k=2: real retrospective paired experiment

## Verified scientific result

**Conclusion:** k=2 sparse prompt resampling of a **fixed, row-independent OOF predictor** does not repair the known Phase7→Phase15 CV/Public ranking reversal. **It cannot be adopted as a blanket validation fix for NTL.** This is a useful negative diagnosis, **not a new hidden-test oracle**, and does not disprove k=2 for group-dependent centering, assignment, or other methods whose predictions change with available prompt companions.

Actual GitHub Actions [#44660204293](https://github.com/kolihhan/name-kaggle/actions/runs/44660204293) **SUCCESS**. Original historic 2,699-row, 150-prompt, 18-label Phase7 OOF from [#41195328534](https://github.com/kolihhan/name-kaggle/actions/runs/41195328534) and Phase15 OOF from [#41270900644](https://github.com/kolihhan/name-kaggle/actions/runs/41270900644). Exact row-ID/truth/prompt/fold matches; unchanged predictions, no retraining. CPU-only 5,000 independent **paired** row subsets for each k=1,2,4, sampled uniformly without replacement within every original prompt group (same subset for both models, no outcome-aware model/row selection). Random seed 20261120. Aggregate-only JSON artifact `ntl-sparse-k2-aggregate-44660204293`. Implementation: [script](../../scripts/ntl_retrospective_sparse_prompt_k2_ruler_20261120.py).

| Row subsets | Valid draws | BA(Phase15)-BA(Phase7) mean | 95% empirical draw quantiles | Fraction selecting the Public-better Phase7 |
| --- | ---: | ---: | --- | ---: |
| full historical OOF | 1 | **+0.035270** | not applicable | 0% |
| k=1 | 4,988 / 5,000 | +0.035593 | [-0.054979, +0.123251] | 21.6921% |
| **k=2** | **5,000 / 5,000** | **+0.035148** | **[-0.022903, +0.093134]** | **11.9%** |
| k=4 | 5,000 / 5,000 | +0.035205 | [-0.002089, +0.073176] | 3.1% |

Historic official scores: Phase7 **0.369**, Phase15 **0.357**: **Public −0.012** even though full OOF **+0.035270**. The selected Public winner for k=2 happens only in 595/5000 random sparse draws. **Mean is unchanged**; some draws reverse because of extra sampling variance, not successful distribution-shift correction.

These intervals describe variability across random k-thinned draws of one fixed dataset, **NOT confidence intervals for actual hidden-test generalization**. Source support/serving mismatch and individual Public score noise can both explain part of the reversal. Repeatedly sampling a spent 150-prompt OOF does not add independent examples. Only two historic models were included, so estimating a CV/Public Pearson correlation or any general model-ranking calibration is prohibited. This particular diagnostic does not test k=2 group-dependent prediction methods.

## Immediate resource decision

1. **NO** further GPU or LoRA tuning just to implement universal k=2 evaluation. If a future method uses grouping/centering/assignment at inference, test its prediction semantics under k=1,2,4,18 separately.
2. Own 0.702 published-asset champion still lacks lineage-clean full-ensemble OOF. Some publicly trained artifacts have official-train exposure, so running that prefit ensemble on train.csv is **in-sample**, not OOF. Fold-by-fold retraining or source-verified independent aligned evaluation would be expensive; **defer** until a concrete candidate survives cheaper legitimate source/served-model verification. Do not promise zero-cost strong-champion complementarity.
3. F1 Colab [#44544600364](https://github.com/kolihhan/name-kaggle/actions/runs/44544600364) actually started with owner ADC credentials, but **CANCELLED** after 2h35m with `Connection was lost`, no result artifact, no C1/C2. It is an infrastructure failure, not a science failure. Do not blindly re-run the full extractor; require separate short throughput/pretrained-model-revision probe and checkpointed batches before any expensive GPU.
4. Own Stage1 A1 Qwen3-8B LoRA24 already underwent a paired 347-row/12-label old-0.679-champion source test: 24 rescues and 66 regressions (net -42), negative on both sources. [Result](../ntl-lora24-a1-paired347/RESULT.md). This is *existing complementarity evidence against unconditional replacement*, not 0.702-model complementary evidence, and is adaptively spent. No new family gate may be tuned from it.
5. No Apex 1.0 official probe. Archived 23/216 true-label typography exposures do not establish 0.702 actual rescues; one noisy official public run would not be high-value absent real score vectors. Keep champion official **0.702 (#58514818)** immutable.

## Next high-information task

Review provenance and existing frozen 347-row mismatches to **select a failure class only as a future hypothesis**, not to fit a threshold. A new prospective test should involve genuinely disjoint generator/serving provenance, targeted 18-label support where possible, and a same-row strong comparator. Until an admissible inexpensive candidate can answer a specific new mechanism, spend **no GPU** on F1 v0 rerun or full 0.702 fold retraining.

**Actual competition submission change: NONE.** This is a CPU mechanism-rejection study, not an accuracy gain.
