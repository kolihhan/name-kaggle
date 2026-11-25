# NTL Fable252 v2 — executed real result, 2026-11-25 13:24 MYT

## Authenticated result
- The *previously accepted* Kaggle private inference kernel `kolihhan/ntl-fable-preservation-252-v2-20261125` now reports **`KernelWorkerStatus.COMPLETE`**.
- [Authenticated GitHub Actions read-only aggregate #45553286140](https://github.com/kolihhan/name-kaggle/actions/runs/45553286140) downloaded the exact private output file `ntl_fable_preservation_252_aggregate.json`. This is evidence of actual completed GPU inference and observational labelled replay, not an official leaderboard submission.
- Frozen source: recovered Evgen public historical **v36**, all 58/58 cell prefixes and lengths verified, historical source SHA256 `5988d64922650bf94490810effb7278907b301c07f581efbc6d4865d84e63d52`. [CPU preflight #45541570603](https://github.com/kolihhan/name-kaggle/actions/runs/45541570603) and private launch [#45541703599](https://github.com/kolihhan/name-kaggle/actions/runs/45541703599) passed.

## Exact frozen 252-row 18-model observations

| Pair | Before | After | Rescues | Regressions | Changed predictions | Difference |
|---|---|---|---:|---:|---:|---:|
| Unprotected Apex counterfactual vs actual *Fable-protected* Apex | 149/252 = **0.5912698413** | 151/252 = **0.5992063492** | **2** | **0** | **3** | **+2/252 = +0.007936508 (0.79 pp)** |
| Real pre-Apex vs actual Fable-protected Apex | 150/252 = **0.5952380952** | 151/252 = **0.5992063492** | **8** | **7** | **22** | **+1/252 = +0.003968254 (0.40 pp)** |

- `v1_status="applied"`, `real_glm_stage_run=true`; the GLM likelihood step actually completed.
- `fable_protection_eligible_rows=3` and `protected_final_predictions_changed=3` after original Apex. **2 previously incorrect Fable truth rows rescued** and **0 previously correct rows regressed**; third changed prediction did not change correctness.
- Claude Fable 5 truth class: **6/14 correct in unprotected counterfactual → 8/14 with protection**.
- GLM 5.3 truth class: **7/14 → 7/14**; GLM 5.3 Flash: **5/14 → 5/14**. The remaining one changed row is not specified by the aggregate; do not invent its true label.
- Frozen candidate field `candidate_promote=false`: the result is observational, NOT a competition promotion.
- 252 rows comprise a *historically spent*, publisher-labelled 18-way external synthetic sample (14 responses for each of 18 classes). Provider/model identity not independently authenticated. The 59.92% **external-panel BA is not directly comparable with the official Public 0.718**, and is not a new Kaggle submission score.

## Statistical and scientific reading
- The **mechanism delivered a positive direction** on the sampled source: net +2 exact matched responses and no correctness regressions in the observed comparison. The 3 protected decisions make this a **small-n, fragile result**. The exact two-sided matched-pair sign/McNemar test on *correctness-discordant* rows (2 rescues, 0 regressions) is **p=0.5**. It does not establish a reliable generalization gain.
- Crucially, do not conflate the **+2** isolated Fable guard effect with the **+1** effect of the entire Apex stage relative to pre-Apex. The rest of Apex is not all positive: **8 rescues / 7 regressions** across 22 changed predictions.
- The post-hoc evaluation is not a freshly held-out source-provenant benchmark. Do **not** tune thresholds to these 252, run the same cohort again searching more favourable numbers, or extrapolate to Public 0.718→0.726.

## Actionable decision
- **Result: POSITIVE BUT NOT PROMOTABLE**. No fresh neural training, no small classifier, and no official Kaggle submission was launched by this read-only verification.
- Focus the next *different* experiment on **paired genuine full18 original 0.718 predictions** in source-diverse, independently verified data; measure same-row rescues/regressions and 9h total runtime. One small positive observation doesn't justify another 9h Kaggle rerun of the same guard.
- Preserve recovered v36, regression smoke test and 61-cell launch logic; the original v1 `NameError` failed prior to science, while this corrected v2 reached scored observational output.
