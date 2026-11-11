# Kimi K3 reasoning high/off — final matched-pair result

**Status: COMPLETE; FAIL the original promotion/evidence criterion.** Verified 2026-11-11 after [one private inference-only Kaggle job / GitHub Actions #42974357365](https://github.com/kolihhan/name-kaggle/actions/runs/42974357365) (job `130369860929`, conclusion **SUCCESS**, output printed 2026-11-11 07:50:38 UTC). This is not a Kaggle competition submission.

## Before-result provenance

- [Precommitted new 80-pair contract](CONTRACT.md); source `willrice/name-that-llm-openrouter-harvest`, file `setting_trials.jsonl`, SHA-256 `a38745dd0b648773204193d2a34a9b1ccd5ce9a41cfd01b51eea49ceed56d87e`, licensed CC BY 4.0 with source credit.
- Exactly **80 matched Kimi K3 prompt IDs × 2 reasoning conditions = 160 response texts**. Same non-null requested/served model ID, provider and target words across each pair; finish_reason `stop`; 40–250 observed words.
- Frozen **own scored Public BA 0.679** Kaggle champion (`58210835`); original 18-label unrestricted inference, full assets and three **no-supervised-training** guards, no new model updates, no new submission.
- Previously inspected `calls.jsonl` source offered 1,000 nominal pairs but only **12** with identical target lengths; the separate `setting_trials.jsonl` was selected after this failed feasibility gate and *before* champion predictions. This is a sequential **data-source selection**, even though the new pair-level result criterion was fixed before inference.

## Verified predictions and predeclared test

| Paired Kimi K3 outcome | Value |
| --- | ---: |
| Both conditions classified correctly | 7 / 80 |
| Both wrong | 22 / 80 |
| High correct / Off wrong | **35 / 80** |
| Off correct / High wrong | **16 / 80** |
| Reasoning high recognized Kimi K3 | **42 / 80 = 52.50%** |
| Reasoning off recognized Kimi K3 | **23 / 80 = 28.75%** |
| Paired high-minus-off recall | **+23.75 percentage points** |
| Exact two-sided discordant-pair binomial / McNemar p | **0.010973563** |
| Identity output changed between conditions | **73 / 80 pairs = 91.25%** |
| Average observed length difference (high − off) | **+6.6 whitespace-delimited words** |

Four predeclared sorted-prompt blocks, 20 pairs each: high-minus-off correctly classified **+3, +2, +9, +5** = **+15, +10, +45, +25 percentage points**, so direction is positive in all four blocks.

**Decision:** Predeclared criterion required a difference >=15 pp, same direction in 4/4 blocks **and exact p < 0.01**. The first two pass; **p = 0.010973563 does not pass**. The source code's `evidence_criterion_met = false` is correct. No re-thresholding to `p < 0.05`, post-hoc selection of blocks, inference rerun, protocol-specific correction, or model promotion is authorized.

## Interpretation — deliberately narrow

- **FACT:** For this one source and Kimi model, the frozen model's predicted identity depends strongly *descriptively* on two reasoning settings; the high/off text lengths and style marker rates differ. The same model is recognized substantially less often in `off`.
- **INFERENCE:** The classifier may be using protocol/channel correlated features rather than invariant generator identity. A high/low comparison on one exact model is a **diagnostic of sensitivity**, not evidence of a deployable 18-class correction.
- **UNRESOLVED:** Exact request text/other sampling settings, immutable provider weights, semantic prompt equivalence, prior exposure through the champion's prefit assets, and transfer to other models. Because this Kimi bank was sought after previously seeing source and error diagnoses, the quoted p-value cannot serve as independent confirmatory evidence of the *entire adaptive research program*.
- **NOT A FINDING:** "Reasoning mode causally changes attribution accuracy" or "a Kimi protocol detector will improve hidden Kaggle BA". Neither is established.
- This sample has now been observed and is **SPENT for model selection**.

## Evidence preservation

Private per-row passages and predictions remain with the private Kaggle Notebook `kolihhan/ntl-kimi-k3-protocol80-frozen-0679`; only aggregates, SHA and study decision are copied into this public repo. [Original execution](https://github.com/kolihhan/name-kaggle/actions/runs/42974357365) records the `NTL_KIMI_PROTOCOL80_FINAL_VERIFIED` aggregate. Competition Public champion remains **0.679**, submission **58210835**, and there was **no** new Kaggle competition submission.

## Next-action decision

**Close the Kimi high/off lane as an evaluated, non-promotable diagnostic.** Do not build a Kimi exception from these observations. Prioritize an independent **model × prompt × protocol** coverage inventory; if no suitable data exist, declare full-18 transfer blocked rather than running another GPU experiment.
