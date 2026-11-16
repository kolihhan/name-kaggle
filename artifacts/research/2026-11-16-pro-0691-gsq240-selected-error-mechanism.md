# PRO: frozen NTL 0.691 Lightning GSQ-selected error anatomy, no leakage

Date 2026-11-16 MYT. This is a **post-inference retrospective diagnostic on the first 1,200 historical official TRAIN rows**. These rows are spent and may have been present in upstream fitted artifacts; this is **not an independent external 18-class validation** or proof that GSQ caused improvement.

## Exact authentic evidence

- Frozen own best [Kaggle #58375806, Public BA 0.691](https://github.com/kolihhan/name-kaggle/actions/runs/43743764085), previous best 0.683.
- Actual existing true 1,200-text Lightning V2 predictions SHA256 `926dffc6ab0909f63767e8fcec7ca9488f7d1dc7c49656b0eaa8cf50064e0826`; 1,149/1,200 correct, **51 errors**, vs 18 exact official labels.
- Read-only, trained-weight-inference-free [GSQ residual targeting #43780908827](https://github.com/kolihhan/name-kaggle/actions/runs/43780908827) and [additional frozen class-family decomposition #43781225182](https://github.com/kolihhan/name-kaggle/actions/runs/43781225182) **SUCCESS**.
- Two **SHA-pinned** `v1/flash_scores.npz` and `v1/glm53_scores.npz` contain **240** opaque-ID rows each. Recovered exact `scale1200_####` IDs **without executing any object pickle**, by inspecting NumPy header and `pickletools.genops` only. Sets were identical (**240/240 overlap**); numerical `scores` are token-level 4-column arrays, **not** full `[1200,18]` 18-label logits.

## Observed, not causal, residual errors

| Condition | N | Final wrong | Final wrong rate | Cross-family mistakes | Same-family sibling mistakes |
|---|---:|---:|---:|---:|---:|
| Both GSQ experts selected | 240 | **33** | **13.750%** | **27** | **6** |
| Not selected | 960 | 18 | 1.875% | 3 | **15** |
| Full spent sample | 1200 | **51** | **4.250%** | 30 | 21 |

- Only **20%** of rows selected, but contain **33/51 = 64.706%** of residual misclassifications. Selected error rate is **7.333×** nonselected.
- Among the selected *errors*, **27/33 = 81.8%** confuse the **model family** itself, versus **6/33 = 18.2%** sibling within family.
- Among **unselected errors**, **15/18 = 83.3%** are sibling confusions and 3/18 cross-family.
- Selected errors by **true** family: Kimi 13, GLM 12, Anthropic 7, Google Gemini 1. These are **not class priors for hidden transfer** and must not become model-specific replacement rules.
- Example whole-sample confusion counts: `kimi-k3 → claude-fable-5` 6, `kimi-k3 → glm-5.3` 4, `gemini-3.7-flash → gemini-3.8-flash` 3.

## Decision (Yuanzhe-style competing hypotheses)

**H_sibling-only** cannot plausibly repair most *selected* residual failures: at most **6/33** selected wrong rows are same-family by definition, barring changing family identity too. Retire the **selected-cohort sibling-only** hypothesis as dominant. Do **not** generalize this to the full hidden-test set or claim all sibling specialists are useless.

**H_cross-family representation/gate** is more consistent with selected residuals, but two competing causes remain: (a) truly confusable generator traits requiring stronger model-identity features, or (b) prompt/source/channel shift causing overconfident family assignments. The next scientific intervention must distinguish them using *source- and prompt-disjoint 18-class same-row paired evaluation* against the **unchanged** 0.691 champion. Avoid fitting 18-class thresholds or hand-label-specific rules on the spent 1,200 sample.

**Causal caveat**: these are *post-GSQ final predictions*, so selected rows having more errors does **not** establish that GSQ increased errors, failed to fix them or was the reason for higher official Public BA. A real pre/post stage ablation would require matched input-row 18-class score vectors; the saved `[27869,4]` expert token scores are insufficient. **No actual component-off experiment was claimed or performed.**

No new GPU, model training or Kaggle competition submission. Official champion remains **0.691**.
