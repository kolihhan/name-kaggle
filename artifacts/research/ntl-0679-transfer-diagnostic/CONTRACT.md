# Frozen 0.679 Champion — Fresh-Source Transfer Diagnostic

Date: 2026-11-11, before any new-model predictions. **One private Kaggle inference-only job. No competition submission.**

## Question

Does the exact source-pinned, no-new-supervised-training `0.679` champion (`58210835`) retain exact-generator identification on 2 new public data-generation sources? The aim is **diagnosis**, not new competitor selection or leaderboard tuning.

## Fixed inputs and labels

1. `toanplam/ntl-fresh-generation-pilot-nov08`: 107 passages from 9 authored prompts, CC0-1.0, 12 labels (8/9 per label), generated 2026-11-08.
2. `nimithashreea/ntl-proxy-candidate-generations-20261110`: 240 passages from 20 Seyit-derived prompts, CC BY 4.0, 20 each for same 12 labels, generated 2026-11-10. Attribute the prompt bank.
3. Six missing labels: `claude-fable-5`, `gemini-3.7-flash`, `glm-5.3`, `glm-5.3-flash`, `kimi-k3`, `qwen3.8-max`. **No full 18-class BA claims.**
4. Source-frozen own Kaggle Notebook: `kolihhan/ntl-ozertuu-0683-guarded-no-train`, original Kaggle submission `58210835` Public BA `0.679`, source **code-cells SHA-256 `28fd128b745f0d95b70f79ea9b2c546d981edf9be08207cbf71a2d9e80ee1c86`** established by [preflight #42955249204](https://github.com/kolihhan/name-kaggle/actions/runs/42955249204). The three source guards must each appear exactly once.

## Predeclared single-run protocol

- Clone code from our **own scored private** Notebook with the same prefit weights, all 18 output classes, all original configs, and all no-training guards. Attach two public generation datasets as Kaggle inputs; set `is_private=true`; **do not create a Kaggle competition submission**.
- In an early new cell, read only the two public datasets locally in Kaggle. Validate label set/counts and unique prompt counts; produce exactly 347 query rows with `id,text`, plus separate ground truth `id,source,model,prompt_id`. Predeclare deterministic interleave/shuffle seed `20261111`.
- Exact UTF-8 whitespace-normalized text SHA overlap with official **training** rows and between the two fresh cohorts must be zero; otherwise abort without scoring. This cannot prove semantic non-overlap, that pretrained weights did not see examples, or exact served-model identities.
- Replace **only** the source Notebook's `DATA` pointer (currently official sample) and `CONFIG['paths']['test_csv']` (after its intact original input audit) with local query. Preserve original model logic and `gsq_share=0.15`, native share `0.43`, target mix `0.52`, and original 18-way decoding. No labels enter model input.
- Confirm `preview=False`, so the 3-row public fixture shortcut cannot mask a full 347-row inference run. Keep guards for supervised Gaussian `gauss.fit`, Atlas train mode and core `--train-final`. Unlabeled query grouping under the frozen source is allowed.
- Once inference finishes, evaluate **12-target-label conditional balanced accuracy with the original 18-way predictions** (mean per-observed-label recall), per-source/per-label/provider recall, off-family errors, sibling confusion, out-of-12 predictions and biggest error pairs. Do not collapse predictions to 12 labels. Report source-specific metrics (Toan and Nimitha), plus combined as exploratory summary. **Never directly compare the conditional 12-target BA to the 0.679 official 18-class Public BA.**
- Save only aggregate JSON and per-row `id,source,prompt_id,expected,predicted` to **private Kaggle Notebook output**; no raw response texts in outputs or GitHub logs/commits. Do not publish the attached source data.
- No weight fitting, calibration, threshold tuning, alternate seeds, GPU training, repeat runs for score selection, or leaderboard submission.

## Result and stop conditions

Success means one private job **COMPLETE** with 347 nonfixture predictions, exact row alignment and source-wise diagnostics. Failure to load all external assets, OOM, hit quota/time, baseline source drift, dataset label mismatch, collisions, or attempted supervision means **FAIL CLOSED**; no altered-model retry without a new documented mechanism. If one source is weak, analyze error categories, but do not promote any new classifier on the score. A claimed 0.70+ breakthrough still requires independent exact-18-label evidence and new model-level improvement.

## Provenance

[Champion official result](../candidates/public-ozertuu-0683-guarded-repro/OFFICIAL_SCORE_20261111.md), [fresh-source triage](ntl-fresh-12-label-cohort-triage-20261111.md), [public-source safety gate](../../PUBLIC_VISIBILITY.md). No source code or example text copied from Kaggle to public GitHub.
