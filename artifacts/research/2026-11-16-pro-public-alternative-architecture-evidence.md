# NTL PRO — independent external data and live public alternate architecture audit (2026-11-16)

## Goal and current baseline

Own **authenticated official Kaggle Public Balanced Accuracy = 0.691**, submission #58375806. Target ≥0.700. Reproduce or discover a genuine mechanism that improves out-of-source **18-label attribution**, not more leaderboard thresholds, used official TRAIN folds, stale preview shortcuts, or an arbitrary source-domain benchmark.

## New real public Kaggle notebook inventory

- Authenticated Kaggle public [notebook registry Action #43809843346](https://github.com/kolihhan/name-kaggle/actions/runs/43809843346) retrieved the most recent **300** public notebooks across three pages (public list sorted by hotness). This source reports **notebook titles and votes, NOT authentic public leaderboard scores**.
- Distinct candidates on 2026-11-16:
  - [Ertuğrul Özer, *Residual Sibling Gating Apex Better!*](https://www.kaggle.com/code/ozertuu/residual-sibling-gating-apex-better)
  - [Saner Turhaner, *0.693 · Real GLM-5.3 & GLM-5.3-Flash Likelihood*](https://www.kaggle.com/code/sanpier/0-693-real-glm-5-3-glm-5-3-flash-likelihood)
  - [lbjyang, *ntl 700 v3 share40 cover20 instrumented*](https://www.kaggle.com/code/lbjyang/ntl-700-v3-share40-cover20-instrumented)
- **Important:** the strings “0.693” and “700” are titles, not our own verified submitted scores; do not report them as gains or as independently sourced scores.

## Evidence of real different mechanisms

[Authentic static source/module fingerprint comparison #43810053266](https://github.com/kolihhan/name-kaggle/actions/runs/43810053266) checked the actual public Kaggle Notebook code and metadata for four new candidates. [Pinned source module and asset delta #43810418154](https://github.com/kolihhan/name-kaggle/actions/runs/43810418154):
- New OZERTUU/Sanpier lineage **24–26 code cells, 26 dataset mounts, 9 model mounts, 9 kernel mounts**. Frozen own 0.691 champion: **50 code cells, 23 dataset mounts, 6 model mounts, 9 kernel mounts**.
- Compared with own champion, these new notebooks add **3 model resources**: `alexchilton/llama-3-1-8b-base`, `qwen-lm/qwen-3/8b`, `qwen-lm/qwen-3/14b` (exact Kaggle model sources are in the Action). They also add **3** dataset resources, and replace rather than simply tweak the main modules. **Zero common exact module hashes**. Thus the perceived advantage cannot be attributed solely to GLM likelihood, a sibling gate, or one blend parameter. Total runtime/model memory and hidden performance **unknown**.
- [The 0.700-titled lbjyang fork exact cell diff #43810671569](https://github.com/kolihhan/name-kaggle/actions/runs/43810671569) used the same **50 code cells** and same dataset/kernel assets as own 0.691, but changed **7 cells** including input/router logic, a share scalar **0.36 → 0.40**, added cover weight **0.20**, and had DIFFERENT model-source metadata. Multiple simultaneous differences mean this is **not** a clean one-variable promotion experiment. Title is not an authentic official score.

## New data provenance attempt

- A separate paid-API generation source, [Maniceet Sahay's OpenRouter + OpenAI corpus](https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations), publicly claims **7,753** rows, **10** labels and per-row actual served-model provenance. Crucially it does not cover all 18 classes, and `own` rows may not contain raw request/return model IDs.
- Authenticated [Kaggle source intake #43810239396](https://github.com/kolihhan/name-kaggle/actions/runs/43810239396) **FAILED (access denied)** for the stated dataset route. No file schemas or true eligibility counts were fetched; do not claim this source is a proven 10-class heldout or try to rely on unavailable input. Do not mirror source text publicly.
- Previous Oleg external source `0.412` 18-class stress test is now spent, and 216/216 rows were from `sensitive_topics`; 10-label prose source has only four all-model-shared prompt groups and cannot support a fair 10×12 independent test. These are **not** valid adaptation targets.

## Falsifiable next research, not a submission

**H_model_attribution_new_representation:** OZERTUU/Sanpier stages provide genuinely complementary prefit model information; test matched same-row **18-way** out-of-source predictions when an admissible source exists, with class-level rescues/regressions (not just an aggregate or guessed title score). Dominant alternative **H_proxy_dataset_and_routing**: different inputs/model mounts or adaptive class mass calibration artificially boost the public leaderboard while reducing private/source transfer.

**H_prompt_conditioned_assignment:** prompt-centering/group assignment can suppress topic artifacts but may force false equal-model quotas because hidden model mixture is not uniform. Only advance if a source-provenance-safe paired prompt cohort shows improvement across prompt-disjoint clusters; avoid fitting quota on spent public/train data.

**STOP:** No new official Kaggle submission based on one public title or train-spent improvement; no new model training under current track. Potential future candidate requires exact frozen source, accessible prefit assets, production true inference with no 3-row preview shortcut, compliant license, stable runtime and a credible paired generalization test. Preserve official champion 0.691 unless evidence is stronger.

No new GPU, no new competition submission and no training occurred in this audit.
