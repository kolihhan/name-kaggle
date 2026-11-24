# Fable-preservation exact-stage replay, 252-row full18 diagnostic (2026-11-24)

## Frozen hypothesis and user restrictions
The low-scoring 18-way research variants must not simply be rejected: determine whether a **specific** strong GLM-to-Fable correction gets lost when final Apex uses weaker sibling-style evidence. Do not train a standalone small model, do not retrain GLM/LLM weights, do not submit to the official competition, do not tune numeric guards after observing this 252 set.

### Provenance
- Exact previously scored **0.718 Public BA** reproduction `kolihhan/ntl-evgen-forensic-eda-repro-20261123`, original private Notebook source SHA256 `74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56`. It is *attributed external source*, not our new classifier.
- Keep all 9 original model, 10 kernel, 26 original dataset dependencies. Add exactly 1 known public labelled external dataset: `nischaydnk/ntl-topic-synthetic-18llms`, pinned `generations.csv` SHA `bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c`.
- Deterministically select the **same** 14 blocks as the November 14 252-row GLM observational experiment (7 short/4 median/3 long, 14 per all 18 classes). This cohort is SPENT and its served-model identity/protocol is unverified; it is **not** a fresh or independent evaluation and cannot authorize competition submission.
- Select input rows **before** native 0.718 scorer. Preserve the original official TRAIN source for existing upstream/pretrained components; fake test has only id/text. True external labels go only into a private diagnostic path and are not passed to scorer, model pipeline, selection, GLM likelihood or routing. Do not print inputs/IDs/logits.

### One intervention = final-stage decision preservation, no new inference head
- Run one pinned original 0.718 scorer on this 252-row development panel; only data path substitution and a bounded 5.8h deadline are allowed, plus read-only trace sidecars. Same exact native inference, GLM weights and pre/post-Apex residual as scored code. No standalone model training; any supervised `Residual.fit` already in the 0.718 original is held unchanged, not an experimental learner.
- Snapshot the post-GLM/pre-Apex prediction before actual Apex. Save existing final-logits and post-GLM logits **inside Kaggle ephemeral memory only**, after checking IDs and class order. The scored notebook's real Apex code remains unmodified.
- After actual Apex, compare:
  * A: exact real 0.718 protected output;
  * B: counterfactual **same-row Apex output without the new Fable protection**; derive it from same pre-Apex predictions, the **same** in-memory already-computed `refined_lp`, and identical frozen typography rules, but with `_protect=0`. No new model forwards/fits for B.
  * C: pre-Apex GLM stage output, no postprocessor.
- Check all observed A-B changed predictions are in the new `fable_return` protected mask. No changes to current submission, all label order/IDs and 14-per-class coverage verified. Record labelled **rescues/regressions, BA18, protected count, actual GLM applied/fallback and real stage runtime**.
- `fable_return` exact original donor logic: pre-v1 GLM family winner, post-v1 `claude-fable-5` winner, after-before margin >=0.30 and final margin >=0.25. These thresholds are **prior-art and frozen**, not ours to tune.
- Stop condition is mechanism-specific: if `applied` but no protected rows in this cohort, result = **NON-DISCRIMINATING** and request different genuinely disjoint data; if protected rows with net gain on this source, result is interesting **exploratory** evidence only. If net loss, diagnose cases/diagnostic source shift instead of declaring evidence conservation invalid. No Public-LB tuning or official submission.
- If rerun has insufficient Kaggle GPU quota, unavailable inputs, incompatible runtime, or missing post-stage logits, report **BLOCKED** without labeling it as negative scientific result.

### Resource control
One new PRIVATE Kaggle T4 two-GPU inference kernel, at most 5.8h native/GLM deadline and 6h overall Kaggle timeout, if quota >=6 GPU hours. Never trigger re-push of existing kernel. No Scheduler.
Run a deterministic CPU-only unit check of the counterfactual simulation and source guards before pushing. All outputs aggregate only; downloaded original source and ephemerally prepared text must be scrubbed. The final output is `ntl_fable_preservation_252_aggregate.json`, never raw row predictions or actual private logits.
