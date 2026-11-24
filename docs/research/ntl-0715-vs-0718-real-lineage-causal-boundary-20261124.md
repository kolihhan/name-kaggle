# NTL 0.715 versus 0.718 — source-verified causal boundary (2026-11-24)

## Evidence
Authenticated read-only Actions [#45395458628](https://github.com/kolihhan/name-kaggle/actions/runs/45395458628), final code reachability [#45396243540](https://github.com/kolihhan/name-kaggle/actions/runs/45396243540), embedded-stage diff [#45396586997](https://github.com/kolihhan/name-kaggle/actions/runs/45396586997), and final Apex-cell diff [#45396809915](https://github.com/kolihhan/name-kaggle/actions/runs/45396809915). No training, Kaggle GPU, hidden test text, output predictions, competition data or official submission used by these audits.

Two **previously scored** pinned user-owned PRIVATE notebooks:
- 0.715 official Public BA, submission `58680531`, `kolihhan/ntl-estopadilla-glm53-0716-repro-20261122`, SHA256 of concatenated code cells `202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8`.
- 0.718 official Public BA, submission `58755304`, `kolihhan/ntl-evgen-forensic-eda-repro-20261123`, file matches the original public Evgen notebook SHA256 `74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56`. This is **external public code reproduced**, not own original GOLD.
- Both have identical 9 model, 26 dataset, 10 kernel sources. All 15 script modules shared by both notebooks are byte-identical: including `infer.py`, `pngy_residual.py`, `gauss.py`, `ntl660_readout.py` etc. New notebook adds only `prefix_variant_stage.py` and `prefix_decisions.py` as %%writefile modules, plus extra orchestration/EDA cells.

## What actually changed, not what file names suggest
**Confirmed active installation path:** New notebook's setup cell asserts original embedded `v1_stage.py` SHA256 `d88b0834a751770a2c250151b683701309cc66316a1b8af396e5c33e8eed5fff` and executes `_stage_path.write_bytes(Path('/kaggle/working/prefix_variant_stage.py').read_bytes())`. The later stage runs `V1_DIR/'v1_stage.py'`. Thus the variant IS used; earlier suspicion that the new file is dead was falsified.

**Stage diff:** Original embedded `v1_stage.py`: 261 lines. Installed new stage: 265 lines, precisely **4 lines added and 0 removed**, no new functions. The material new side effect is saving `/kaggle/temp/prefix-votes.npz` with post-stage logits, aligned IDs and labels; *this added file is not by itself an altered prediction rule*. It supplies evidence for the final-stage protection.

**Final Apex postprocessing:** one actual old vs new code cell compared, **13 added lines and 1 removed**, with `from prefix_decisions import fable_return` absent in old and present in new. New V1 `report.json` status must be `applied`; post-stage argmax is checked against the pre-Apex submission; the `prefix-votes.npz` logit matrix is read. New protection ORs into existing GLM mask, preventing the later weaker sibling-style output from overwriting a selected **GLM→Claude Fable** flip. The fable-return helper requires all of:
  1. pre-stage argmax is `glm-5.3` or `glm-5.3-flash`;
  2. post-GLM argmax is `claude-fable-5`;
  3. Fable's margin over the top GLM increased by at least **0.30**;
  4. Fable's final margin over top GLM is at least **0.25**.
**Important:** this is a preserved third-party mechanism. Neither the 0.003 point Public improvement nor our source audit proves those protected changes were correct.

**Unused function:** `sibling_agreement(...)` appears only as a definition in `prefix_decisions.py`, with zero direct call sites in the exact published Notebook code. Do not falsely credit it for 0.718.

**Preview false negative resolved:** `FAST_COMMIT` uses a cheap preview path without actual model/GLM inference; true `final-logits.npz` is absent on that branch. Hence older sidecar's `NTL_TRACE_PRE_GLM_NOT_VERIFIED=FileNotFoundError` and missing post-stage audit are expected from a 3-row preview. This DOES NOT prove failure or success in full official rerun. The old sidecar experiment had no genuine post-stage measurement.

## Rival explanations for score difference
- H1: cross-family Fable protection prevents incorrect post-GLM erasure of correct large-likelihood changes. **Prediction:** an exact same-row old-vs-new Apex replay differs only on the protected GLM→Fable subset, with rescues > regressions on independent truth.
- H0: public BA difference (+0.003) results from runner variability, label mix, other orchestration interactions, or a mix of protected rescues/regressions. The new-stage post-vote write is observational, not independently a scorer boost.
- We cannot attribute `+0.003` to Fable protection causally without **same-row, same-source, paired final decisions and labels**. Multiple changes or unknown implicit runtime behavior remain possible; publicly scored submissions alone are not a controlled paired ablation.

## Best next study (no small-model training; do NOT rerun 9h hidden to guess)
Freeze original 0.718 scoring path, collect genuine aligned **pre-Apex logits, post-V1 logits, post-Apex decisions and eligible truth** on a qualified independent full18 label cohort. Replay old and new *Apex decision rule only* from identical intermediate states **without additional large-model inference**, and count:
- number meeting V1 `applied`, GLM→Fable, pre/post margin checks;
- among exactly changed decisions: correct-to-wrong vs wrong-to-correct, per-class balanced recall and paired prompt-bootstrap CI;
- evidence of protected GLM decisions later contradicted by weaker style readouts;
- effect on runtime and fraction of users actually covered by V1.
Unchanged rows must remain bytewise identical. Test against a no-change fallback and simple residual-alone control. If no independent truth / genuine full-stage intermediates are available, call this **BLOCKED_BY_EVIDENCE** and do not invent a causal success rate, new score or another leaderboard test. C1/C2 12-class external panels omit GLM/Fable and cannot validate this specific mechanism; prior Oleg216 may be source-contaminated.

## Decision
- `DIAGNOSE` / `NEW_HYPOTHESIS`, not `PASS` and not `PROMOTE`: there is now a mechanistic candidate involving **evidence conservation of valid cross-family expert flips**.
- No new training (including RoBERTa), no parameter tuning on spent public score, no Kaggle submission, no claim of own original GOLD.
- Preserve the 0.718 champion (external code attribution intact). Prior GLM rescue-per-token 0.714 was a different priority-heuristic experiment and does **not** disprove this protection hypothesis.
