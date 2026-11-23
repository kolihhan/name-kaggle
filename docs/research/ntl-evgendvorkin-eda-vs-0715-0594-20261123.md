# NTL architecture review — Evgen Dvorkin forensic EDA vs scored 0.715 / failed 0.594 joint decoder

Date: **2026-11-23**, READ-ONLY source review; **NO** model changes, GPU, Kaggle scoring, training, or re-submission.

## Exact source provenance

- Target: public Kaggle Notebook [evgendvorkin/name-that-llm-deep-forensic-eda-with-analogie](https://www.kaggle.com/code/evgendvorkin/name-that-llm-deep-forensic-eda-with-analogie).
- Authenticated public Kaggle code pull: [GitHub Actions #45102180674](https://github.com/kolihhan/name-kaggle/actions/runs/45102180674), **58 cells**, Notebook SHA256 `74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56`; code pulled only for this read and deleted from runner.
- Notebook **Part I** (cells 0–12) = author's own six EDA code cells and explanation; **Part II** (cells 13–57) = attributed reproduction of **dennis6886's** 29-cell strong inference notebook, itself built on Estopadilla v5, Alex Chilton, LumiBee et al. **0.719** is the source owner's *reported* score, **not verified score of Evgen's integrated V36**. The notebook explicitly labels its own Kaggle rerun **pending** (cell 56). Do not pass off score as evidence of a new EDA classifier.
- Independently verified own baseline: Kaggle **#58680531 0.715**. Same user's **#58729013 0.594** from appended global Hungarian, delta **−0.121**. Authenticated receipts [#45100579731](https://github.com/kolihhan/name-kaggle/actions/runs/45100579731).
- Original scored-source reconstruction: [read-only Kaggle source inspection #45102428512](https://github.com/kolihhan/name-kaggle/actions/runs/45102428512), **32 original cells SHA256 `202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8`**.

## What the external notebook ACTUALLY tests

| EDA Part I | What code does | Scientific value | Limitation |
|---|---|---|---|
| 150 prompt × 18 class coverage | Cross-tab prompt/model, missing-cell checks | Shows official **train** has nearly exhaustive one-response-per-model matched prompts | Test `prompt_id` is unobserved; cannot claim the hidden-test grid is equally exhaustive |
| Length | Within-prompt character residual and per-model means | Separates task effects from author style | Normal/SE intervals across prompt-correlated examples not a direct hidden performance estimate |
| Typography + punctuation | Per-character or per-word rates; subtract prompt means; prompt-centered eta-squared | Generates interpretable style candidates instead of raw topic-sensitive features | Large eta-squared **not** incremental benefit over original 0.715; training-only `prompt_id` cannot be used at hidden inference |
| 18 matched answers | Show representative family examples for median-IQR prompt | Controlled qualitative analogies | No independent performance evidence |
| Validation protocol | 3-fold random StratifiedKFold vs **prompt-grouped** StratifiedGroupKFold; char 3–4gram SVC | Shows why random-split and prompt-heldout estimate different generalization conditions | Their baseline is a weak proxy; ours has separately frozen 5-fold grouped and strong GPU model, no same-row 0.715 OOF |

## Part II code architecture: what is inherited vs incremental

### Present already in our 0.715 scored base

- `FINAL_LOGITS='/kaggle/temp/final-logits.npz'` and `infer.py --save_final_logits FINAL_LOGITS` save a full 18-way **pre-GLM** score matrix with aligned IDs/labels; [verified original source](https://github.com/kolihhan/name-kaggle/actions/runs/45102428512). **Earlier claim that no strong logits existed was incorrect.** It is **not** automatically the final post-GLM 0.715 score matrix; any integration must reflect subsequent GLM decisions.
- Family-mass-preserving sibling residual in `pngy_residual.py`: conditional posterior within Claude/GPT/Gemini/GLM families; default `strength=0.10`, gated by within-family margin. The mass-preserving mechanism exists in the 0.715 source already and is not novel in this notebook.
- Style-model `ntl660_readout.py`: word, char, character shape and function-word representations; inferred lexical `style_groups` and conditional sparse group centering for sizes 2–18. **Prompt grouping is used as a FEATURE**, not an unconditional source-label reassignment.
- Group-covariance `gauss.py`: prompt residual two-way model; exact joint marginals for small groups, approximate EM with soft repulsion for larger groups, rather than enforcing hard label uniqueness globally.
- GLM-5.3 and GLM-5.3-Flash segment-based user-prefix likelihood stage is allocated a separate runtime budget; results receive explicit protection against later sibling-style decisions.

### Additional described in Dennis source, not our prior baseline

- Guarded `fable_return()`: protects a GLM→Claude Fable routing only when the post-likelihood margin improves by at least **0.30** and resulting Fable-vs-GLM margin is ≥**0.25**; traced in target Notebook's `prefix_decisions.py`, and invoked on report `status=applied`. This is a narrow prior-art research candidate, not newly invented here.
- Other typography 'invariants' are zero/near-zero occurrences in **train**, not universal laws; treat as calibrated **soft evidence**, not categorical exclusion on hidden data. Some conditions in donor code are exact hard filters and have potential distribution-shift risk.

## Core architecture defect of our failed add-on

The appended `kaggle/ntl_0715_label_only_joint_final_20261122.py`:

1. After all original GPU, GLM and sibling routing stages finish, it trains a new **weak 18-class char-TFIDF LinearSVC** (historical official grouped OOF BA ≈**0.316**).
2. Standardizes its SVC margins per passage, adds `+1.5` to the strong **final predicted label**, then Hungarian-assigns a **distinct** model to every row in each inferred 2–18 group.
3. This assignment can **swap ALL predictions even if the original strong labels are already distinct and valid**. It does not check the original strong evidence margin, preserve GLM/family decisions, require duplicate original labels, or measure correction risk per family.
4. Hence weak scores can overwrite protected, genuinely stronger decisions. The fixed bonus 1.5 is a numerical parameter on WEAK row-standardized margins, not a calibrated likelihood-ratio tie to the 0.715 model.
5. Strong per-row score evidence **WAS available** before GLM and could have been used, albeit with post-GLM evidence reconciliation. Instead the final stage discarded that signal.
6. Results from Phase5b surrogate (0.367→0.423) and much stronger original GPU champion 0.715 are not same-row comparable. A positive GroupKFold proxy experiment does **not** validate post-processing on a stronger, differently calibrated model.

### Further mechanism evidence (not exact hidden-label diagnosis)

- Original historical Phase5b 5-fold proxy: self-score Hungarian BA **0.4494**, raw strong **0.3671**, weak-score strong-label Hungarian **0.4227**, [#45032497796](https://github.com/kolihhan/name-kaggle/actions/runs/45032497796). Thus the weaker substitute scores already lose ≈0.0267 against the stronger model's own evidence, even in surrogate data.
- Exact unchanged lexical clustering on **official train** at ~540→2699 inference rows: pair precision roughly **1.000 → 0.9774**, with only **2 / 2699** source-label errors mathematically forced by mixed-prompt duplicate true sources at full size; [#45101746731](https://github.com/kolihhan/name-kaggle/actions/runs/45101746731). This strongly weakens *training-distribution cluster contamination alone* as an explanation of −0.121. Hidden-test prompt structure and cluster purity are not observed.
- Fixed **synthetic**, deliberately imperfect ~0.714-BA strong anchors on **real OOF weak scores**: with weak-correlated anchor errors, shipped Hungarian changes ≈731/2699 rows and yields only **+0.002** BA, versus ≈568 changed and +0.022 under minimum-change collision-only. With random anchor errors, shipped +0.070 and collision-only +0.072. [#45101921859](https://github.com/kolihhan/name-kaggle/actions/runs/45101921859). These are counterfactual sensitivity tests, **NOT** real champion accuracy and **cannot prove why 0.594 occurred**. They show correction value depends sharply on correlation between strong and weak errors, and that forced non-collision relabeling is risky.
- Alternative unresolved causes: hidden prompt group structure differs from training; strong rerun environment, model timeout or GLM stage behavior differs; code rerun may produce variable scores; final-stage changed-row and provider transition counts under hidden scoring are unavailable. **Do not attribute all −0.121 to any single mechanism without stronger evidence.**

## Proposed corrected inference architecture — conditional evidence, not post-hoc 18-way force

```text
Original scored 0.715 GPU pipeline, unchanged
  ├─ Save pre-GLM 18-way logits with exact IDs and class order
  ├─ GLM user-prefix likelihood + routing / protected decisions
  └─ Existing family-conditional sibling/post-processing
                  |
       Explicit final decision / evidence ledger
                  |
  Candidate source of extra signal, NOT a free-form label vote:
    a) prompt-centered length/punctuation/shape, trained only on real train
    b) per-family sibling conditional specialist, NOT 18-way SVC override
    c) inferred Prompt links as uncertain soft context, no assumed test one-each rule
                  |
       Correction gate with explicit NO-CHANGE choice
   preserve strong family mass and protected GLM decisions;
   only adjust within a FAMILY and only if measured posterior gain
   exceeds trained-side, held-out validated correction risk
                  |
   paired, exact strong-model baseline vs treatment OOF/control
        rescues / regressions / class recall / altered rows / cost
                  |
   ONLY then single new official submission
```

Practical soft fusion, **only within one selected family** `F`:
`P_new(m|x) = P_base(F|x) [(1-α_x)P_base(m|x,F) + α_x P_specialist(m|x,F)]`.
Set `α_x=0` for confident/protected cases and absent independent evidence. Learn or freeze `α` solely from matched train-side **heldout strong model error data**, never from the new Public score. For cross-provider correction require a separately validated strong likelihood/routing mechanism, not the weak char SVC.

## Next experiment, honest gate

1. Reuse **existing original 0.715 score pathways**, confirm which final per-family scores and post-GLM masks are available. Do NOT assume the pre-GLM logit vector alone captures final champion certainty.
2. Collect a **paired sample of original 0.715 strong model outputs** on unseen labelled prompts (no train contamination, no reuse of a model's training prompts). If 0.715 original score path cannot be reproduced or independently labelled paired data cannot be sourced, label the idea **unvalidated / not deployable**.
3. Before proposing any new correction, report by provider/sibling pair: changed fraction, rescues, regressions, precision of proposed flips, class Balanced Recall, prompt group pair precision and elapsed time. Include **strict no-change** and shuffled/no-group controls. Do not tune against public LB 0.594/0.715.
4. Retain the old 0.715 Champion; code comparison itself entails **no scoring submission**. Reading the Dennis 0.719 source as prior art is useful for architecture, not license to claim 0.719 was scored on this account or an original contribution.

**Scientific conclusion:** The main design error was not 'joint inference cannot work.' It was deploying an uncalibrated 18-class weak-model label rewriter *after* a stronger multi-expert pipeline, with no preservation of expert decisions and no validated real-champion paired rescues. Evgen's EDA is strongest as a hypothesis generator for style evidence; much of the mature sibling/GLM stack was already in our scored baseline. Fix the evidence hierarchy before adding more model complexity.
