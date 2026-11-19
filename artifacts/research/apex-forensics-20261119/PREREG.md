# NTL — Apex Residual forensic diagnostic, 2026-11-19

**Scope:** One preregistered, CPU-only, read-only-to-Kaggle experiment with a **previously spent diagnostic cohort**, not an independent score comparison with the official 0.702 system. No new external searching, test scores, Kaggle GPU, private-score attempts, tuning or inference submissions.

## Real question
Before considering an Apex override veto, answer the first falsifiable question: *Does the existing fitted `pngy_residual.Residual` stage carry label-discriminative information that complements a previously frozen strong model's correct answers on precisely the same 18-way Oleg panel?* A strong model at `0.702` and its pre-Apex per-row logits are **not available on this panel**. The existing historical 0.691 frozen strong model does have 216 exact IDs/predictions. Do not fabricate 0.702 OOF, logits or stage rescues.

**Mechanism H1:** The residual's independent style evidence rescues some hard sibling errors, and a family-level veto may avoid many cross-family regressions.

**Mechanism H0:** The residual has weak out-of-source generalization; its disagreement with the old strong model mostly replaces correct judgments, with no material net rescue. In this case abandon blind override rules, investigate whether the final Apex contributes distinct evidence at all.

## Prespecified inputs
- Pin official `data/train.csv` SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`, 2,699 rows, 150 prompt groups, exactly 18 labels. Fit residual **only** to official TRAIN, not Oleg and not any official held-out validation.
- Kaggle public 0.702 scored source `kolihhan/ntl-0702-apex-pinned-nov18`, exact code SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`. Extract **only** the exact `%%writefile pngy_residual.py` source block, no model inference. The stage uses `Residual(labels).fit(train.text,train.model)` and `features(query.text)` in `apex_postprocess.py`.
- Historical frozen Oleg216 original source `olegzholobov/name-that-llm-response-corpus` `external_short_wide.parquet` SHA256 `15a67d022315b2508af8edd749a7c041d42cc37e266106028c73e405659c400f`, `external_cell_metadata.parquet` SHA256 `397af75e5f2809107d0758a147646a68a0132479a6ea77432d86243e92ad9079`, selector `scripts/ntl_oleg18_frozen_selector_20261116.py` (fixed 12/class =216, 216 unique prompt groups). Original Kaggle `kolihhan/ntl-oleg18-frozen0691-probe-nov16` output `submission.parquet`: exactly 216 IDs, 89 correct `=0.412037`. No new selection or family label tweaking.

## Fixed three *diagnostic proxy* policies
- `B`: untouched older frozen strong model **0.691** predictions on Oleg216 (not the current 0.702 pipeline).
- `R`: `pngy_residual.Residual`'s own predicted class from assembled class-wise `predict_proba` evidence. This is the **actual fitted residual evidence stage**, but its direct argmax **IS NOT the Apex final output**, because the real Apex blends this evidence with pre-Apex logits and postprocessing.
- `P0`: exact old strong model class for rows initially predicted `glm-5.3` or `glm-5.3-flash`; `R` otherwise. A **label-only hypothetical policy**, not an exact Apex inference ablation.
- `P1`: identical `P0`, but keep old class when `R` recommends a cross-family replacement. **Diagnostic veto only**. Actual proposed cross-family veto operates on **pre-Apex family probabilities** absent here, so P1 is **not its scored implementation**.

## Outputs
Per policy: 18-class BA and correct count, cross-family errors, within-family errors, per-class recall. Relative to B: same-row rescues/regressions, original-correct changed, original-wrong changed, crossfamily/sibling disagreement counts. P1 vs P0: same-row paired delta, rescue/regression, ten-thousand replicate 18-label stratified bootstrap (balanced 12 examples/class), exploratory only. Report aggregate JSON and **no row labels, IDs, generated passage text, or logits** outside ephemeral CI memory.

**Promotion rule:** NONE. This is **descriptive** on the repeatedly inspected Oleg216, with known narrow sensitive-topic source shift and unverified upstream lineage. `n=216`, 12/class, large paired uncertainty: under ~30% discordance, paired BA SE ~0.037 and 80%-power MDE approximately 0.10+. Even a nominal +0.018 from four changed rows would be noise. Do not select a GPU rerun/submission based on this alone.

**Stop condition:** If residual-only suggestions mostly break old-correct predictions or P0 falls below B, reject blind/residual-alone replacements. If P1 outperforms P0, retain the falsifiable *family-supported override* hypothesis, not an accuracy claim, pending an actual source-faithful Apex/logits observation. This single check should replace repeated external-source inventories and weak retraining.
