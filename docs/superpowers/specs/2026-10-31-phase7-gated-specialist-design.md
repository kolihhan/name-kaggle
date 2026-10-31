# Phase 7 — Gated external specialist design

Date: 2026-10-31

## Intent

Test whether the audited 2,503-row GPT/Gemini corpus can contribute its strong covered-label signal without repeating Phase 6's damage to the 10 uncovered labels.

## Goals

- Keep the canonical Phase 5b official-only anchor unchanged.
- Train a separate specialist only for the 8 labels covered by the external corpus.
- Use one predeclared deterministic gate with no threshold, blend-weight, class-weight, sample-count, or SVC-C sweep.
- Score only official OOF validation rows under the permanent `GroupKFold(prompt_id)` mapping.

## Non-goals / anti-goals

- Do not globally augment the 18-class anchor again.
- Do not tune against the permanent folds.
- Do not alter the covered/uncovered label set after seeing results.
- Do not claim the external corpus helps uncovered classes.

## Approaches considered

1. **Global score blending / thresholding.** Rejected: it introduces tunable weights or thresholds and can reproduce Phase 6's covered-class bias.
2. **Gated 8-class refinement — selected.** The anchor makes the 18-class coarse decision. Only when the anchor predicts one of the 8 externally covered labels does an 8-class specialist replace the exact label. Anchor predictions in the other 10 labels are never changed.
3. **Higher-diversity new model/data source.** Still valuable, but no newly audited source currently covers the weak/uncovered labels, and the Maniceet source remains blocked. Keep as a later independent direction.

## Selected mechanism

For each fold:

1. Reproduce the exact Phase 5b official-only anchor prediction on official validation rows.
2. Build a separate specialist training set from:
   - official training rows whose label is one of the 8 covered labels; plus
   - all 2,503 audited external rows.
3. Fit the same strong-sparse feature construction and `LinearSVC(C=1.5, class_weight=balanced)` as Phase 5b, but as an 8-class classifier.
4. Predict specialist labels for all official validation rows.
5. Final prediction rule:
   - if anchor prediction is uncovered: keep anchor prediction;
   - if anchor prediction is covered: replace with specialist prediction.

This gate never changes an anchor prediction from uncovered → covered or covered → uncovered. Therefore, for every true uncovered class, whether the row is correct or incorrect is unchanged by the ensemble; uncovered-class recall must equal the anchor exactly. Only discrimination within the 8-label covered subset can change.

## Success criteria

Primary:

- OOF balanced accuracy > Phase 5b anchor `0.36709`.

Required invariants:

- uncovered 10-label per-class recall exactly matches Phase 5b (within numerical reporting tolerance);
- same 2,699 official OOF rows, permanent fold fingerprint, no prompt leakage;
- no parameter sweep or adaptive gate.

If balanced accuracy does not improve, reject this specialist mechanism and move to a genuinely different representation/data direction rather than tuning the gate on these folds.
