# Contextual choice CPU v1 result

**2026-11-06 MYT. Closed at the frozen CPU gate.**

## Hypothesis and evidence

Sentence position might add exact-model evidence beyond the same 44 wording-choice counts. The fixed detector found choices in 1,489 of 2,699 passages, including 628 plain LoRA24 errors. The optimistic balanced-accuracy gain ceiling was 0.232744 on those LoRA24 predictions, with opposite choices in the same position category on 34 prompts. Those coverage numbers justified testing the idea with the Phase 7 CPU system. [Feasibility design](../contextual-choice-attribution-decision-20261106.md#current-kaggle-decision)

## Experiment

The saved local run compared Phase 7, Phase 7 plus choice counts, and Phase 7 plus sentence position on the same five permanent prompt-grouped folds. The typography check applied the locked normalization to validation inputs with the same fitted models. The contract was committed before the run completed. [Frozen contract](CONTRACT.md)

| System | Grouped balanced accuracy | Normalized typography | Correct on raw text |
|---|---:|---:|---:|
| Phase 7 | 0.370052 | 0.309269 | 999 |
| Phase 7 + counts | 0.370428 | 0.307788 | 1,000 |
| Phase 7 + counts and position | 0.370798 | 0.307788 | 1,001 |

The position arm rescued 9 Phase 7 errors and broke 7 correct Phase 7 predictions. The counts arm rescued 9 and broke 8. Compared with counts, the position arm changed aggregate recall only for GLM-5.3, from 0.166667 to 0.173333.

| Frozen check | Observed | Outcome |
|---|---:|---|
| Reproduce Phase 7 within 0.0005 | Exact match to archived 0.3700521999 | Pass |
| Position over Phase 7 ≥ 0.010 | +0.000746 | Fail |
| Position over counts ≥ 0.005 | +0.000370 | Fail |
| Positive delta over counts on ≥ 3 folds | 1 of 5 | Fail |
| Nonnegative deltas versus both controls under typography normalization | −0.001481 vs Phase 7, 0 vs counts | Fail |

## What the negative result covers

The model uses a common position-specific baseline and a model-specific tilt shared across positions. For any pair of models, switching from one form to the other changes their relative log evidence by the difference in their tilts, independent of position. This tests a correction for pooled position frequencies. Model-specific changes in wording preference across positions remain unmeasured.

A read-only check using the same detector found 144 sentence-opening events out of 2,733 matches. Eight pairs appeared at the start of a sentence, and zero of the 44 had observations for both forms at both positions, even after pooling the labels. The catalogue has broad passage coverage and sparse evidence for how a wording preference changes with position.

## Decision

Close this candidate without retuning. The preflight and Kaggle submission were conditional on passing the CPU gate. The negative result applies to this Phase 7 variant and the spent validation rows. Reopen the broader wording-choice question only with a different source of context and enough comparable occurrences of both forms to test it. [Evaluation policy](../../../EVALUATION_POLICY.md)

## Verification

- Source of the saved CPU evidence was the local `choice_cpu_result.json` (SHA-256 `5887efc111bb7af59e1abf1ea2d39a946ce9b2b3118bb24387f7b9055da659d3`). The historical run recorded 469.73 seconds.
- Rechecked the Gate B archive SHA-256 `fa675fcf8ac3860984b449748cf067c3da29389a921417e5e1047bb22a7be8ce`, recomputed the read-only coverage counts, and verified byte matches for the training set, folds, original Phase 7 code, and external snapshot against the research branch. [Gate B evidence lineage](../../candidates/two-stage-lora24-v1/GATE_B_RESULT.md)
- Reconstructed the raw balanced accuracies from the saved per-label recalls, reconciled rescue and regression totals, and re-applied all five frozen checks. Two synthetic unit checks passed for context handling and fixed Phase 7 routing.
