# Contextual choice CPU v1

## Hypothesis

At equal wording-choice coverage, accounting for sentence position can add enough information about the exact generator to improve the 18-label CPU Phase 7 system. The observed choice and the type of slot are the only new inputs.

## Evidence

The fixed 44-pair feasibility check joined official training text to the 2,699 archived Gate-B predictions. It found 1,489 eligible passages, including 628 plain-head errors. The macro best-case recovery ceiling was 0.2327442. Thirty-four prompts had opposing realizations of a pair in the same position category among different models. This passed the pre-computation feasibility thresholds of 0.05 and 20 prompts. These are upper bounds and coverage observations on spent data. [Earlier research](../contextual-choice-attribution-decision-20261106.md#current-kaggle-decision)

## Experiment

- Reproduce the original Phase 7 anchor, 8-class specialist and fixed routing from `scripts/make_submission.py` at research base `f16f5c08c9ca`. Use the exact official training rows, permanent folds and frozen 2,503-row external snapshot used in the original Phase 7 submission. All official fitting is fold-local. The external snapshot is used only by the unchanged specialist.
- Use the already-locked 44-pair detector from the feasibility check. Normalize case, whitespace and apostrophe variants. Context has two values, sentence opening and other, assigned by a fixed left-boundary rule. Context scoring and count scoring see the same events.
- Fit the common pooled choice frequency for each pair with add-one smoothing. Contextual frequencies shrink toward that pair frequency with four pseudo-events. Per label and pair, fit a single log-odds tilt with ridge penalty 2, using a common pooled offset. The count control removes the context distinction with the same fit and regularization.
- Add the mean observed-event log likelihood to the corresponding raw SVC margin with coefficient 1. The pre-adjustment Phase 7 gate fixes the head and covered-versus-uncovered partition for each row. No choice means zero change. Score the original Phase 7 control, Phase 7 plus count evidence and Phase 7 plus contextual evidence on exactly the same validation rows.
- Robustness check uses a fixed typography normalization on validation text only. Map curly quotes to straight, en/em dashes to hyphens and remove outer whole-passage bold wrapping. Use the same trained classifiers. This tests sensitivity to a plausible formatting change beyond resplitting the same data. Neither this nor the permanent OOF scores is fresh hidden-test evidence.
- Report aggregate 18-class balanced accuracy, paired rescues and regressions, fold deltas and the fixed typography check. All diagnostics are aggregate.

## Decision

Check the reproduced Phase 7 score against 0.3700522 within 0.0005. Advance the contextual variant only with at least +0.010 macro BA over Phase 7, at least +0.005 over the count control, a positive count-control delta on at least three of five folds and nonnegative deltas against both controls under the typography stress. Failure closes this candidate without retuning.

After a pass and a clean private CPU Notebook preflight, use one exact version for an official Kaggle submission. Compare to the historical Phase 7 public 0.369. At least +0.010 supports continuing this mechanism in the CPU system. A smaller change is insufficient evidence for this candidate. A Public score is a system-level transfer result. There is no promotion based on fitting the same score again. The independently developed 0.520 and third-party 0.668 remain separate reference systems. [Exact Phase 7 submission](../../submission/FIRST_SUBMISSION.md)
