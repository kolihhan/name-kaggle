# NTL paired 18-class authentic-source transfer gate — preregistration
Date: 2026-11-24. Branch: research/ntl-prompt-conditional-style-transfer-20261124.
Code SHA at first implementation: 1ef4e63f1e2e6be337f8be47f6a3702c58da0371.

## Status
IMPLEMENTED, NOT EXECUTED ON REAL SCORED DATA. No Kaggle competition submission or new BA measurement.

## External methodological motivation
SemEval 2024 Task 8 Subtask B first-place AISPACE (https://aclanthology.org/2024.semeval-1.212/) reported source-classifier comparisons and confusion patterns on 6 classes / ~127k rows; NTL has 18 classes / 2699 official TRAIN. LLM Prompt Recovery 5th (https://github.com/mponty/llm-prompt-recovery) motivates independent candidate evidence and ranking, but has different labels and scoring. Neither result implies a transferable gain here. The 2026-11-24 SemEval feature-source experiment already failed its 3/5 stability gate and must not be repeated or converted to BA.

## Falsifiable hypothesis
A source-conditioned attribution candidate genuinely improves *the same authentic prompt-disjoint 18-class rows* over the frozen strong scorer, with positive paired rescues minus regressions, positive 18-class BA delta, and improvement surviving prompt-cluster bootstrap and source strata. Competing explanation: surface/protocol artifacts inflate in-source accuracy but do not transfer across publishers.

## Inputs
PRIVATE CSV with row_id,prompt_id,label,incumbent_pred,candidate_pred,source. Predictions must originate from the SAME row, and exact scorer version/SHA, label mapping, panel rights, source/provider/version provenance and prompt split must be documented separately. Never substitute synthetic toy data, 3-row Kaggle notebook previews, repeated official TRAIN folds or source panels without full 18 labels for independent OOD validation.

## Frozen metrics and gate
18-class balanced accuracy, paired rescues/regressions, per-class recall, source-stratified rescues/regressions, prompt-group resampling 2000 replicates seed 20261124; report 95% interval. Promotion requires delta BA >0, rescues > regressions, lower cluster-bootstrap CI >0 and no material source collapse; panel independence and provider provenance separately checked. A failure is recorded without changing thresholds. Local scores are not Kaggle Public BA.

## Cost and stop
CPU-only evaluation; no GPU, training, model download or paid inference in this phase. Stop if no independent lawful full18 panel and paired strong predictions can be obtained; do not claim a score. A one-shot Kaggle bet, if separately justified under doctrine, requires a NEW preregistered algorithm, executable notebook/runtime preflight and owner-approved submission ledger; this script alone does not qualify.

## Champion preservation
Owner-confirmed 0.715 official Public BA #58680531 is external estopadilla V5, not original NTL research. Do not mutate champion frozen files, running Stage Trace notebooks or existing submissions.
