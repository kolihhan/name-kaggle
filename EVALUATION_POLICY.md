# Future Candidate Evaluation Policy

Status: **FROZEN**

Effective: 2026-11-01

This policy governs new competition candidates after the Phase 19 validation diagnostic and the domain-separability audit. It is a model-selection policy, not a new modeling phase.

## Why this policy exists

The project has consumed the same 2,699 official training rows through many sequential experiments. Phase 19 showed that Phase 15's local advantage over Phase 7 survives three alternate prompt-group assignments, so one permanent fold mapping is not sufficient to explain the Public-LB collapse. At the same time, those alternate assignments reuse the same official rows and are now spent diagnostics.

The practical risk is therefore broader than one bad fold map: a candidate can become increasingly optimized for the distribution represented by the official training set while still failing to transfer to the hidden competition distribution.

The policy must prevent two opposite mistakes:

1. treating the permanent grouped OOF score as a hidden-test oracle; and
2. manufacturing fake validation freshness by repeatedly generating more splits of the same 2,699 rows.

## Frozen reference state

- Competition champion: **Phase 7 gated specialist**.
- Phase 7 OOF balanced accuracy: `0.3700521998508576`.
- Phase 7 Public LB: `0.369`.
- Phase 7 submission ref: `57838845`.
- Phase 15 is not the champion despite OOF `0.4053218990802883`; its Public LB is `0.357`.
- Phase 19 shadows A/B/C are **spent diagnostic folds**.
- The Kaggle Public leaderboard is external evidence only, not a tuning target.

## 1. Freeze the candidate before evaluating it

Every new candidate must have a written contract committed before its competition-facing result is observed.

The contract must state:

- the hypothesis and why it is materially different from a rejected route;
- exact training data and provenance;
- exact model / representation / routing / ensemble rule;
- which parameters are fixed versus trainable;
- official-data fold isolation rules;
- the primary metric and required diagnostic outputs;
- the predeclared transfer / robustness check;
- the stop condition if the candidate fails.

Do not begin with a broad sweep and call the winner the candidate. Hyperparameter, threshold, source-weight, label-subset, routing, or blend searches on consumed validation evidence are not permitted.

A failed candidate may lead to a new candidate only when the hypothesis or mechanism materially changes. A cosmetic retune is not a new experiment.

## 2. Official grouped OOF remains required, but is not sufficient

For historical comparability, continue to report the permanent prompt-grouped 5-fold OOF balanced accuracy.

Requirements:

- no random row split;
- every `prompt_id` remains entirely inside one fold;
- every stage that learns from official text or labels must respect the fold boundary;
- official validation rows must not affect feature fitting, adaptation, routing, calibration, threshold selection, or model selection for that fold;
- preserve all OOF predictions so error overlap and per-class behavior can be inspected.

The permanent OOF score is a **comparison coordinate**, not the sole promotion oracle.

Do not create more alternate prompt-group salts merely because a result is surprising. Phase 19 already tested fold-assignment sensitivity for the critical Phase7/Phase15 case. Additional re-splitting of the same rows does not create a new hidden distribution.

## 3. Every candidate needs one predeclared transfer check

Before running the candidate, choose the smallest transfer / robustness check that matches its failure mode. It must test something other than merely reassigning the same official rows.

Acceptable examples include:

- a source-level or corpus-level holdout when the candidate depends on external data;
- a leave-source-out diagnostic for representation learning;
- an external pre-competition evaluation corpus that was not used to fit the candidate, when labels and provenance make the comparison meaningful;
- for a complementary model, fixed honest-OOF disagreement / conditional-correctness analysis against Phase 7 plus a predeclared ensemble rule;
- a predeclared ablation that removes the suspected shortcut or external component and tests whether the claimed mechanism survives.

The check must be selected before seeing the candidate result. It is evidence about transfer risk, not a substitute hidden leaderboard.

The following do **not** count as independent transfer evidence by themselves:

- another salt over the same 2,699 official rows;
- bootstrap intervals over the same OOF predictions;
- post-hoc per-class slices chosen because they look favorable;
- a domain classifier or threshold designed after seeing the candidate result;
- Kaggle Public LB score.

## 4. Rules for external-data and two-stage candidates

External data is not automatically good or bad. Phase 7 transferred; Phase 15 did not.

For any candidate using external data:

- freeze exact source membership before evaluation;
- record provenance, competition timing, license / terms status, and answer-overlap checks;
- do not admit or remove sources based on the candidate's OOF result;
- report per-class recall so gains caused by covered labels and losses on uncovered labels remain visible;
- predeclare at least one source/domain robustness check appropriate to the candidate.

For two-stage representation learning, a preferred honest structure is:

1. learn a representation from admissible public external data without using official validation rows;
2. inside each official fold, fit any official-data adaptation / classifier only on that fold's training partition;
3. evaluate on untouched official validation prompt groups.

If a stage updates parameters from official text, it is fold-local unless the contract explicitly proves why seeing validation text cannot leak competition-facing information.

## 5. Rules for complementary models and ensembles

A candidate does not need to beat Phase 7 as a standalone model if its purpose is complementary error correction.

Before examining candidate OOF results, freeze the combination rule. Examples: fixed equal-score averaging, a fixed structural gate, or another deterministic rule justified by the candidate mechanism.

Required reporting:

- candidate standalone balanced accuracy;
- Phase 7 balanced accuracy on the same OOF rows;
- ensemble / routed balanced accuracy under the frozen rule;
- disagreement rate;
- `Phase7 only correct` and `candidate only correct` counts;
- per-class recall deltas;
- whether the improvement is concentrated in a narrow external-coverage subset while other labels degrade.

Do not search blend weights, thresholds, gates, class subsets, or routing rules after seeing OOF predictions.

## 6. Promotion gate

A candidate may become a **Kaggle submission candidate** only when all of the following hold:

1. its contract was frozen before the evaluated result;
2. grouped OOF is leakage-safe and reproducible;
3. the candidate has a predeclared transfer / robustness check beyond merely making another split of the official rows;
4. its claimed mechanism is supported by the required diagnostics rather than only one aggregate OOF number;
5. any ensemble or gate was fixed before candidate OOF inspection;
6. data provenance / competition-rule questions are resolved sufficiently for submission;
7. submission preflight passes without using Public LB feedback to alter the model.

A higher permanent OOF score alone is **not** enough for promotion.

## 7. Kaggle Public LB consumption rule

For a promoted candidate, one clean Kaggle submission may be used as an external transfer check after local evaluation is frozen and archived.

After the Public score is observed:

- record it as evidence;
- do not tune thresholds, weights, sources, model parameters, routing, or label subsets toward that score;
- a poor Public result rejects that candidate unless an independently demonstrated implementation / submission bug explains the failure;
- any follow-up model change must start as a materially new frozen candidate contract.

Infrastructure-only resubmission is allowed only when the defect is demonstrated independently of leaderboard score and the intended model predictions were not actually submitted.

## 8. What is permanently spent

Do not present these as unseen evidence for future candidates:

- the permanent Phase 1 grouped fold mapping;
- Phase 19 shadow A;
- Phase 19 shadow B;
- Phase 19 shadow C;
- the observed Phase 7 and Phase 15 Public scores;
- post-result diagnostics already used to explain Phase 15.

They remain valid historical evidence and regression references.

## 9. Current stop list

Until a materially new candidate is frozen under this policy, do not:

- retune or rescue Phase 15;
- reopen Phase 10–12 post-hoc gates;
- reopen Phase 17 balancing;
- use the contaminated Phase 16 final public neural checkpoints as honest OOF models;
- run blind hyperparameter sweeps on the permanent folds;
- mint new prompt-group splits just to obtain a more favorable validation view;
- use Public LB as a model-selection loop;
- submit another model to Kaggle.

## Next authorized step

Brainstorm **one** materially different modeling direction under this policy. Do not open its implementation phase until its candidate contract states the mechanism, fold isolation, transfer check, and fixed promotion evidence.