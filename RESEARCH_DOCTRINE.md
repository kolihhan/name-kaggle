# Research Doctrine — Avoid Methodological Lock-In

This file is a durable research rule for Name That LLM. It supplements `AGENTS.md`, `README.md`, `EVALUATION_POLICY.md`, experiment contracts, and result artifacts. Frozen contracts still control any already-running or already-evaluated candidate.

## Core lesson

Do **not** copy a previous competition winner's method mechanically. This doctrine does not say "use RL", "use a larger model", or "ensemble more".

The durable lesson is:

> Do not let an initially sensible research restriction become permanent after the evidence supporting it has changed.

A disciplined experiment loop can still converge to the wrong local optimum if the search space itself is artificially restricted.

## 1. Separate engineering correctness from competitive strength

Engineering correctness includes reproducibility, leakage control, valid folds, deterministic/reproducible execution where required, correct packaging, passing tests, and legal runtime constraints.

Competitive strength requires evidence that the representation/model actually transfers: disjoint validation, meaningful slices, external/source transfer, strong alternatives, and eventually hidden-distribution evidence.

Never use engineering correctness as evidence of model quality.

## 2. Every major restriction needs a pivot trigger

Whenever a research plan says things such as:

- do not fine-tune;
- do not use larger models;
- do not ensemble;
- keep the current feature representation;
- keep the current CV recipe;
- do not use pseudo-labels or augmentation;
- do not revisit the problem formulation;

record:

```text
RESTRICTION:
WHY IT EXISTS:
CURRENT EVIDENCE SUPPORTING IT:
PIVOT TRIGGER:
STATUS: ACTIVE / REOPENED / RETIRED
```

A restriction without a pivot condition is not an acceptable long-term research rule.

## 3. Detect representation/model-class ceilings

Treat these as signals that the current method class may be at a ceiling:

- many feature/model/ensemble tweaks with diminishing returns;
- gains concentrated on already-reused folds or known error slices;
- OOF gains without source-transfer or Public-LB transfer;
- increasingly correlated model-family errors;
- repeated fixes to the same label/source artifacts;
- growing rule complexity around one incumbent representation;
- a different simple representation systematically wins;
- offline metrics stop predicting competition performance.

When several appear, stop asking only "what phase comes next?" and ask whether the representation, model family, objective, data strategy, calibration, or problem formulation itself is limiting transfer.

Previously rejected classes may be reopened when current evidence warrants it: trainable encoders, different backbones, different feature representations, distillation, external-data specialists, pseudo-labeling, augmentation, calibration, retrieval/metadata features when legitimate, ensembling, self-training, synthetic data, alternative losses, or a reformulated classifier/ranking objective.

Reopening is not automatic promotion. It means the old ban must be re-justified against current evidence.

## 4. Maintain local and global hypotheses simultaneously

Every active research lane must track:

**Local:** what is the next smallest discriminating experiment?

**Global:** what is the current dominant bottleneck hypothesis?

Possible global bottlenecks include representation, insufficient data, source/domain shift, model capacity, optimization, class confusion structure, calibration, correlated errors, metadata/source artifacts, evaluation mismatch, or an architectural ceiling.

After a few meaningful experiments, revisit the global diagnosis. Do not create Phase N+1 merely because Phase N stopped improving.

## 5. Optimize for hidden transfer, not a convenient proxy

Permanent OOF folds are historical comparability tools, not an unlimited fresh oracle after repeated adaptive use.

Do not let any single CV mapping, source holdout, public leaderboard observation, class slice, or incumbent model family silently become the objective.

Ask:

> If the system truly became better at model attribution, what other independent evidence should also improve?

Prefer improvements that survive multiple independent views.

## 6. Diagnose why combinations work

For every promoted feature, model, specialist, or ensemble answer:

1. What changed?
2. Which labels/sources improved?
3. Which labels/sources regressed?
4. What mechanism plausibly caused the gain?
5. What alternative explanation could produce it?
6. Does it survive disjoint transfer evidence?
7. Are the components adding independent information or only correlated votes?
8. What new prediction follows from the explanation?

An ensemble that raises OOF without adding independent information is not automatically progress.

## 7. Build a learning loop, not only a patch/phase loop

Every research direction records four items.

- **Hypothesis** states the mechanism and what observable result it predicts.
- **Evidence** states why it is worth testing and what remains uncertain.
- **Experiment** fixes the cheapest discriminating test, its control and its stopping rule before execution. When local GPU is unavailable, prefer available artifacts and bounded CPU checks. A candidate that passes the frozen checks can use one exact Kaggle Notebook version for a hidden-transfer test.
- **Decision** records continue, modify or stop from the observed evidence. A controlled negative result closes a direction. A leaderboard score records transfer of the submitted system and never becomes a parameter search.

Weak loop:

```text
error
-> add feature / specialist / threshold
-> retest familiar folds
-> repeat
```

Preferred loop:

```text
error
-> identify failure class
-> gather representative independent evidence
-> change representation / data / training / objective if necessary
-> evaluate disjointly
-> analyze residual failures
-> update the global hypothesis
```

Handcrafted rules remain appropriate for invariants, leakage prevention, provenance, deterministic gates, and experiment hygiene. They must not become the main mechanism for learning model identity.

## 8. Preserve anti-overfitting discipline

Architecture-level flexibility does **not** authorize looser evaluation.

Continue to enforce fold isolation, no leakage, no Public-LB tuning loop, explicit DEV vs transfer vs final evidence, frozen contracts, paired comparisons where appropriate, exact provenance, FACT / INFERENCE / UNKNOWN, and retention of negative results.

The goal is to be more willing to change the model class, not more willing to overfit.

## 9. Name That LLM-specific watch list

Explicitly monitor:

- CV / Public-LB divergence;
- source/domain transfer failures;
- model families making highly correlated errors;
- hidden-domain shift;
- metadata or source artifacts;
- representation ceilings;
- calibration problems;
- pseudo-label or augmentation opportunities;
- ensembles that improve OOF without adding independent information.

Do not optimize indefinitely around one CV scheme or one ensemble recipe.

For an already-frozen candidate, do not mutate the contract mid-run in the name of this doctrine. Finish or stop it according to its preregistered gate, then reconsider the architecture at the next decision boundary.

## 10. Mandatory checkpoint before a major new direction

Before opening a substantial new research lane, record:

```text
CURRENT BEST:
REAL EVIDENCE OF STRENGTH:
MAIN FAILURE MODE:
CURRENT BOTTLENECK HYPOTHESIS:
IS THIS A LOCAL BUG OR MODEL-CLASS LIMITATION?:
WHAT METHOD CLASSES ARE CURRENTLY EXCLUDED?:
WHY ARE THEY EXCLUDED?:
HAS THE ORIGINAL REASON FOR EXCLUSION CHANGED?:
SMALLEST EXPERIMENT THAT DISCRIMINATES BETWEEN THE COMPETING EXPLANATIONS:
WHAT RESULT WOULD CAUSE AN ARCHITECTURE-LEVEL PIVOT?:
```

Do not reflexively continue the existing lineage. Do not reflexively redesign it either. Follow the strongest independent evidence available.


## 11. Competition-Grandmaster research operating loop (2026-11-15)

**Purpose:** learn the transferable *decision process* demonstrated by strong cross-domain Kaggle competitors, including Yuanzhe Zhou. This is our own operational synthesis, **not** a claim that the competitor published this exact protocol. The goal is **official 18-class hidden balanced accuracy**; our authenticated best remains **0.683** until a later official score is verified.

**Research beats ritual:** an artifact, green CI, status poll, time-budget proof, or compliant submission is engineering feasibility, **not** a better attribution system. Do not spend the main research budget adding gates, auditing the same source, or copying another Notebook merely because it has a higher score in its title.

**For each research cycle, follow this sequence:**
1. **Model the problem:** state the competition metric and data-generating process; list the dominant observed confusions, data-lineage uncertainty, source/prompt shifts, and why the incumbent fails. Label observations FACT / HYPOTHESIS / UNKNOWN. Inspect original solutions and strong alternatives **for a mechanism**, not for a template.
2. **Competing hypotheses:** maintain at least two *materially different* explanations (e.g., generator-specific conditional token likelihood vs source/channel/length artifacts vs correlated ensemble failures). Name the cheapest **observable prediction that differs** between them. Distinguish source effect from generator identity.
3. **Value of information:** prioritize experiments by expected ability to change a submission decision or underlying research belief, not number of scripts/runs. Prefer one cheap discriminating test to several restatements of an already-known negative. Keep no-training as a **current candidate-specific constraint**, not a permanent ban on future method classes.
4. **One controlled change and honest evidence:** freeze candidate/control/labels/source and evaluation protocol first. Analyze same-row rescue/regression, per-class recalls and source slices; a 12/18-label diagnostic cannot certify 18-way BA, and repeatedly seen corpora are not fresh holdouts. Unused evidence must remain untouched until the mechanism is frozen. No adaptive Public LB parameter search.
5. **Promotion decision:** pass feasibility separately from quality. A candidate only earns an official submission budget when it has a testable generalization argument and sufficient disjoint evidence, or a clearly labeled, tightly bounded one-shot research bet. A leaderboard score is an outcome, not a mechanism explanation. Record cost, time, regressions, alternative explanations and confidence.
6. **Stop or pivot:** if 2–3 *independent* informative experiments contradict a mechanism, retire that hypothesis or broaden the representation/model/objective search; do not hide a failure behind another cosmetic phase, blend coefficient, or guard. Conversely, **do not** permanently prohibit a model class because a tiny or spent dataset was negative.

**Minimum evidence table for an executable new idea:**

| Field | Required entry |
| --- | --- |
| Scoring target | Official 18-class BA; comparison to 0.683 |
| Observed bottleneck | Concrete error cohort(s), source, support and lineage |
| Hypotheses A vs B | Competing mechanisms and a different expected observation |
| Intervention | ONE main scientific change; strong frozen control |
| Independent evidence | Source/prompt-group disjointness, coverage and prefit exposure |
| Discriminating test | Same-row prediction/rescue/regression + source/class stratification |
| Cost / stop rule | GPU time, runtime budget and failure criterion declared before run |
| Decision | Promote / revise / retire with what is still unknown |

**Immediate lane (not automatic promotion):** compare *mechanisms* behind sibling-model confusions and source/prompt shift on verified evidence, not a fourth no-label Lightning throughput test. The frozen Lightning0694 scale test is useful for runtime **only**. The verified WeMM fold0 84-row experiment is **3 rescue / 7 regression on spent 12-label sources** and does not justify blending or condemn every alternative representation. First seek or certify a fresh full-18 and source-aware benchmark, or another observable diagnostic that can separate representation ceiling from source artifacts without touching a final blind set.

When updating \`NOW.md\`, write a single short **research decision** before operational status; log an independent negative once instead of duplicating it across runs.

