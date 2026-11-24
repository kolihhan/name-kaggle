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

## 11. Competition failure-triage: a low score is evidence, not a stop command (2026-11-24)

**Owner direction:** Never discard a research idea *only* because an initial BA, CV score, public score, or pilot is low. Equally, do not claim a promising mechanism merely because one slice rose. The scientific question is **why** the score moved and whether the relevant signal is potentially independent of the strong incumbent. Use the principles of Kaggle winning-solution analysis, hypothesis-driven debugging, and scientific-critical-thinking (source skills studied 2026-11-24); treat external skill text as guidance, never as an automatic submission or install directive.

### Required diagnosis before retiring a mechanism

1. **Verify the measurement:** population/labels, same-row controls, metric (BA18 vs BA12 vs binary), dataset hashes, train/eval prompt isolation, scored-vs-preview status, source version and actual completed full hidden job. Do not compare different populations numerically as if paired.
2. **Separate a weak model from a weak idea:** diagnose model capacity and signal coverage; source metadata and channel/length confounds; train-data scarcity, optimization, preprocessing or implementation failure; calibration, prompt/source shift, and training/inference mismatch; runtime and scale. Do not call one trained RoBERTa head a complete test of ATLAS's high-capacity causal likelihood hypothesis.
3. **Study error partitions, not just BA:** per-class recall (including near-zero), sibling confusion matrix, full-18 versus external-12 support, length/domain/prompt/source cohorts, rescue/regression overlap with strong anchor, confidence and execution cost. If row-aligned predictions are not available, explicitly say this root-cause investigation remains blocked—do not manufacture causal reasons from aggregate BA.
4. **Write competing explanations:** at least two distinct root causes, discriminating predictions and a minimal fixed-control test to tell them apart. Prefer tests of representation, objective and data provenance over another margin, threshold or reweighting of the same exhausted evidence.
5. **Respect predeclared gates:** a failed *frozen candidate recipe* can be closed for leaderboard promotion without declaring the *underlying research family* disproven. Do not tune on the exact held-out rows whose results were already inspected; open a separately specified mechanism experiment with a properly independent falsifier if justified.
6. **Decision labels:** `DIAGNOSE` (failure mode not resolved), `REPAIR_BUG` (objective implementation issue), `NEW_HYPOTHESIS` (material architecture or signal shift), `PARK` (resource/data limit), `PROMOTE` (new independent evidence), `RETIRE_SPECIFIC_RECIPE` (supported negative). `REJECT_IDEA_BECAUSE_SCORE_LOW` is **not** an admissible decision.

### NTL-specific immediate boundaries

- Account's best verified Public BA `0.718` is an externally reproduced Notebook, not an own-method gain. The 0.714 Rescue-per-Token official result falsifies its competitive claim in that execution, **not** all compute-aware large-model inference.
- SCaL source-conditioned masked-word model and trainable last-two RoBERTa 18-class prompt contrastive lost to same-row strong Sparse. Do not infer that high-capacity causal generation evidence or all contrastive source identity learning is useless.
- CPR's C1-only small positive alongside C2/Fold0 negatives requires source-shift/shortcut diagnosis before reusing its general claim, but does **not** authorize retuning CPR-v0 on spent C1/C2.
- No further small-model training for NTL per explicit owner preference; this is a **workflow/resource restriction**, not a proof of impossibility. Existing large pretrained native generator signals and their true error overlap are the priority. Only the owner can reopen the small-training restriction.

### Minimum output for a low-score candidate

`Observed BA/CI + source + class support → root-cause hypotheses → missing evidence → cheapest discriminating test → competitive/runtime cost → specific decision`. Keep separate what is FACT, plausible explanation, and UNKNOWN.
