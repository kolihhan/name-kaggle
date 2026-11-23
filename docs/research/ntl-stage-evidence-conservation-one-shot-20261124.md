# Frozen one-shot research bet — Stage evidence conservation (2026-11-24)

## Status
IMPLEMENTED / NOT VALIDATED / NOT SUBMITTED. Candidate is `scripts/ntl_stage_evidence_conservation_20261124.py`; original 0.715 scorer unchanged. Historical Public 0.715 (#58680531) is attributed to the **external Estopadilla V5** reproduction, not an original model.

## Doctrine checkpoint
- CURRENT BEST: own official Public BA 0.715 (attributed third-party source), not independently verified Private BA.
- REAL EVIDENCE: Kaggle official receipt; original source SHA `202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8`.
- MAIN FAILURE MODE: GLM family recognition and later stage label conservation are incompletely measured. Historic 0.683 scorer on 64 external GLM prompt pairs had 6 rescues/17 regressions after likelihood override, **not the current 0.715 scorer**.
- BOTTLENECK: cross-family decision preservation rather than missing backbone capacity (HYPOTHESIS only).
- LOCAL VS GLOBAL: this is a **local inference safety experiment**, not proof of model-class ceiling.
- EXCLUDED: new 14B/QLoRA training, weak surrogate classifiers, more GLM threshold tuning, blind ensemble; current evidence does not justify cost.
- PIVOT: If cross-family edits are rare/zero, or correctly rescue more than they regress, RETIRE this rule. If they are frequent and demonstrably net harmful on independent same-row labels, compare with evidence-source improvement, not more ad-hoc thresholds.

## Precommitted candidate
C0: final unmodified 0.715 post-stage model labels.
C1: when **pre-GLM family != post-GLM family**, preserve the pre-GLM label; when family stays the same, keep post-GLM prediction. No fitted weights, class thresholds, randomness or score tuning. Mechanism: preserve provider-family evidence while allowing within-family GLM sibling updates. This is intentionally falsifiable; it may block beneficial cross-family rescues and should NOT be submitted merely because it runs.

Data: exact same-row pre/post outputs from hash-pinned 0.715, not synthetic or a different 0.691 scorer. Ground truth never consulted by C1 decision. First check real frozen Stage Trace `kolihhan/ntl-0715-frozen-stage-trace-research-20261123`. This trace currently exposes only aggregate counts and the private pre-stage snapshot is ephemeral; obtaining the actual aligned pre/post pair may require a separate private instrumented run. **Do not treat trace counts as labelled evidence.**

## Evaluation and stop gates
1. Structural: 18 labels, exact ID order, no overwrite, same Kaggle assets/runtime, no training; smoke test and CI green.
2. Feasibility: real trace shows nonzero cross-family changes; otherwise stop C1 (zero treatment effect).
3. Truth: evaluate C0/C1 on an authentic independent prompt/source-disjoint, full-18, same-row cohort; require positive 18-way BA, per-class recall, rescues > regressions, bootstrap confidence and provenance. If unavailable, one-shot submission is **not automatically justified**. An exception requires a fresh explicit, reproducible positive mechanism result, successful private full inference, exact output, deadline and cost checks; one submission only, no LB tuning.
4. If cross-family edits have more rescues than regressions, retire immediately; never sweep thresholds.
5. Do not submit a candidate that has only synthetic smoke and an unlabelled trace.

This candidate is deliberately distinct from Dennis's Fable-return and GLM-prefix variants: it constrains **family-changing final decisions** rather than changing GLM scoring order, prompt prefix or adding a trained classifier. It may overlap existing family-preserving residual rules; this is not claimed to be novel prior-art-free. It has no demonstrated benefit.

## Cost
CPU-only implementation/test. New GPU use = 0. New official submissions = 0. Reuse queued trace; no duplicate private push.
