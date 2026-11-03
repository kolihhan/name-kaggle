# V1 Mechanism Audit Design

Date: 2026-11-03 MYT

## Status

**POST-HOC DIAGNOSTIC ONLY.** This audit consumes already-spent official OOF predictions. It cannot promote, tune, route, blend, change v3, or authorize a Kaggle submission.

## Question

Why does `two-stage-lora24-v1` work, where does it still fail, and which failure modes are plausibly addressed by the already-frozen DCT temporal-fingerprint v3 candidate?

## Inputs

Use immutable completed evidence only:

- official train from Gate-B artifact `13125872633`;
- v1 plain/centered Gate-B OOF from the same artifact;
- Phase14 sparse OOF from artifact `13067907516`.

No model fitting, hyperparameter search, alternate folds, threshold search, Public-LB feedback, or new competition submission.

## Required diagnostics

1. Reproduce balanced accuracy for v1 plain, v1 centered, and Phase14 sparse.
2. Split v1 plain errors into same-family versus cross-family confusions.
3. Measure plain accuracy by text-length quartile.
4. Control the length observation descriptively within each true model: compare accuracy below/above that model's median word count and report how many labels improve on the longer half plus the macro mean delta.
5. Measure centered and sparse rescue/break counts relative to plain, including whether rescued plain errors were same-family or cross-family.
6. Report mutually exclusive agreement patterns among plain/centered/sparse and their conditional accuracies.
7. Report top plain hard-confusion pairs.

## Interpretation boundary

- Agreement patterns may explain why a frozen ensemble can work, but this audit must not derive a new production routing rule from spent OOF.
- Length/family patterns may motivate future mechanisms, but cannot alter the frozen v3 PASS gate.
- If v3 later passes its predeclared transfer gate, these diagnostics remain explanatory only.
- If v3 fails, any v4 must freeze a materially new mechanism and a fresh transfer check before evaluation.

## Reproducibility

Implement a pure CPU analysis script with a JSON report. Add focused unit tests and a public GitHub Actions workflow that downloads the two immutable workflow artifacts by ID, runs the audit, and uploads the JSON/Markdown evidence.