# Candidate Result — prompt-residual style representation v1

Status: **GATE A FAIL — ROUTE CLOSED**

Date: 2026-11-01

The frozen contract is preserved unchanged in `CONTRACT.md`. This result records the first successful scientific execution of Gate A and applies the preregistered stop rule.

## Scope and safety

- Scope: external source-transfer Gate A only.
- Official competition rows scored: `0`.
- Official grouped OOF authorized: **no**.
- Kaggle submission authorized: **no**.
- Competition test data used: **no**.
- Public-LB tuning used: **no**.

## Frozen data and model

- Phase 14 conservative external subset: `43,781` rows.
- Phase 15 terms-clean external subset: `26,025` rows.
- Phase 15 unique external prompt IDs: `2,801`.
- Phase 15 external labels: `11`.
- Raw public-pool SHA256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`.
- Encoder: `answerdotai/ModernBERT-base`, frozen.
- Max length: `256`.
- Pooling: attention-mask mean, then L2 normalization.
- Projection: `LinearDiscriminantAnalysis(solver="svd")` on prompt-centered residuals.
- Classifier: `LinearSVC(class_weight="balanced", C=1.0)`.
- Source holdout leakage exclusion: both `prompt_id` and `prompt_key`.

## Execution provenance

The scientific execution used commit `d9d86098eafb968bf0b4b9f85b32159a7394aa91` (`[gate-a-run] embed exact scripts in single Kaggle code file`) and GitHub Actions run `41297798343`.

The Kaggle kernel completed and produced `prompt_residual_style_gate_a.json`. The parent workflow later failed in its GitHub-side verification step because that verifier imported the Phase 14 module without installing NumPy. This happened after the output had already been downloaded and did not affect the scientific run or its JSON.

The already-completed Kaggle output was then recovered **without rerunning the model** by commit `85a046ba411778ba853ed7de5148bd0694e627c5`, workflow run `41299047062`, which completed successfully.

Recovered evidence artifact:

- artifact ID: `13079928676`;
- artifact name: `prompt-residual-style-gate-a-recovered`;
- artifact ZIP SHA256: `f1c74f9365bf5e574d28f2e2d69b1758d7f5fbd69e9508b836b15fe8fc110546`.

## Gate A results

| Held-out source | Raw frozen BA | Projected BA | Delta |
| --- | ---: | ---: | ---: |
| abhinav | 0.4215699021 | 0.3630891049 | -0.0584807972 |
| hailey | 0.4683566042 | 0.4565561927 | -0.0118004115 |
| kaplinskiy | 0.4285594578 | 0.3663976185 | -0.0621618393 |
| keshav | 0.3896161616 | 0.3556565657 | -0.0339595960 |
| lucasboesen | 0.4270560561 | 0.3885417443 | -0.0385143118 |
| maniceet | 0.2935053792 | 0.3353810543 | +0.0418756751 |
| prvsiyan | 0.3674644146 | 0.3610026847 | -0.0064617299 |
| prvsiyan_v1 | 0.4998034163 | 0.4245267123 | -0.0752767040 |
| semenoff | 0.3580004874 | 0.3587121386 | +0.0007116512 |
| seyit | 0.3684550790 | 0.3267139634 | -0.0417411156 |
| willrice | 0.3617992743 | 0.3053262665 | -0.0564730078 |
| zakir | 0.4216208288 | 0.3570130733 | -0.0646077555 |
| zbigniew | 0.4387421893 | 0.3803485679 | -0.0583936214 |

Aggregate over all `13` eligible seed sources:

- mean raw frozen BA: `0.4034268654288547`;
- mean projected BA: `0.36763582206930817`;
- mean projected-minus-raw delta: `-0.03579104335954662`;
- projected representation worse on `11 / 13` sources.

Frozen decision returned by the executed report:

```text
status = fail
official_oof_authorized = false
reason = mean_delta_or_source_consistency_rule_failed
```

## Interpretation

### FACT

The frozen prompt-residual LDA projection failed the preregistered external transfer gate. It reduced mean held-out-source balanced accuracy by about `0.03579`, and it was worse than raw frozen ModernBERT embeddings on 11 of 13 eligible sources.

Only `maniceet` showed a material positive delta; `semenoff` was approximately flat. These are descriptive post-result observations and are **not** authorization for source-specific rescue or cherry-picking.

### INFERENCE

Under this exact frozen construction, prompt centering followed by supervised LDA appears to remove transferable model-attribution signal more often than it isolates a robust style signal. The raw frozen embedding retains substantially better cross-source transfer.

### UNKNOWN

This result does **not** establish that all neural, style-based, or representation-learning approaches are bad. It rejects this specific prompt-residual + LDA mechanism under this dataset, encoder, and preregistered transfer test. A fundamentally different candidate would need a new contract frozen before seeing its results.

## Stop rule applied

Per `CONTRACT.md`:

- do not run Gate B official grouped OOF for this candidate;
- do not tune LDA solver, dimensionality, source subset, source weights, `C`, encoder, or eligibility thresholds;
- do not rescue favorable sources post hoc;
- do not search a blend/routing threshold;
- do not make a Kaggle submission from this candidate.

`prompt-residual-style-v1` is closed.