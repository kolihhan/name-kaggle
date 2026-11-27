# v36 evidence conservation and trace contract — 2026-11-28

## Actual evidence, not a proposed new model
- Scored reproduction: 0.718, submission 58755304. Recovered v36 source SHA 5988d64922650bf94490810effb7278907b301c07f581efbc6d4865d84e63d52; public author attribution remains intact.
- 0.715 vs 0.718 audit: 15 shared emitted modules identical, 9/26/10 model/dataset/kernel mount counts identical. Actual installed V1 variant adds four lines including ID-keyed post-GLM score capture; final Fable guard is active. See docs/research/ntl-0715-vs-0718-real-lineage-causal-boundary-20261124.md. Public score difference is not a paired causal estimate.
- Private spent full-18 panel: pre-Apex 150/252; final 151/252; 22 changed, 8 rescues, 7 regressions, 7 wrong-to-wrong changes. The isolated Fable guard changes 3, rescues 2, regresses 0; do not add these net effects. See docs/research/ntl-fable252-v2-actual-252-results-20261125.md.
- Existing native and post-GLM arrays contained explicit IDs/labels at runtime (validated by snapshot_pre_apex and evaluate in scripts/ntl_fable_preservation252_runtime_20261124.py). They were in /kaggle/temp; only aggregate and final output persisted. The 75-row likelihood caches lack explicit IDs; no safe row alignment proven. Read-only workflow 46031699064 searches for retained selection sidecars and the exact source's writer path. At writing, not complete.

## Required single-pass trace if an authorized real inference is possible later
Do not change inference decisions. At each boundary, persist the following in a private run-local directory, never in public GitHub or public Kaggle artifacts. Preserve original ID order and explicit ordered 18-class names. Every artifact has producer code SHA, input cohort SHA, model/asset version manifest, shape, dtype, coverage, and SHA256. The truth join occurs only after scoring.

| Stage | Required evidence | Purpose |
|---|---|---|
| Native core | full 18-way ID-keyed logits, per-expert logits where exposed, expert weights and availability | isolate representational vs combination failures |
| V1 selection | exact ordered selected IDs, mask, priority and per-expert coverage/status | measure who was actually scored |
| V1 likelihood | both models' row/token offsets, explicit candidate ID axis, feature scores, post-V1 full logits | separate bad gating from bad likelihood evidence |
| Apex | pre-Apex predictions, sibling/residual logits, protection mask, typography mask and final decisions | count overwrite vs correction events |
| Runtime | each stage time, peak memory, fallbacks and skip reason | establish real hidden-scale feasibility |

Admission: IDs unique, exact expected ID set for full stages; subset IDs unique and within cohort for selected stages, explicit mapping to offsets, exact label set/order, finite logits, valid score orientation; independent joins by ID/class, not positional assumptions. Compare final predictions to untouched baseline and require identical decisions with tracing enabled. Any fallback or partial stage remains explicit; no imputing missing scores. Do not export raw official competition text/IDs or raw private prediction arrays publicly. Output aggregate metrics only for authorized public report.

## Research decision
The bottleneck is unobserved stage behavior, not lack of speculative modules. Two prompt-verifier pilots are closed as prompt variants; neither answers whether an independent token-likelihood method repairs v36 mistakes. Preserve all verified positive evidence and avoid new experiments until the stage inventory and source-eligibility check establish a comparable baseline. If the retained IDs cannot be recovered, the correct conclusion is missing evidence, not failed hypothesis. Next real inference requires authorization and compute; no GPU or submission in this audit.
