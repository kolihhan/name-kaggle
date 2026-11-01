# Phase 19 — historical shadow-validation diagnostic result

Date: 2026-11-01

## Verdict

**Phase 15's local advantage over Phase 7 is robust to prompt-to-fold reassignment. Split-assignment fragility is not sufficient to explain the Public-leaderboard collapse.**

The three pre-frozen shadow assignments all reproduced a substantial positive Phase15 − Phase7 OOF balanced-accuracy delta without retuning either system. The stronger remaining explanation is external/hidden distribution mismatch, consistent with the independent Phase 18 transfer-risk audit.

This does **not** make Phase 15 the competition champion. Phase 15 still scored `0.357` Public versus Phase 7's `0.369`, and these shadows reuse the same 2,699 official rows. They diagnose fold-assignment sensitivity, not hidden-distribution transfer.

## Frozen comparison

Permanent reference:

| Split | Phase 7 BA | Phase 15 BA | Phase15 − Phase7 |
|---|---:|---:|---:|
| Permanent GroupKFold | `0.3700521999` | `0.4053218991` | `+0.0352696992` |
| Shadow A | `0.3667089237` | `0.3986502610` | `+0.0319413373` |
| Shadow B | `0.3626373353` | `0.4079194631` | `+0.0452821278` |
| Shadow C | `0.3652373850` | `0.4075441213` | `+0.0423067363` |

Across the three shadows:

| Quantity | Mean | Sample SD | Min | Max |
|---|---:|---:|---:|---:|
| Phase 7 BA | `0.3648612147` | `0.0020616949` | `0.3626373353` | `0.3667089237` |
| Phase 15 BA | `0.4047046151` | `0.0052465821` | `0.3986502610` | `0.4079194631` |
| Phase15 − Phase7 | **`+0.0398434004`** | `0.0070032271` | `+0.0319413373` | `+0.0452821278` |

The delta is positive on **3 / 3** shadow assignments.

## Prompt-group bootstrap

Each shadow used the precommitted 5,000-replicate prompt-group bootstrap with seed `20261101`.

| Shadow | Mean delta | 95% interval | P(delta > 0) |
|---|---:|---:|---:|
| A | `+0.0318016303` | `[+0.0151791573, +0.0483539790]` | `0.9998` |
| B | `+0.0451864663` | `[+0.0272101321, +0.0635091341]` | `1.0000` |
| C | `+0.0422154506` | `[+0.0255506757, +0.0592592593]` | `1.0000` |

These intervals are uncertainty descriptions only; they are not promotion thresholds.

## Stable class pattern

The per-class direction is unusually stable across A/B/C.

Positive Phase15 − Phase7 recall on all three shadows:

- `claude-fable-5`
- `claude-opus-4-6`
- `claude-opus-4-8`
- `claude-opus-5`
- `claude-sonnet-5`
- `gemini-3.1-pro-preview`
- `gpt-5.5`
- `gpt-5.6-luna`
- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `gpt-6-astra`
- `qwen3.8-max`

Negative on all three shadows:

- `gemini-3.6-flash`
- `gemini-3.7-flash`
- `gemini-3.8-flash`
- `glm-5.3`
- `glm-5.3-flash`
- `kimi-k3`

All eleven labels represented in Phase 15's terms-clean external corpus are in the stable-gain group. Most stable losses are labels without Phase 15 external coverage. This supports a coverage/domain mechanism rather than a one-split accident.

## Interpretation

### FACT

- The Phase 15 advantage survives all three frozen prompt-group reassignments.
- The three shadow deltas range from `+0.03194` to `+0.04528`, close to the permanent split's `+0.03527`.
- Every shadow bootstrap interval remains above zero.
- Phase 15 still loses the real Public leaderboard to Phase 7: `0.357` vs `0.369`.
- Phase 18 independently found materially stronger official-to-external response proximity for Phase 15's corpus than for Phase 7's control corpus.

### INFERENCE

The original permanent fold mapping was **not the main cause of the Phase 15 collapse**. Repeated reuse of that mapping remains a valid process-risk concern, but changing prompt-to-fold assignment does not remove Phase 15's local advantage.

The leading explanation is now **external/hidden distribution mismatch**: Phase 15 learns signal that generalizes across held-out official prompt groups but does not transfer to the Public hidden distribution.

### UNKNOWN

The exact hidden-distribution mismatch remains unresolved. Possible dimensions include prompt/topic bank, formatting, generation parameters, provider/source artifacts, response length, temporal effects, or public-corpus construction fingerprints. The hidden test labels are unavailable, so the repo cannot identify the exact cause from leaderboard evidence alone.

## Validation consequence

Do not return to using the permanent fold map as the sole future model-selection oracle. The project has adaptively consumed it through many sequential decisions.

However, do not misdiagnose Phase 15 as merely overfit to that one mapping. Phase 19 directly argues against that narrow explanation.

The three Phase 19 shadows are now **spent diagnostic folds**. They must not be presented later as unseen holdouts for a new candidate developed with knowledge of these results.

## Competition consequence

- Competition champion remains **Phase 7**.
- Phase 7 OOF BA: `0.3700521998508576`.
- Phase 7 Public LB: `0.369`.
- Phase 7 submission ref: `57838845`.
- Phase 15 remains rejected as competition champion despite robust official-data OOF uplift.
- No Kaggle submission is authorized by Phase 19.

## Evidence

Successful diagnostic workflow: `41283425138`.

Archived artifacts:

- Shadow A: artifact `13073794077`, digest `sha256:037833626e719be9358bb93b41df8c5edc9c63021386d089d77405a38f390d66`
- Shadow B: artifact `13074717264`, digest `sha256:521910aef02aca08794a35d8ec4192ff71e3f0ff2919a9285b4b54203ae4eb7b`
- Shadow C: artifact `13074905139`, digest `sha256:8882b7eb9396dbe75c8940668cd5de56aafa247e4d0df94b41e46cb4c11d5a7f`

Frozen shadow CSV SHA-256: `8b6cd708ba07fc7bd774ec349873a6deda484f999d1780a118b9e0b5cb95cebb`.

## Next authorized diagnostic

Run one deliberately simple, audit-only domain-separability check comparing official responses against the frozen Phase 15 external corpus, with the Phase 7 external corpus as a known-transfer control. Control for model identity so corpus classification cannot win merely because the external label sets differ. Inspect simple style/format signals first. Do not tune a competition model or submit to Kaggle from this audit.
