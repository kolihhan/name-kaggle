# Phase 9 result — REJECT

Date: 2026-10-31

## Verdict

**Do not promote Phase 9. Keep Phase 7 as the current best system.**

The one-shot frozen Phase 9 experiment completed successfully in GitHub Actions run `41199035742`. The implementation and structural invariants passed, but the model result was negative.

| System | Official 5-fold OOF balanced accuracy |
|---|---:|
| Phase 7 reproduced baseline | **0.3700521999** |
| Phase 9 five-class gated specialist | **0.3659781258** |
| Delta | **-0.0040740741** |

Phase 7 therefore remains the best promoted CPU OOF system.

## Experiment integrity

The same run reproduced Phase 7 before applying the Phase 9 gate. The reproduced Phase 7 score matches the promoted Phase 7 result.

Phase 9 used exactly the frozen 2,921-row pre-competition external candidate and exactly five specialist labels:

- `claude-opus-4-6`
- `glm-5.3`
- `glm-5.3-flash`
- `kimi-k3`
- `qwen3.8-max`

No threshold, class weight, `C`, feature, label-set, row-selection, or gate sweep was performed after observing the result.

The structural safety contract held:

- recall for all other 13 labels was exactly unchanged;
- Phase 7 predictions outside the five-label gate were never changed;
- OOF contains 2,699 rows / 2,699 unique IDs / zero nulls;
- fold sizes remain `540/540/540/540/539`.

## What happened

The specialist itself could classify its five true labels reasonably well (`0.505333` balanced accuracy on those true-label rows), but using it as a gate on Phase 7 predictions made the final system worse.

Per-label recall deltas versus Phase 7:

| Label | Delta |
|---|---:|
| claude-opus-4-6 | -0.0200 |
| glm-5.3 | -0.0267 |
| glm-5.3-flash | -0.0133 |
| kimi-k3 | +0.0200 |
| qwen3.8-max | -0.0333 |

The gate touched 182 rows. Relative to Phase 7 it rescued 27 errors but broke 38 previously correct predictions, net **-11 correct rows**.

This is useful evidence: clean/pre-competition external data does not automatically transfer to the official attribution distribution, even when the specialist is structurally isolated from the other 13 labels.

## Decision

- Phase 9 promotion: **REJECT**
- current best: **Phase 7, BA 0.3700521999**
- do not rescue Phase 9 by tuning this same permanent-fold route
- keep the Phase 8 provenance/audit work as reusable data intelligence

Any next experiment should introduce a genuinely different signal or validation-safe modeling hypothesis, not a post-hoc weight/threshold/subset tweak of this failed five-class specialist.
