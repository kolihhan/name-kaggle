# Phase 12 — HealthBench Opus 5 / Sonnet 5 paired specialist result

Date: 2026-10-31

## Verdict

**REJECT. Keep Phase 7 promoted. Do not submit this system to Kaggle.**

The frozen paired specialist contains strong Opus 5 / Sonnet 5 signal when evaluated directly on the two true classes, but the frozen Phase 7 pair-routing rule does not convert that signal into a better final classifier.

## Verification chain

- Frozen contract: `artifacts/phase12/CONTRACT.md`
- HealthBench admission audit: `artifacts/phase12/healthbench_anthropic_audit.json`
- Full test suite before the permanent-fold shot: workflow `41219926001` — **SUCCESS**
- Frozen CV workflow: `41220411996` — **SUCCESS**
- Frozen result artifact: `13049648474`
- Artifact digest: `sha256:8d558956f23e321f893956e23afce720ceacf4cacce929609afa9ad439ecd0f2`
- HealthBench source revision: `7a116ea7b4fefe0c72eabb8398ec390c6375f823`
- Permanent fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`

The run re-ran the admission gate before modeling and again confirmed `training_authorized=true` for exactly 600 Opus 5 and 610 Sonnet 5 response-only rows.

## Frozen comparison

| Metric | Phase 7 | Phase 12 | Delta |
|---|---:|---:|---:|
| Balanced accuracy | 0.3700521998508576 | 0.3685707183693761 | -0.0014814814814814725 |
| Opus 5 recall | 0.6133333333333333 | 0.6000000000000000 | -0.0133333333333333 |
| Sonnet 5 recall | 0.3800000000000000 | 0.36666666666666664 | -0.0133333333333334 |
| Pair mean recall | 0.49666666666666665 | 0.4833333333333333 | -0.01333333333333336 |

The promotion contract therefore fails both required gain gates.

## Specialist-only signal

The binary specialist itself was strong on the 300 official rows whose true labels are in the pair:

- pair balanced accuracy: **0.8200000000000001**
- Opus 5 recall: **0.8066666666666666**
- Sonnet 5 recall: **0.8333333333333334**

This is real discriminative signal. The failure is in the tested routing mechanism, not a claim that the HealthBench corpus contains no useful signal.

## What the frozen gate actually changed

The Phase 7 baseline predicted Opus 5 or Sonnet 5 on 383 rows. The frozen pair gate changed only 21 final predictions:

- 12 Opus 5 -> Sonnet 5
- 9 Sonnet 5 -> Opus 5
- 2 changed rows rescued an incorrect Phase 7 prediction
- 6 changed rows broke a correct Phase 7 prediction
- 13 changed rows stayed incorrect after the swap

Net effect: both target recalls fell by exactly 2/150 = 0.0133333333333333.

The structural safety invariant worked as designed:

- all other 16 per-class recalls were exactly preserved;
- every changed prediction stayed inside `{claude-opus-5, claude-sonnet-5}`.

## Decision

`promote_phase12 = false`.

Do **not** tune thresholds, margins, C, feature weights, sample counts, external weighting, prompt subsets, or the pair gate using these same permanent folds. That would turn this one-shot test into result-dependent tuning.

No second Kaggle submission is authorized. The promoted competition system remains Phase 7 (`OOF 0.3700521998508576`, Public LB `0.369`).

## Next

Move back to reconnaissance for genuinely new information. Priority should be a clean pre-competition source for a still-uncovered label such as Gemini 3.7 Flash, or a materially different representation/routing hypothesis justified before another permanent-fold shot. Do not continue polishing the rejected Phase 12 pair gate.
