# Phase 17 — global prior balancing result

Date: 2026-11-01

## Verdict

**REJECT Phase 17. Do not tune this Sinkhorn configuration on the permanent folds and do not submit it to Kaggle.**

The single frozen experiment passed every implementation and validation invariant, but uniform global prior balancing slightly reduced balanced accuracy.

## Frozen comparison

| System | OOF balanced accuracy | Delta |
|---|---:|---:|
| Phase 5b official-only sparse anchor | `0.36708923688789463` | — |
| **Phase 17 fixed Sinkhorn** | **`0.3667238379318916`** | **`-0.00036539895600301575` vs Phase 5b** |
| Phase 7 promotion threshold | `0.3700521998508576` | Phase 17 is `-0.003328361918965961` below |

`promote_phase17 = false`.

Fold BAs after the fixed global transform:

`0.375926 / 0.338889 / 0.377778 / 0.387037 / 0.354023`

## What was tested

The base learner was exactly the canonical Phase 5b official-only model. For every untouched validation row, the experiment collected the full 18-class `LinearSVC.decision_function` margins. Only after all 2,699 OOF margin vectors were complete did it apply the pre-committed label-free transform:

1. row softmax at temperature exactly `1.0`;
2. uniform target class mass `2699 / 18 = 149.94444444444446`;
3. exactly 100 iterative column-balance / row-normalization steps;
4. final argmax.

No external data, true labels, prompt IDs, fold IDs, per-class recall, or leaderboard information entered the transform.

## Invariants

All committed invariants passed:

- OOF margin matrix: `2699 x 18`;
- all margins finite;
- transformed values finite;
- all transformed rows normalized;
- raw margin argmax reproduced Phase 5b exactly at `0.36708923688789463`;
- external training rows: `0`;
- final class probability masses reached the uniform target to max absolute error `1.7053025658242404e-13`;
- 2,699 OOF rows / 2,699 unique IDs / no null predictions.

This means the negative result is not an implementation failure. The global equal-mass constraint itself did not improve this base score matrix.

## Selected recall movement

The transform helped some classes, including:

- `claude-fable-5`: `0.14094 -> 0.15436`;
- `claude-opus-5`: `0.61333 -> 0.64667`;
- `gemini-3.1-pro-preview`: `0.79333 -> 0.83333`;
- `gemini-3.7-flash`: `0.39333 -> 0.41333`;
- `gpt-6-astra`: `0.62000 -> 0.65333`;
- `qwen3.8-max`: `0.47333 -> 0.50667`.

But losses elsewhere were larger in aggregate, especially `kimi-k3` (`0.24000 -> 0.16667`) and `glm-5.3-flash` (`0.20000 -> 0.16667`).

## Consequence

Do **not** rescue this result by tuning:

- softmax temperature;
- iteration count;
- class priors;
- per-class multipliers;
- score clipping/scaling;
- a blend weight with raw Phase 5b predictions.

Those are now result-dependent knobs on the permanent folds.

A next experiment must add genuinely new evidence or representation.

## Evidence

- frozen contract: `artifacts/phase17/CONTRACT.md`
- implementation: `scripts/run_phase17_global_balance.py`
- successful full test run: `41281294225`
- successful frozen CV workflow: `41281667159`
- workflow artifact: `phase17-global-balance-frozen-shot`, ID `13073625903`
- artifact digest: `sha256:699a172d137aa7fe3a25b6016ef23158f33368537fa71c6a337396a8e745503e`

No Kaggle submission was made from Phase 17.
