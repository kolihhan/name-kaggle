# Phase 11 — Fable 5 binary override result

Date: 2026-10-31

## Verdict

**REJECT. Do not promote Phase 11.**

The single frozen permanent-fold evaluation completed successfully under the precommitted contract. Fable recall improved slightly, but overall balanced accuracy decreased, so the promotion rule failed.

## Frozen result

- Phase 7 balanced accuracy: **0.3700521998508576**
- Phase 11 balanced accuracy: **0.3696843151876709**
- balanced-accuracy delta: **-0.00036788466318665725**
- Phase 7 `claude-fable-5` recall: **0.14093959731543623**
- Phase 11 `claude-fable-5` recall: **0.1476510067114094**
- Fable recall delta: **+0.006711409395973172**
- no Phase 7 Fable prediction removed: **PASS**
- every changed prediction changed only to Fable: **PASS**
- `promote_phase11`: **false**

## What happened

The binary detector was extremely conservative on official OOF data:

- detector balanced accuracy: **0.5019831556783788**
- detector Fable recall: **0.006711409395973154**
- detector `__other__` recall: **0.9972549019607844**
- detector predicted Fable on only **8 / 2,699** official OOF rows

After the monotonic override, only **5** Phase 7 predictions actually changed:

- **1** was a true Fable rescue;
- **4** were non-Fable false positives;
- Phase 7 predicted Fable on 83 rows; Phase 11 predicted Fable on 88 rows.

The one rescued Fable row raised Fable recall, but the four false-positive overrides reduced recall elsewhere, including `claude-opus-4-6` and `gpt-5.6-luna`, so the macro balanced-accuracy result fell.

## Interpretation

The admitted FusionCube corpus is clean enough for training, but this frozen binary mechanism did not transfer well enough to the competition distribution. The detector mostly learned `__other__` and almost never fired on official Fable examples.

This is a clean mechanism rejection. It does **not** authorize post-hoc threshold tuning.

## Contract consequence

Do **not** tune the following against these permanent folds:

- SVC decision threshold;
- `C` or class weights;
- positive/negative weighting or resampling;
- the 344-row source subset;
- feature ranges or feature caps;
- override exceptions or routing rules.

Any future experiment must use genuinely new evidence or a separately justified mechanism frozen before another permanent-fold evaluation.

## Kaggle consequence

**Do not submit Phase 11 to Kaggle.**

The promoted system remains **Phase 7**, OOF balanced accuracy **0.3700521998508576**.

## Evidence

- frozen contract: `artifacts/phase11/CONTRACT.md`
- admission audit: `artifacts/phase11/fusioncube_fable5_audit.json`
- machine-readable result: `artifacts/phase11/fable_binary_override.json`
- successful frozen GitHub Actions run: **41217629105**
- workflow artifact: **13049175923**
- artifact digest: `sha256:9168e3b5f13119beef5a8ae90002ade94bd62eea0d81c900a25f2c790fde1d65`
