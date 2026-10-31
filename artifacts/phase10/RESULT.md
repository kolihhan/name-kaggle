# Phase 10 — Opus 4.8 nine-class specialist result

Date: 2026-10-31

## Verdict

**REJECT. Do not promote Phase 10.**

The frozen run completed successfully under the committed contract, but it did not satisfy the promotion rule.

## Frozen result

- Phase 7 reference balanced accuracy: **0.3700521998508576**
- Phase 10 balanced accuracy: **0.37005219985085763**
- reported numeric delta: **+5.551115123125783e-17**
- interpretation: **no material balanced-accuracy gain; floating-point equality**
- Phase 7 `claude-opus-4-8` recall: **0.20666666666666667**
- Phase 10 final `claude-opus-4-8` recall: **0.20666666666666667**
- Opus 4.8 recall delta: **0.0**
- remaining-nine uncovered recall invariant: **PASS**
- anchor predictions outside the nine-class gate unchanged: **PASS**

The contract required both strictly better overall balanced accuracy and strictly better Opus 4.8 recall. The Opus 4.8 condition failed, so `promote_phase10=false`.

## What the new data actually learned

The admitted ARC corpus was not useless. The nine-class specialist itself reached:

- covered-label balanced accuracy: **0.5362962962962964**
- specialist-only `claude-opus-4-8` recall: **0.6266666666666667**

On the 150 official Opus 4.8 rows:

- the specialist identified **94/150** as Opus 4.8;
- **63** of those correct specialist predictions were blocked because the Phase 5b anchor predicted a label outside the nine-class gate;
- inside the gate, the new specialist rescued **4** previously wrong Opus 4.8 rows but also changed **4** previously correct Opus 4.8 rows away from Opus 4.8;
- final Opus 4.8 recall therefore remained exactly unchanged.

This is a routing limitation of the tested mechanism, not evidence that the ARC Opus 4.8 responses contain no model-identification signal.

## Contract consequence

Do **not** tune the following against the same permanent folds:

- gate thresholds or gate membership;
- confidence routing;
- sample counts or ARC settings;
- external-data weights;
- SVC `C`, class weights, or feature limits;
- post-hoc class-specific exceptions.

Those would be result-dependent tuning of the rejected Phase 10 mechanism.

A future experiment needs genuinely new signal or a separately justified mechanism defined before another permanent-fold evaluation.

## Kaggle consequence

**Do not spend a second Kaggle submission on Phase 10.**

The promoted system remains **Phase 7** with frozen OOF balanced accuracy **0.3700521998508576**.

## Evidence

- machine-readable result: `artifacts/phase10/opus48_specialist.json`
- admission audit: `artifacts/phase10/arc_opus48_audit.json`
- frozen contract: `artifacts/phase10/CONTRACT.md`
- successful GitHub Actions run: `41208406346`
