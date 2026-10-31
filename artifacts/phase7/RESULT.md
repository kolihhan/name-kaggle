# Phase 7 — Gated external specialist result

Date: 2026-10-31

## Result

The fixed 18-class Phase 5b anchor was reproduced exactly, while a separate 8-class strong-sparse specialist was trained only on covered official rows plus all 2,503 audited external rows. The specialist could replace a label only when the anchor had already predicted one of those 8 covered classes.

| Condition | OOF balanced accuracy | Δ vs anchor |
|---|---:|---:|
| Reproduced Phase 5b anchor | **0.367089** | — |
| **Phase 7 gated specialist** | **0.370052** | **+0.002963** |

Ensemble fold scores: `0.38333 / 0.33889 / 0.37222 / 0.38704 / 0.36884`.

The specialist itself reached **0.53167** balanced accuracy across the 1,200 official OOF rows whose true label is in the covered 8-class set.

## Gate behavior

- Anchor predicted one of the 8 covered labels on **1,229** rows.
- Anchor predicted one of the 10 uncovered labels on **1,470** rows; every one remained unchanged.
- Specialist changed **454** final predictions.
- Relative to the anchor, it corrected **110** previously wrong rows and broke **102** previously correct rows: net **+8** correct rows.

Mean recall changed as intended:

- covered 8 labels: `0.40833 → 0.41500` (**+0.00667**)
- uncovered 10 labels: `0.33409 → 0.33409` (**0.00000**)

Largest covered-label gains:

- `gemini-3.8-flash`: **+0.10000**
- `gpt-5.6-luna`: **+0.06667**
- `gpt-5.6-terra`: **+0.05333**
- `gpt-5.5`: **+0.02000**
- `gpt-5.6-sol`: **+0.02000**

Covered-label losses:

- `gemini-3.1-pro-preview`: **-0.14667**
- `gpt-6-astra`: **-0.04667**
- `gemini-3.6-flash`: **-0.01333**

All 10 uncovered per-model recall deltas were exactly **0**.

## Decision

**PROMOTE Phase 7 under the frozen contract.** It improves OOF balanced accuracy from **0.367089** to **0.370052** while preserving every uncovered-class recall exactly.

Treat the gain as real but modest. Do not tune the gate, specialist sample count, covered-label set, class weights, or SVC C against these permanent folds. The next useful experiment should add genuinely different information or representation rather than squeeze this same gate.

## Verification

The formal GitHub Actions run passed all 8 Phase 7 contract tests and completed the frozen experiment. The exported OOF artifact contains exactly 2,699 unique official IDs, zero nulls, 18 valid final prediction labels, specialist predictions from exactly the fixed 8 labels, fold sizes `540/540/540/540/539`, and no prompt split across folds. The reproduced anchor balanced accuracy and all five fold scores match the canonical Phase 5b artifact exactly.
