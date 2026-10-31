# Phase 6 — Strong sparse + audited external result

Date: 2026-10-31

## Result

The fixed Phase 5b strong-sparse model was trained on each official training partition plus all 2,503 audited GPT/Gemini external rows, while scoring only untouched official validation rows.

| Condition | OOF balanced accuracy | Δ vs Phase 5b |
|---|---:|---:|
| Official-only Phase 5b anchor | **0.36709** | — |
| Phase 6 + 2,503 external rows | **0.35967** | **-0.00742** |

Phase 6 fold scores: `0.38889 / 0.33148 / 0.34444 / 0.37963 / 0.35396`.

## What happened

The external source is informative but strongly asymmetric:

- 8 externally covered labels: mean recall `0.40833 → 0.50917` (**+0.10083**)
- 10 uncovered labels: mean recall `0.33409 → 0.24008` (**-0.09402**)

Largest gains included:

- `gemini-3.8-flash`: **+0.32667**
- `gpt-5.5`: **+0.13333**
- `gpt-5.6-luna`: **+0.13333**
- `gpt-5.6-terra`: **+0.12667**
- `gemini-3.6-flash`: **+0.10667**

Largest losses included:

- `gemini-3.7-flash` (not present externally): **-0.29333**
- `qwen3.8-max`: **-0.24667**
- `kimi-k3`: **-0.13333**
- `claude-opus-4-6`: **-0.11333**
- `gemini-3.1-pro-preview`: **-0.08667** despite being externally covered.

## Decision

**REJECT global external augmentation for the strong-sparse path.** Keep the official-only Phase 5b model at **0.36709** as the canonical CPU anchor.

This does not mean the external corpus has no useful signal. It improves the represented GPT/Gemini classes substantially, but using it globally shifts the shared sparse representation enough to hurt uncovered labels more than the covered-label gains recover.

Do not respond by tuning external sample counts, class weights, or SVC hyperparameters on the same permanent folds. Any later use of this source needs a structurally different hypothesis—such as a separately trained specialist combined with the official-only anchor—rather than another global-augmentation sweep.

## Verification

The completed GitHub Actions run passed the Phase 6 contract tests and full experiment step. The exported OOF file contains exactly 2,699 unique official IDs, zero nulls, all 18 valid prediction labels, permanent fold sizes `540/540/540/540/539`, and no prompt split across folds.
