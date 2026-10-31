# Phase 5b result — official-only strong sparse anchor

Date: 2026-10-31

## Result

Permanent 5-fold `GroupKFold(prompt_id)`, 2,699 official rows, no external data:

- OOF balanced accuracy: **0.36709**
- Fold scores: `0.38333 / 0.34444 / 0.37593 / 0.38148 / 0.35032`
- Delta vs Phase 5 equal LR blend (`0.34821`): **+0.01888**
- Delta vs Phase 1 char LinearSVC (`0.31930`): **+0.04779**

## Decision

**PROMOTE as the current canonical CPU anchor.**

The feature-union + LinearSVC strategy explains most of the gap left by Phase 5.

Do **not** tune toward the independently observed public `0.37560` score as if it were measured on this repository's folds. The public reference uses `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` and a different fold fingerprint (`d019f60a...`); this repository keeps its permanent `GroupKFold(prompt_id)` mapping (`bf6f47bf...`) unchanged. The score gap is therefore not a clean implementation gap.

Hard labels remain concentrated in sibling-model families, especially `claude-fable-5`, `gpt-5.6-sol`, the GLM pair, and the GPT-5.6 family.

## Next

Do not run a C/style-weight grid or exact-style reproduction on the permanent folds.

The next high-ROI test is **representation-matched external transfer**: train this exact strong sparse model on each official training fold plus the already-audited 2,503-row GPT/Gemini external source, while scoring only untouched official validation rows. Compare directly against **0.36709**.
