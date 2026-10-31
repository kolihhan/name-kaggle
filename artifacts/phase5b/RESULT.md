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

The feature-union + LinearSVC strategy explains most of the gap left by Phase 5. It remains about `0.0069–0.0085` below independent public prompt-grouped references around `0.374–0.376`.

A source-level comparison against the public `multi_c150_s020` implementation shows that the char, word, boundary, punctuation, `C=1.5`, and style weight `0.2` settings are aligned. The main remaining implementation difference is the deterministic style extractor: this repo currently uses the broader 155-feature Phase 5 extractor, while the reference candidate uses a narrower RichStyle definition.

Next: run one exact-style reproduction while keeping every other Phase 5b setting fixed. This is a reproduction check, not a hyperparameter search.
