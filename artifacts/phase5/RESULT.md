# Phase 5 result — official-only style multiview

Date: 2026-10-31

## Result

All scores use the permanent 5-fold `GroupKFold(prompt_id)` mapping and score only the 2,699 official rows.

| View | OOF balanced accuracy |
|---|---:|
| Char TF-IDF + LR | 0.32967 |
| Word TF-IDF + LR | 0.28669 |
| Numeric style + LR | 0.29008 |
| Equal geometric-mean blend | **0.34821** |

Blend fold scores: `0.34444 / 0.32037 / 0.37037 / 0.35741 / 0.34860`.

Phase 1 char LinearSVC was `0.31930`, so the fixed Phase 5 blend improves the honest OOF anchor by **+0.02891**. It remains about **0.026–0.027** below the independently observed public grouped-CV references around `0.374–0.376`.

## What the OOF says

The weaker views are genuinely complementary rather than useless. Relative to the char view, the blend improves recall on 12 of 18 labels, is unchanged on 1, and decreases on 5. The largest recall gains are:

- `qwen3.8-max`: +0.1067
- `claude-sonnet-5`: +0.0867
- `gpt-6-astra`: +0.0733
- `gemini-3.1-pro-preview`: +0.0600
- `gpt-5.5`: +0.0467

The blend therefore validates the style/multiview direction, but the equal three-way probabilistic recipe is not yet a competitive sparse anchor.

## Decision

**KEEP the feature families; do not keep this exact equal-weight LR blend as the final CPU anchor.**

Next, run one prior-art-backed strong sparse configuration rather than a hyperparameter grid:

- fold-local char TF-IDF;
- fold-local word TF-IDF;
- fold-local boundary/punctuation views;
- deterministic style features with a deliberately smaller contribution;
- balanced `LinearSVC` on a single feature union.

The purpose is to test whether the remaining gap comes from the classifier/feature-combination strategy rather than from missing signal. No external data should be added until this official-only sparse anchor is established.
