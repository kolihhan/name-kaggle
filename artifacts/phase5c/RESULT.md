# Phase 5c result — reference RichStyle reproduction

Date: 2026-10-31

## Result

- Phase 5b broad 155-style-feature anchor: **0.3670892**
- Phase 5c narrower 108-feature reference-style reproduction: **0.3670967**
- Delta: **+0.0000075**

The style extractor does not explain the apparent ~0.007 gap to the public `0.37560` figure.

## Why the public number is not directly comparable

The public `arshjohari1/Kaggle-LLM-Challenge` evaluation code uses `StratifiedGroupKFold(..., shuffle=True, random_state=seed)`. This repo's permanent validation contract uses deterministic `GroupKFold(prompt_id)` frozen in Phase 1.

Therefore `0.36709` vs `0.37560` is not an apples-to-apples model comparison. We will not change the permanent folds to chase an external CV number.

## Decision

- Keep **Phase 5b `0.36709`** as the canonical local CPU anchor.
- Keep the broader local style extractor; Phase 5c gives no meaningful reason to replace it.
- Stop sparse micro-tuning on the current official-only folds.
- Next legitimate question: does the already-audited 2,503-row external source improve this stronger sparse anchor under the same permanent official OOF evaluation?
