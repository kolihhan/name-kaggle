# RULES — Name That LLM (checked 2026-11-16)
Source precedence: current Kaggle [overview](https://www.kaggle.com/competitions/name-that-llm/overview/description), [data](https://www.kaggle.com/competitions/name-that-llm/data), [rules](https://www.kaggle.com/competitions/name-that-llm/rules) > third-party summaries. Frozen `EVALUATION_POLICY.md` still governs scientific selection. This file is a compliance inventory, NOT confirmation of personal eligibility or any submission authorization.

## Verified from Kaggle overview/data
- Metric: 18-class **balanced accuracy** (macro per-class recall). Train: 2,699 rows, 150 `prompt_id` groups; held-out test prompts are unseen. Labels must be one of the exact 18 names in overview.
- **Code competition**: notebook re-executed without internet; CPU <=9h, GPU <=9h. Output `/kaggle/working/submission.parquet` with one `id,model` pair per test ID.
- Freely and publicly available external data, including pretrained models, are permitted in notebook; verify individual license, attribution, source identity and lineage. No non-public data or leaked test labels.
- Deadline (overview): 2027-01-13 23:59 UTC; entry deadline 2027-01-06 UTC, subject to updates.

## Verification still needed from full official Rules text / authenticated competition page
- Daily official submission quota: secondary community sources report **5/day**, but full official Rules page did not render to our crawler. Treat as **UNVERIFIED**, not authorization. Verify against authenticated Kaggle account before any submission.
- Final-submission slots: secondary sources report 2. **UNVERIFIED** from official Rules text.
- Paid API / paid external generation: **NOT AUTOMATICALLY PERMITTED**. Competition overview permits *freely and publicly available* data/models; offline scoring prohibits internet API calls. Whether paid API-generated training examples or tools satisfy the competition-specific Reasonableness Standard requires confirmation of current official Rules/host guidance and reproducibility/license. Until confirmed, **do not use paid APIs/data**.
- No private team code sharing or publication of raw competition data; follow existing `PUBLIC_VISIBILITY.md` restrictions. Confirm current precise Kaggle public code-sharing wording before publication.

## Research operating gates (user contract; stricter than Kaggle)
- Never tune thresholds/weights/seeds on Public LB. No new submission until genuine independent full-18, same-row source/prompt-disjoint validation; pretrained candidate's OOF may be N/A, never fabricated.
- Keep prompt groups intact within folds; log (local CV, Public LB) only for actually comparable candidates. Report n, *independent prompt clusters*, and group-aware paired uncertainty.
- Preserve 0.691 scored reference #58375806 unless authenticated official ledger shows otherwise. Pretrained-public reproduction, not an original trained model.
- Existing `EVALUATION_POLICY.md`, `AGENTS.md` and `RESEARCH_DOCTRINE.md` take precedence on frozen contracts.
