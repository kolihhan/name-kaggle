# Repository Agent Instructions

This repository is the source of truth for Ko Lih Han's **Name That LLM** Kaggle competition work.

## Entry order

For substantive work in this repository:

1. Read `PUBLIC_VISIBILITY.md` before any task involving repository visibility, Actions billing, official competition data, or public sharing.
2. Read `README.md` and `EVALUATION_POLICY.md` for current scientific/evaluation state.
3. Read the relevant current contract/result artifact under `artifacts/`.
4. Use the personal workflow router from `kolihhan/kolihhan` when available: Claude-like outer loop, then the smallest relevant inner mode.
5. Execute and verify; update the canonical repo state after durable progress.

Do not restart from chat assumptions when repo state is available.

## Public visibility gate

`PUBLIC_VISIBILITY.md` currently has status **BLOCKED**. Treat that as a hard repository-wide gate.

- Do not make this repository public while official Competition Data remains in reachable Git history or another redistribution blocker remains.
- Do not weaken public-safety guards in workflows that fetch or materialize competition data.
- Do not assume the dataset's CC BY-SA 4.0 label overrides competition-specific data-security obligations.
- Do not assume public GitHub code alone satisfies Kaggle's public Competition Code sharing requirements.
- Destructive cleanup such as history rewriting requires explicit user authorization unless the user already authorized that exact action.
- Public-safety infrastructure work must not change Gate B, permanent folds, candidate thresholds/configuration, evaluation contracts, or submission authorization.

## Experiment discipline

- Preserve the frozen evaluation policy and permanent fold mapping unless a new approved contract explicitly changes them.
- Treat repeatedly reused official folds as adaptive development evidence, not fresh independent transfer evidence.
- Do not use Kaggle Public LB as a tuning loop.
- Change one major scientific variable at a time.
- Do not convert infrastructure/public-safety work into a new experiment phase.

## Completion standard

Before saying work is complete, inspect the resulting repo state and run the relevant checks. Separate verified facts, inference, and unknowns when material.
