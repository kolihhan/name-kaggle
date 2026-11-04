# Repository Agent Instructions

This repository is the source of truth for Ko Lih Han's **Name That LLM** Kaggle competition work.

## Entry order

For substantive work in this repository:

1. Read `PUBLIC_VISIBILITY.md` before any task involving repository visibility, Actions billing, official competition data, or public sharing.
2. Read `README.md` and `EVALUATION_POLICY.md` for current scientific/evaluation state.
3. Read `RESEARCH_DOCTRINE.md` before proposing a substantial new research lane, extending a sequence of local feature/model/ensemble tweaks, or preserving/rejecting a major method class.
4. Read the relevant current contract/result artifact under `artifacts/`.
5. Use the personal workflow router from `kolihhan/kolihhan` when available: Claude-like outer loop, then the smallest relevant inner mode.
6. Execute and verify; update the canonical repo state after durable progress.

Do not restart from chat assumptions when repo state is available.

## Canonical research doctrine

`RESEARCH_DOCTRINE.md` is the durable anti-lock-in rule for this competition.

- Every major research restriction must record why it exists, current supporting evidence, a pivot trigger, and ACTIVE / REOPENED / RETIRED status.
- Maintain both the smallest local next experiment and the global bottleneck/model-class hypothesis.
- Revisit the global diagnosis when local tweaks show diminishing transfer, CV/hidden evidence diverges, or increasingly complex rules are needed to preserve the incumbent.
- Do not interpret the doctrine as permission to overfit or as a requirement to use a more sophisticated method. Frozen evaluation contracts and anti-leakage rules remain binding.

## Public visibility gate

`PUBLIC_VISIBILITY.md` records a **public-by-user-choice / not-certified-public-safe** status. The 2026-11-04 user decision permits continued GitHub Actions and Colab training without a mandatory private-visibility switch. Treat this as an operational authorization, not as proof that historic competition data was purged.

- This repository is already public by explicit user choice. Do not broaden exposure by uploading new official Competition Data or secrets; historical Git exposure remains a separately documented, unverified cleanup issue.
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
- Do not turn a temporary method restriction into permanent doctrine. Apply the restriction/pivot-trigger contract in `RESEARCH_DOCTRINE.md`.
- Do not create Phase N+1 merely because Phase N stopped improving. If repeated local tweaks stop transferring, reconsider the representation, model family, objective, data strategy, calibration, or problem formulation.

## Completion standard

Before saying work is complete, inspect the resulting repo state and run the relevant checks. Separate verified facts, inference, and unknowns when material.
