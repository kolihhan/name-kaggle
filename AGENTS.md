# Repository Agent Instructions

This repository is the source of truth for Ko Lih Han's **Name That LLM** Kaggle competition work.

## Current stop decision (2026-11-04)

**USER-DIRECTED PAUSE:** The Cosmorin published 0.622 V1/V3 reproduction lane is closed after failed/blank-scored Code Competition reruns. Do **not** repeat the same Kaggle submissions, launch additional CPU/GPU probes, restart self-training, or use Public LB as a repair loop. The last verified valid Public LB anchor is **0.520**. Historic submit ID `57983359` threw a hidden-rerun Notebook exception; `57985641` later showed `COMPLETE` with blank score, **not a verified successful evaluation**.

The full dated record is at [`exp/dct-temporal-fingerprint-v3/NOW.md`](https://github.com/kolihhan/name-kaggle/blob/exp/dct-temporal-fingerprint-v3/NOW.md), commit `637e043803a08eaf63300eb8e51411db3ea38405`. That branch is the authoritative source for the halted experiment; the default `main` is historically behind and does not contain its `NOW.md`.

**Reopen only** with a materially new, testable approach explicitly authorized by the user. Redirect work to higher-value ongoing projects rather than retrying the same candidate.

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
- Do not turn a temporary method restriction into permanent doctrine. Apply the restriction/pivot-trigger contract in `RESEARCH_DOCTRINE.md`.
- Do not create Phase N+1 merely because Phase N stopped improving. If repeated local tweaks stop transferring, reconsider the representation, model family, objective, data strategy, calibration, or problem formulation.

## Completion standard

Before saying work is complete, inspect the resulting repo state and run the relevant checks. Separate verified facts, inference, and unknowns when material.
