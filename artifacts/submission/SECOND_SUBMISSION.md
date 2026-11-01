# Second Kaggle submission — Phase 15 terms-clean public sparse

Date: 2026-11-01

## Result

**REJECT as competition champion. Keep Phase 7 as the best verified Kaggle system.**

The second real Name That LLM submission completed successfully, but its Public leaderboard score was lower than the first Phase 7 submission.

| Submission | System | Local OOF BA | Public LB |
|---|---|---:|---:|
| `57838845` | Phase 7 gated specialist | `0.3700521998508576` | **`0.369`** |
| `57861337` | Phase 15 terms-clean public sparse | `0.4053218990802883` | **`0.357`** |

Phase 15 therefore shows a large validation-to-leaderboard gap:

- OOF vs Public: `-0.0483218990802883`
- Public vs Phase 7 Public: `-0.012`

This invalidates the assumption that the Phase 14/15 external-pool OOF uplift transfers reliably to the competition hidden distribution.

## Submission identity

- competition: `name-that-llm`
- submission ref: `57861337`
- description: `Phase 15 terms-clean public sparse submission`
- status: `COMPLETE`
- Public score: `0.357`
- private score: not yet available
- Kaggle kernel: `kolihhan/name-that-llm-phase15-terms-clean`
- kernel version: `1`
- GitHub Actions run: `41276327535`

The final workflow state was SUCCESS.

## Verification chain

Before the submission was sent:

1. full repository test suite passed: **109 passed**;
2. read-only Kaggle guard confirmed exactly one existing submission, `57838845 / COMPLETE / 0.369`;
3. the frozen Phase 15 notebook bundle was built with internet disabled;
4. the Kaggle kernel reached `COMPLETE`;
5. saved `submission.parquet` passed schema, uniqueness, null, and label checks;
6. a second read-only guard again confirmed no new submission had appeared;
7. only then was kernel version 1 submitted.

The saved public notebook smoke output contained three visible test rows and valid labels. The formal code-competition submission then ran/scored through Kaggle and returned ref `57861337`.

## Interpretation

The failure is not a submission-pipeline failure. The notebook, data reconstruction, prediction generation, submission validation, and Kaggle scoring path all completed successfully.

The important evidence is model-selection evidence: the external-data permanent-fold validation used for Phase 14/15 was materially over-optimistic relative to hidden-test performance. Do not treat Phase 15's `0.4053218990802883` as a trustworthy leaderboard proxy.

Do not tune the Phase 15 external-source mix, SVC parameters, view weights, prompt filters, or label exclusions against this Public score. That would turn the leaderboard into a tuning set.

## Current competition champion

Phase 7 remains the promoted Kaggle system:

- local OOF BA: `0.3700521998508576`
- Public LB: `0.369`
- submission ref: `57838845`

The next experiment should require genuinely new evidence or a validation design that explains the Phase 15 transfer failure before another Kaggle submission is authorized.
