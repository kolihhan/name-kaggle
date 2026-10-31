# First Kaggle submission — Name That LLM

Date: 2026-10-31

## Result

- Competition: `name-that-llm`
- System: Phase 7 gated specialist
- Local frozen OOF balanced accuracy: **0.3700521999**
- Kaggle kernel: `kolihhan/name-that-llm-phase7-first-submission`
- Kernel version: **1**
- Submission ref: **57838845**
- Submission status: **COMPLETE**
- Public score: **0.369**
- Private score: not available during the competition

The displayed public score is within about **0.00105** of the frozen local OOF score. Because Kaggle reports the public score to three decimals here, do not over-interpret the exact difference, but there is no evidence of a large local-to-public validation collapse.

## Submission integrity

The submitted notebook used the frozen Phase 7 pipeline rather than a leaderboard-tuned variant:

- official training data only for the base anchor;
- the same eight-label GPT/Gemini specialist used in Phase 7;
- the same deterministic feature construction and `LinearSVC(C=1.5)` configuration;
- the same structural gate: specialist replacement only when the base anchor predicts one of the eight covered labels;
- internet disabled in the Kaggle notebook.

The public external dataset `abdullah1afify/ntl-generations` had grown to 3,559 rows by submission time. The notebook therefore did **not** train on the current full dataset. It reconstructed the exact frozen Phase 7 snapshot by SHA-256 row hashes:

- frozen snapshot rows: **2,503**
- frozen rows still present in public dataset: **2,503 / 2,503**
- public-only newer rows excluded: **1,056**
- frozen-only missing rows: **0**

## Execution record

The first API submit attempt was rejected before creating a submission because the code-competition command omitted the required kernel version number. The notebook itself had already completed successfully and produced a valid `submission.parquet`.

The corrected command submitted the already-verified kernel **version 1**. Kaggle returned submission ref **57838845** and scored it **0.369**.

## Decision

Treat **0.369 Public LB** as the first real external anchor for this repository.

Phase 7 remains the current promoted system. The close agreement between frozen grouped OOF (`0.3700521999`) and the first Public LB (`0.369`) supports keeping the current prompt-grouped validation contract rather than redesigning validation in response to this submission.

Do not tune directly to this single public score. Future experiments should continue to be selected under the frozen local contract, then use Kaggle submissions sparingly as external checks.
