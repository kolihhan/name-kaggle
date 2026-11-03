# Public Visibility Gate

**Status: UNSAFE — REPOSITORY IS CURRENTLY PUBLIC UNDER A NARROW USER-AUTHORIZED CPU-TRAINING EXCEPTION. RETURN TO PRIVATE AFTER THE AUTHORIZED EXECUTION WORK.**

Last verified: 2026-11-03.

This is a repository-wide safety gate. It is independent of Gate B, candidate selection, and Kaggle submission authorization.

## Narrow 2026-11-03 CPU-first exception

The user explicitly authorized continuing the current DCT-v3 work with CPU training first, even if slow.

This exception is limited to the `infra/v3-cpu-shadow-resume` lineage and the minimum default-branch dispatcher definitions required for GitHub `workflow_run` / `workflow_dispatch` to operate correctly.

Allowed:

- bounded GitHub-hosted CPU training chunks for the existing CPU lineage;
- sequential resume only from a verified prior CPU checkpoint;
- default-branch workflow definitions whose only purpose is to dispatch that bounded infra-branch CPU job;
- the immutable external Phase14 safe artifact and public pinned Qwen revision already used by the experiment;
- trainable-only CPU checkpoints and execution metadata.

Required safeguards:

- remove `data/train.csv`, `data/test.csv`, and `data/sample_submission.parquet` from every CPU runner workspace before training and assert they are absent;
- do not materialize or score official competition rows;
- do not use Public LB, Kaggle credentials, or private competition datasets in the CPU lane;
- each heavy chunk must be explicitly bounded and checkpointed;
- no unlimited automatic chain;
- resume only after the previous chunk produced valid finite training evidence;
- do not authorize a Kaggle submission;
- do not represent CPU evidence as the frozen CUDA-v3 transfer gate.

This exception does **not** make the repository public-safe and does not authorize unrelated workflows or broader exposure.

## Immediate incident state

On 2026-11-02 the repository was verified as `public` while official competition files remained tracked in Git, including:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.parquet`

Making the repository private again stops continuing exposure while cleanup is performed, but does not erase the period during which tracked files were public.

## Why public visibility is unsafe

### 1. Official Competition Data is tracked in Git

The Kaggle Data page labels the dataset CC BY-SA 4.0, but competition participation rules can impose additional data-security restrictions during the competition. Do not treat the dataset license alone as permission to expose Competition Data to non-participants.

Official competition pages:

- https://www.kaggle.com/competitions/name-that-llm/data
- https://www.kaggle.com/competitions/name-that-llm/rules

Deleting the files only from the current tree is not sufficient: previously committed files can remain reachable through Git history after repository visibility changes.

### 2. The official-data materialization workflow can commit Competition Data

`.github/workflows/materialize-official-data.yml` downloads and commits byte-identical copies of the official files.

The workflow now fails closed when the repository is public. Keep that guard.

### 3. Public Competition Code has Kaggle-specific sharing requirements

During the competition, do not assume that publishing Competition Code only on GitHub satisfies Kaggle's public-code-sharing rules. Re-check the current official rules immediately before any public release and satisfy any required competition forum/notebook publication.

## Readiness checklist

The repository may be marked `READY` for public visibility only after all of the following are verified:

- [ ] Repository visibility has first been returned to `private` while cleanup is performed.
- [ ] Remove official Competition Data from the current Git tree or replace it with a reproducible authenticated/local bootstrap path.
- [ ] Sanitize reachable Git history if official Competition Data in historical commits cannot legally be exposed to non-participants.
- [ ] Verify `.github/workflows/materialize-official-data.yml` and all other workflows cannot publish/commit Competition Data after public visibility is enabled.
- [ ] Verify external datasets retained in Git have redistribution-compatible licenses and required attribution.
- [ ] Verify repository secrets are referenced only through GitHub Secrets and are not present in files/history/logs.
- [ ] Satisfy the current Kaggle rule for public Competition Code sharing, including any required Kaggle forum/notebook publication.
- [ ] Re-check the current official competition Rules immediately before changing visibility.
- [ ] Update this file from `UNSAFE` to `READY` with dated verification evidence.

## Agent instruction

If your task involves repository visibility, GitHub Actions billing, data handling, or public sharing:

1. read this file first;
2. treat `UNSAFE` as a hard stop on further public exposure except for the explicit narrow CPU-first exception above;
3. do not claim this repository is public-safe;
4. do not rewrite Git history or delete competition evidence unless the user explicitly authorizes that destructive cleanup;
5. keep infrastructure cleanup separate from Gate B and scientific evaluation;
6. do not broaden the CPU-first exception beyond its stated scope.
