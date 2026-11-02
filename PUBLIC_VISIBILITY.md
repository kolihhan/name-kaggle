# Public Visibility Gate

**Status: BLOCKED — DO NOT MAKE THIS REPOSITORY PUBLIC**

Last verified: 2026-11-02.

This is a repository-wide safety gate. It is independent of Gate B, candidate selection, and Kaggle submission authorization.

## Why public visibility is currently blocked

### 1. Official Competition Data is tracked in Git

The current repository contains official competition files including:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.parquet`

The Kaggle Data page labels the dataset CC BY-SA 4.0, but competition participation rules can impose additional data-security restrictions during the competition. Do not treat the dataset license alone as permission to expose Competition Data to non-participants.

Official competition pages:

- https://www.kaggle.com/competitions/name-that-llm/data
- https://www.kaggle.com/competitions/name-that-llm/rules

A deletion from the current tree alone is not sufficient: previously committed files can remain reachable through Git history after repository visibility changes.

### 2. The official-data materialization workflow can commit Competition Data

`.github/workflows/materialize-official-data.yml` downloads and commits byte-identical copies of the official files.

This workflow must fail closed when the repository is public. It must not be used as a public redistribution mechanism.

### 3. Public Competition Code has Kaggle-specific sharing requirements

During the competition, do not assume that publishing Competition Code only on GitHub satisfies Kaggle's public-code-sharing rules. Re-check the current official rules immediately before any public release and satisfy any required competition forum/notebook publication.

## Readiness checklist

The repository may be marked `READY` for public visibility only after all of the following are verified:

- [ ] Remove official Competition Data from the current Git tree or replace it with a reproducible authenticated/local bootstrap path.
- [ ] Sanitize reachable Git history if official Competition Data in historical commits cannot legally be exposed to non-participants.
- [ ] Verify `.github/workflows/materialize-official-data.yml` and all other workflows cannot publish/commit Competition Data after public visibility is enabled.
- [ ] Verify external datasets retained in Git have redistribution-compatible licenses and required attribution.
- [ ] Verify repository secrets are referenced only through GitHub Secrets and are not present in files/history/logs.
- [ ] Satisfy the current Kaggle rule for public Competition Code sharing, including any required Kaggle forum/notebook publication.
- [ ] Re-check the current official competition Rules immediately before changing visibility.
- [ ] Update this file from `BLOCKED` to `READY` with dated verification evidence.

## Agent instruction

If your task involves repository visibility, GitHub Actions billing, data handling, or public sharing:

1. read this file first;
2. treat `BLOCKED` as a hard gate;
3. do not change repository visibility to public;
4. do not rewrite Git history or delete competition evidence unless the user explicitly authorizes that destructive cleanup;
5. keep this infrastructure cleanup separate from Gate B and the scientific evaluation process.

Saving GitHub Actions minutes is not sufficient reason to bypass this gate.
