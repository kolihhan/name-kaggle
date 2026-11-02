# Public Visibility Gate

**Status: UNSAFE — REPOSITORY IS CURRENTLY PUBLIC. RETURN TO PRIVATE BEFORE CONTINUING PUBLIC-SAFE CLEANUP.**

Last verified: 2026-11-02.

This is a repository-wide safety gate. It is independent of Gate B, candidate selection, and Kaggle submission authorization.

## Immediate incident state

On 2026-11-02 the repository was verified as `public` while official competition files remained tracked in Git, including:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.parquet`

**Immediate required action:** change repository visibility back to `private` before continuing cleanup. Making the repository private again stops continuing exposure while cleanup is performed, but does not erase the period during which tracked files were public.

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
2. treat `UNSAFE` as a hard stop on further public exposure;
3. do not claim this repository is public-safe;
4. do not rewrite Git history or delete competition evidence unless the user explicitly authorizes that destructive cleanup;
5. keep this infrastructure cleanup separate from Gate B and the scientific evaluation process.

Saving GitHub Actions minutes is not sufficient reason to bypass this gate.
