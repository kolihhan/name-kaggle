# Public Visibility Gate

**Status: PUBLIC BY USER CHOICE; NOT VERIFIED PUBLIC-SAFE.**

Last verified: 2026-11-02.

This is a repository-wide safety gate. It is independent of Gate B, candidate selection, and Kaggle submission authorization.

## 2026-11-04 current user direction (supersedes mandatory return-private instruction)

User explicitly confirmed that they want to **keep using the public GitHub repo** and do not want training or GitHub Actions held up by repeatedly requiring a visibility reversal. Do **not** insist on making this repo private after every training or preflight job, and do not claim the older temporary return-private condition is still a user requirement. GitHub Actions and narrowly scoped, data-safe GPU training are authorized workflows while the repository is public.

This is an **operational preference**, **not** a verification that Git history was sanitized. Removing official data from current branches or sparse-checkouts does *not* remove previously committed bytes from reachable Git history. Historical exposure was documented below; its cleanup has not been verified here. Keep the facts separate from the user's accepted workflow choice.

Minimal safeguards still apply: never commit credentials/secrets or official competition data, do not weaken the official-data materialization fail-closed guard, and preserve the frozen evaluation/holdout and submission authorization boundaries. Do not re-open the same public/private warning on every routine training turn; revisit only if new material evidence or a specific release/sharing request warrants it.

## Temporary execution exception

On 2026-11-02 the user explicitly authorized temporarily changing `kolihhan/name-kaggle` to `public` so the direct non-GitHub-Actions submission-preflight/training path can be executed, with the instruction to return the repository to `private` after training/preflight completes.

This exception does **not** mean the repository is public-safe and does not erase the prior exposure incident. While this exception is active:

- do not use GitHub Actions for new work;
- do not broaden public exposure beyond what is necessary for the direct execution path;
- do not publish secrets, credentials, private Kaggle datasets, or new competition artifacts to GitHub;
- keep Kaggle submission unauthorized until the frozen preflight gate passes;
- return the repository to `private` after the external preflight/training run completes or if the execution attempt is abandoned.

Current direct execution implementation lives on `exp/first-kaggle-submission` and uses `scripts/two_stage_lora24_direct_preflight.py`. It materializes the fixed private Kaggle base/runtime inputs, launches the private internet-disabled T4 preflight kernel, verifies the resulting submission artifacts, and does **not** submit to the competition.

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
2. recognize that the 2026-11-04 user direction authorizes continued public-repo GitHub Actions training, while historical public-data safety remains unverified;
3. do not claim this repository is public-safe;
4. do not rewrite Git history or delete competition evidence unless the user explicitly authorizes that destructive cleanup;
5. keep infrastructure cleanup separate from Gate B and the scientific evaluation process;
6. do not enforce an automatic return-to-private or GitHub Actions ban; guard against new data/credential exposure and keep science evaluation rules separate.

Saving GitHub Actions minutes by itself is not sufficient reason to bypass this gate.
