# Kaggle public frontier candidates — authenticated, no GPU (2026-11-11 MYT)

Status: **PUBLIC SOURCE AUDIT COMPLETE; OWN OFFICIAL SCORE UNCHANGED**.

## Why this exists

Previous web search overlooked newer potential 0.682/0.683 public Notebook sources. An authenticated, read-only Kaggle CLI scan on the existing research branch found three public candidates. This is new **competitive source intelligence**, not a new machine-learning model or Kaggle competition improvement.

## Evidence

- [Frontier list — Actions #42869895507](https://github.com/kolihhan/name-kaggle/actions/runs/42869895507)
- [Static source & input manifests — Actions #42870239492](https://github.com/kolihhan/name-kaggle/actions/runs/42870239492)
- [Same-index code-cell comparison & preview outputs — Actions #42870549318](https://github.com/kolihhan/name-kaggle/actions/runs/42870549318)

**Kaggle score-data limitation:** `kaggle kernels list --competition name-that-llm --sort-by scoreDescending -v` returns `ref,title,author,lastRunTime,totalVotes`, but no `score` or `publicScore` field. The authenticated Python `ApiKernelMetadata` model also provides no numeric score. Treat 0.683/0.682 as authors' **title claims**, not independently authenticated official scores. Sorting by score alone does not supply a numeric official score.

## Candidate findings

| Public notebook | Title-claimed score | Latest upstream status | Code cells | Source SHA256 | Kaggle preview submission.parquet |
| --- | ---: | --- | ---: | --- | --- |
| `ozertuu/0-683-0-682-qwen3-8-27b-gsq-15-mix-0-52` | 0.683 | COMPLETE | 40 | `14d48eb60e4201cd749e0db7f6b9fae89796d4cbb10a4f2f7006300f267535c6` | listed |
| `alexchilton/name-that-llm-0-682-all-public-what-failed` | 0.682 | COMPLETE | 42 | `1a2e9346758ee05c389d75d07b2fe2cbef6c8dfb14e9eee26a36eaba453c945d` | listed |
| `omerfarukmerey/0-682-0-681-qwen3-8-27b-gsq-12-mix-0-50` | 0.682 | COMPLETE | 40 | `8ae25e1e848adbeeba2adcefb8548629131090ba4123e882aeb730911c0da4d7` | NOT listed by public CLI `kernels files -v` |

All three public notebook metadata instances specify `enable_internet=false`, `machine_shape=NvidiaTeslaT4`, `competition_sources=[name-that-llm]`, 20 dataset, 8 linked Notebook and 6 model inputs. This establishes **source access and static configuration only**, not permission/licensing of every source, scoring license, runtime memory, 4,000-row hidden re-execution, or valid competition submission.

Ozertuu and Omer share **36/40 exactly byte-identical code cells at identical indices** (four changed cells at 3, 33, 36, 39). This is strong evidence that they share the same basic architecture, NOT independent competing systems. Alex uses 42 code cells and cannot be aligned by same-index hashing; shared dependencies alone do not prove the same code.

Unexecuted source contains `.fit` and `.train` textual markers (one each), no `.backward` marker. Presence is not evidence that supervised fitting occurs during scoring; absence of `.backward` alone does not establish no-training. Require active-path audit and fail-closed runtime guards before considering reproduction.

## Decision

1. **Priority public reproduction candidate: Ozertuu 0.683**, only conditional on actual upstream license, independent official Public Score, active no-supervised-training path and offline hidden-sized runtime feasibility. Do not claim it is a breakthrough or our own 0.683.
2. **Hold Omer** because current preview file list did not contain `submission.parquet`; this could be output naming or upload/retention issue, not proof it cannot work.
3. **Hold Alex** pending direct source comparison and official score verification.
4. Existing `58167751` Lightning v2 submission had last authenticated status **PENDING and blank Public Score** in [status workflow #42618511398 attempt 11](https://github.com/kolihhan/name-kaggle/actions/runs/42618511398). This document does not claim a fresh status query.
5. Do not create repetitive leaderboard-tuning submissions or relax no-training requirements. No GPU or new competition submission was used in these audits.

## Public repository safety

[PUBLIC_VISIBILITY.md](../../PUBLIC_VISIBILITY.md) explicitly records user authorization (2026-11-04) to continue data-safe GitHub Actions while public. Do not require a visibility flip for routine approved workflows. However historical official competition data is publicly tracked and not proven sanitized; never commit additional competition text/data or credentials, and do not weaken guard workflows.

## Actual execution

- Created and iterated one reusable read-only `.github/workflows/ntl-public-frontier-notebooks-readonly.yml` on the research branch.
- Three complete source manifests, preview file-list checks and one pairwise code-cell comparison executed successfully.
- No model adaptation, Kaggle submission, GPU training, or artificial result promotion.
