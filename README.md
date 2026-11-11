# Name That LLM — Kaggle

## NEW verified Kaggle competition submission — WeMM `58244483` (score pending)

**Authenticated one-shot Kaggle entry:** `58244483`, submitted **2026-11-11 16:38:25 UTC / 2026-11-12 00:38 MYT**, status `PENDING`, no Public/Private score yet. [Independent official ledger #43060338073](https://github.com/kolihhan/name-kaggle/actions/runs/43060338073) and [complete preflight + private Notebook + competition submission #43060048662](https://github.com/kolihhan/name-kaggle/actions/runs/43060048662). The private Notebook preview is COMPLETE, but hidden official evaluation has not yet scored.

**Candidate:** frozen public Ozertuu dual-embedding + Tencent WeMM 4B representation with an explicit no-new-supervised-training sibling-style ablation. Exact executable source hash, preexisting three training guards, and the removal of two supervised LogisticRegression fit sites were verified. This is one user-authorized exploratory competition test, not a claim of transfer, independent model development, or an exact reproduction of the original public Notebook. [Detailed research provenance and status](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/artifacts/candidates/dual-embedding-wemm-frozen-v1/SUBMISSION_20261112.md).

**Best verified Public BA remains `0.679` (`58210835`, COMPLETE)** until an authenticated numeric score for `58244483` is published. MGT-v1 `58233227` still PENDING on the latest check. No repeat submissions or leaderboard-driven retuning.

---

## NEW official submission — MGT-v1 `58233227` (score pending)

**Authenticated Kaggle acceptance:** `58233227` submitted at **2026-11-11 09:44:14 UTC**, status **`PENDING`**, no publicScore yet, verified by [Kaggle ledger and private Notebook snapshot #42998099464](https://github.com/kolihhan/name-kaggle/actions/runs/42998099464). Private Notebook `kolihhan/ntl-margin-gated-transport-v1` preview **COMPLETE**; this does **not** prove hidden inference complete.

This is an **exploratory, no-new-training inference change**, not a second copy of the same champion and not a new pretrained model: one continuous, label-agnostic confidence gate shrinks the existing Sinkhorn category-prior correction on high-margin rows, retaining original pretrained weights, all source inputs and three hard no-training guards. Strict synthetic and source-hash preflight succeeded. The user expressly authorized one new official submission despite the missing independent full-18 held-out benchmark. [Frozen algorithm contract](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/artifacts/candidates/margin-gated-transport-v1/CONTRACT.md) · [Exact submission report](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/artifacts/candidates/margin-gated-transport-v1/SUBMISSION_20261111.md).

**Best VERIFIED Public BA remains 0.679** (`58210835`). Do not infer new MGT-v1 Public BA, Private BA or generalization until Kaggle publishes them. Do not retune or resubmit MGT-v1 based on the Public leaderboard.

---

## Latest verified Kaggle Public leaderboard result — 0.679

**2026-11-11 — Our Kaggle submission `58210835`: COMPLETE, Public Balanced Accuracy 0.679.** This improves the previously verified best of 0.668 (`58022340` / `58045266`) by **+0.011** (1.1 percentage points).

This is a hash-pinned, no-new-supervised-training **reproduction of an existing public Ozertuu Notebook**, not an independently developed model. The author's 0.683 title is not our score. **No Private Score has been published.**

The other reproduction, `58167751`, was still PENDING at the 2026-11-11 10:28 MYT official check. No duplicate submissions or leaderboard-driven retuning.

- [Official Kaggle status evidence — Actions #42874538415, attempt 2](https://github.com/kolihhan/name-kaggle/actions/runs/42874538415)
- [Full score, provenance and safety report (research branch)](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/artifacts/candidates/public-ozertuu-0683-guarded-repro/OFFICIAL_SCORE_20261111.md)
- [Research status and original-method development (research branch)](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/NOW.md)

---

## Research evidence gate — 2026-11-11

**Latest nine-source evidence (2026-11-11):** [authenticated read-only scan #42993607587](https://github.com/kolihhan/name-kaggle/actions/runs/42993607587) qualified **0/9 public candidates** for a new, unseen, source/served-model-qualified 18-label holdout. `prvsiyan` has 18-label 300-prompt blocks but is a **direct incumbent input**; `lxdlxdlxd` has strong per-call provenance but only **13/18** labels; `crtproduction` merged 18-class source's latest file/version metadata remains unverified. [Detailed matrix](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/artifacts/research/ntl-18class-independent-validation-gate-20261111.md) and [Issue #1](https://github.com/kolihhan/name-kaggle/issues/1). No new GPU, training or competition submission.

The best **authenticated Kaggle Public** score is still **0.679** (submission `58210835`, attributed public-source reproduction). The private score and a new original-method gain are unverified.

The no-training **Kimi K3 protocol study completed**, not running: high `42/80` versus off `23/80` correctly identified, a +23.75-percentage-point matched difference, but **exact p = 0.010973563 did not pass the original p < 0.01 gate**. It is a diagnostic, not a new classifier or Kaggle competition submission.

**Active research blocker:** no verified fresh, source/prompt-independent, protocol-documented **18-label** same-row validation benchmark against the 0.679 champion. Do not interpret repeatedly examined 12-label cohorts, historical CV splits, or Public LB scores as independent transfer proof. No new supervised training, blend tuning or model submission is authorized by this note.

[Full independent evidence gate](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/artifacts/research/ntl-18class-independent-validation-gate-20261111.md) · [Kimi final result](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/artifacts/research/ntl-kimi-protocol80/RESULT.md) · [Data-qualification issue #1](https://github.com/kolihhan/name-kaggle/issues/1) · [Research-branch evaluation policy](https://github.com/kolihhan/name-kaggle/blob/research/attribution-bottleneck-review-20261106/EVALUATION_POLICY.md)

---

Working repository for the Kaggle competition **Name That LLM**: identify which of 18 language models generated each passage.

Competition: https://www.kaggle.com/competitions/name-that-llm

## Current state

### Phase 1 — frozen validation contract

- Official training data: **2,699 rows / 150 prompts / 18 models**.
- Validation: deterministic **5-fold `GroupKFold(prompt_id)`**.
- Prompt overlap between train and validation: **0 in every fold**.
- Metric: **balanced accuracy**.
- Best classical baseline: character TF-IDF (1–5 grams) + balanced LinearSVC, OOF **0.31930**.
- Hierarchical lab→model classification did not materially help.

The permanent fold mapping is regenerated by `scripts/make_folds.py`. It was verified against the original Phase 1 mapping: **2,699 / 2,699 rows identical**.

### Phase 2 — external data

The first usable external source has now passed both the raw-data gate and the frozen-CV ablation:

- Source snapshot: GPT + Gemini generations from `RohanExploit/kaggle-name-that-llm-top20`.
- Raw audit: **2,503 rows / 430 independent prompts / 8 exact competition labels**.
- Duplicate normalized texts: **0**.
- Exact overlap with official train: **0**.
- Maximum TF-IDF word 1–2 similarity to an official train passage: **0.166**.
- `model_seen` provenance blank rows: **0**.
- 40–250 word protocol coverage: **98.4%**.

Frozen-CV ablation, using the same Phase 1 folds:

| Experiment | Balanced accuracy | Δ vs official-only |
|---|---:|---:|
| Official only | 0.31929 | — |
| Official + all 2,503 external rows | **0.34116** | **+0.02187** |
| Official + class-balanced 1,120-row slice | 0.33597 | +0.01668 |

**Decision:** promote the audited 2,503-row source into the first encoder-stage experiment. The sparse-model result is an admission gate, not evidence that TF-IDF is the final modeling direction.

### Maniceet 7,753-row source

`maniceet/name-that-llm-openrouter-openai-generations` remains promising from its public metadata, but it has **not** passed our raw-row audit yet.

Verified access state:

- `KAGGLE_API_TOKEN` authentication succeeds;
- the same token can access the official `name-that-llm` competition API;
- Kaggle dataset API calls for this specific resource return HTTP 403 from both Kaggle CLI and `kagglehub`.

Treat the 7,753-row source as **blocked on resource-specific dataset access/consent**, not as verified training data. Do not use its reported row-level claims as if we had independently inspected the raw file.

## Canonical workflow

The repo source of truth is the reusable code plus committed evidence, not inline experiment code hidden inside CI YAML.

```text
data/                    Official competition data and audited external snapshots
scripts/                 Reproducible audit / fold / experiment code
artifacts/phase1/        Frozen validation audit and baseline evidence
artifacts/phase2/        External-data reconnaissance, audits, ablations, access state
.github/workflows/       Execution wrappers plus a few one-shot historical diagnostics
```

The reusable Phase 2 audit and ablation paths are now implemented in `scripts/`. The Kaggle access diagnostic workflows are historical one-shot tools and are **not** canonical experiment logic.

Key scripts:

- `scripts/bootstrap_data.sh` — restore official competition data.
- `scripts/make_folds.py` — recreate the frozen grouped folds.
- `scripts/audit_gpt_gemini_external.py` — reproduce the 2,503-row external raw audit.
- `scripts/run_external_ablation.py` — reproduce official-only vs external-data frozen-CV ablation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Recreate the exact Phase 1 folds
python scripts/make_folds.py

# Re-run the frozen-CV external-data ablation
python scripts/run_external_ablation.py
```

If official data is missing, use `scripts/bootstrap_data.sh /path/to/name-that-llm.zip`.

## Validation contract

Do **not** use a random row split. Rows sharing a `prompt_id` must stay in the same fold. The hidden Kaggle test uses unseen prompts, so prompt-grouped CV is the local source of truth.

External data is admitted only after it improves this same frozen grouped OOF evaluation. Data derived from official train prompts must be fold-isolated or excluded during model selection to avoid prompt-semantic leakage.

## Next experiment

Run the **first encoder baseline** under the same frozen folds:

1. encoder trained on official fold-training rows only;
2. same encoder/config trained on official fold-training rows + the promoted 2,503-row external source;
3. compare OOF balanced accuracy and per-model recall on the untouched official validation rows;
4. only then decide whether encoder tuning, LoRA, or another external source is worth the next compute budget.

Do not wait for the blocked 7,753-row Maniceet source before starting this comparison.

## Data attribution

The competition Data page lists the official dataset under **CC BY-SA 4.0**. See `data/README.md` for source and attribution. External source attribution and provenance are recorded in `artifacts/phase2/`.
