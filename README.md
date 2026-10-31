# Name That LLM — Kaggle

Working repository for the Kaggle competition **Name That LLM**: identify which of 18 language models generated each passage.

Competition: https://www.kaggle.com/competitions/name-that-llm

## Current state

### Phase 1 — frozen validation contract

- Official training data: **2,699 rows / 150 prompts / 18 models**.
- Validation: deterministic **5-fold `GroupKFold(prompt_id)`**.
- Prompt overlap between train and validation: **0 in every fold**.
- Metric: **balanced accuracy**.
- Best Phase 1 classical baseline: char TF-IDF (1–5) + balanced LinearSVC, OOF **0.31930**.
- Permanent fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.

### Phase 2 — external data

The audited 2,503-row GPT/Gemini source improved the sparse baseline:

| Experiment | Balanced accuracy | Δ |
|---|---:|---:|
| Official only | 0.31929 | — |
| Official + 2,503 external | **0.34116** | **+0.02187** |
| Official + balanced 1,120-row slice | 0.33597 | +0.01668 |

This was an admission signal for further testing, not proof that the source helps every representation.

### Phase 3 — frozen ModernBERT gate

| Condition | Balanced accuracy | Δ |
|---|---:|---:|
| Official only frozen ModernBERT | **0.30224** | — |
| Frozen ModernBERT + 2,503 external | **0.29041** | **-0.01183** |

**Decision:** the 2,503-row source fails the encoder-stage admission gate. Recall fell for all 8 labels directly covered by the external source.

### Phase 4 — official-only trainable ModernBERT

| Model / representation | OOF balanced accuracy |
|---|---:|
| Phase 1 char TF-IDF | **0.31930** |
| Phase 3 frozen ModernBERT | **0.30224** |
| Phase 4 trainable ModernBERT | **0.25781** |

The five-fold Kaggle GPU run passed the official-data SHA, fold fingerprint, row-count, label-count, and no-external-data checks.

**Decision:** reject this full-fine-tuning recipe. Do not respond with a blind epoch or hyperparameter sweep.

### Phase 5 — official-only style multiview

| View | OOF balanced accuracy |
|---|---:|
| Char TF-IDF + LogisticRegression | 0.32967 |
| Word TF-IDF + LogisticRegression | 0.28669 |
| Numeric style + LogisticRegression | 0.29008 |
| Equal geometric-mean blend | **0.34821** |

The blend improved Phase 1 by **+0.02891**, but equal weighting gave too much influence to the weaker word/style branches.

### Phase 5b — strong sparse CPU anchor

A single fold-local feature union was evaluated with no external data:

- char TF-IDF 1–5
- word TF-IDF 1–2
- boundary-char TF-IDF 3–5
- punctuation-aware TF-IDF
- deterministic style features at weight `0.2`
- balanced `LinearSVC(C=1.5)`

| Model | OOF balanced accuracy | Δ vs Phase 1 |
|---|---:|---:|
| Phase 5 equal LR blend | 0.34821 | +0.02891 |
| **Phase 5b strong sparse anchor** | **0.36709** | **+0.04779** |

Fold scores: `0.38333 / 0.34444 / 0.37593 / 0.38148 / 0.35032`.

**Decision:** promote **0.36709** as the current canonical CPU anchor.

Do not tune toward the independently observed public `0.37560` as if it were a same-fold target. That public result used `StratifiedGroupKFold(shuffle=True, random_state=42)` with a different fold fingerprint; this repo keeps its permanent `GroupKFold(prompt_id)` mapping unchanged.

See:

- `artifacts/phase5b/CONTRACT.md`
- `artifacts/phase5b/strong_sparse_anchor.json`
- `artifacts/phase5b/RESULT.md`

### Maniceet 7,753-row source

`maniceet/name-that-llm-openrouter-openai-generations` remains blocked on resource-specific Kaggle dataset access/consent. It has not passed our raw-row audit and must not be treated as verified training data.

## Canonical workflow

```text
data/                    Official competition data and audited external snapshots
scripts/                 Reproducible audit / fold / experiment code
artifacts/phase1/        Frozen validation audit and baseline evidence
artifacts/phase2/        External-data reconnaissance and ablations
artifacts/phase3/        Frozen encoder contract, result, and decision
artifacts/phase4/        Trainable encoder contract, result, and decision
artifacts/phase5/        Style/multiview contract, result, and decision
artifacts/phase5b/       Current strong sparse CPU anchor
.github/workflows/       Execution wrappers plus historical diagnostics
```

Key scripts:

- `scripts/bootstrap_data.sh` — restore official competition data.
- `scripts/make_folds.py` — recreate the permanent grouped folds.
- `scripts/audit_gpt_gemini_external.py` — reproduce the 2,503-row external audit.
- `scripts/run_external_ablation.py` — reproduce the Phase 2 sparse external-data ablation.
- `scripts/run_frozen_encoder_ablation.py` — reproduce Phase 3.
- `scripts/train_modernbert_cv.py` — reproduce Phase 4.
- `scripts/run_style_baseline.py` — reproduce Phase 5.
- `scripts/run_strong_sparse_anchor.py` — reproduce Phase 5b.

## Validation contract

Do **not** use a random row split. Rows sharing a `prompt_id` must stay in the same fold. The competition test contains unseen prompts, so prompt-grouped CV is the local source of truth.

External data is admitted stage by stage. An uplift under one representation does not automatically authorize the same source for another representation.

## Next experiment

Run one **strong-sparse + audited external** transfer test under the same permanent folds:

1. keep the exact Phase 5b feature pipeline and `LinearSVC(C=1.5)` unchanged;
2. for each fold, train on the four official training folds plus all admitted 2,503 GPT/Gemini external rows;
3. fit all TF-IDF vocabularies/scalers inside that fold's combined training set;
4. score only untouched official validation rows;
5. compare directly against the official-only Phase 5b anchor **0.36709**;
6. record per-model recall, especially the 8 externally covered labels versus the 10 uncovered labels.

Do not reintroduce the failed ModernBERT external path unless a new representation-specific reason appears.

## Data attribution

The competition Data page lists the official dataset under **CC BY-SA 4.0**. See `data/README.md` for source and attribution. External source attribution and provenance are recorded in `artifacts/phase2/`.
