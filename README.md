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

The audited 2,503-row GPT/Gemini source improved the weak sparse baseline:

| Experiment | Balanced accuracy | Δ |
|---|---:|---:|
| Official only | 0.31929 | — |
| Official + 2,503 external | **0.34116** | **+0.02187** |
| Official + balanced 1,120-row slice | 0.33597 | +0.01668 |

This was an admission signal for representation-matched testing, not proof that the source helps every model.

### Phase 3 — frozen ModernBERT gate

| Condition | Balanced accuracy | Δ |
|---|---:|---:|
| Official only frozen ModernBERT | **0.30224** | — |
| Frozen ModernBERT + 2,503 external | **0.29041** | **-0.01183** |

**Decision:** the external source fails the encoder-stage admission gate.

### Phase 4 — official-only trainable ModernBERT

| Model / representation | OOF balanced accuracy |
|---|---:|
| Phase 1 char TF-IDF | **0.31930** |
| Phase 3 frozen ModernBERT | **0.30224** |
| Phase 4 trainable ModernBERT | **0.25781** |

**Decision:** reject this full-fine-tuning recipe. Do not respond with a blind hyperparameter sweep.

### Phase 5 — official-only style multiview

| View | OOF balanced accuracy |
|---|---:|
| Char TF-IDF + LogisticRegression | 0.32967 |
| Word TF-IDF + LogisticRegression | 0.28669 |
| Numeric style + LogisticRegression | 0.29008 |
| Equal geometric-mean blend | **0.34821** |

The blend improved Phase 1 by **+0.02891**.

### Phase 5b — canonical strong sparse CPU anchor

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

**Decision:** keep **0.36709** as the canonical CPU anchor.

Do not tune toward the independently observed public `0.37560` as if it were a same-fold target. That public result used `StratifiedGroupKFold(shuffle=True, random_state=42)` with a different fold mapping; this repo keeps its permanent `GroupKFold(prompt_id)` mapping unchanged.

### Phase 6 — strong sparse + audited external

The exact Phase 5b model was retrained with all 2,503 audited GPT/Gemini external rows added to every official training partition. Validation remained untouched official rows only.

| Condition | OOF balanced accuracy | Δ vs Phase 5b |
|---|---:|---:|
| Official-only Phase 5b | **0.36709** | — |
| Phase 6 + 2,503 external | 0.35967 | **-0.00742** |

The result is asymmetric:

- externally covered 8-label mean recall: `0.40833 → 0.50917` (**+0.10083**)
- uncovered 10-label mean recall: `0.33409 → 0.24008` (**-0.09402**)

**Decision:** reject global external augmentation for the strong-sparse path. The corpus contains useful specialist signal for represented GPT/Gemini classes, but global inclusion hurts uncovered classes enough to reduce balanced accuracy.

See:

- `artifacts/phase6/CONTRACT.md`
- `artifacts/phase6/strong_sparse_external.json`
- `artifacts/phase6/RESULT.md`

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
artifacts/phase6/        Strong-sparse external-transfer test and rejection
.github/workflows/       Execution wrappers plus historical diagnostics
```

Key scripts:

- `scripts/bootstrap_data.sh` — restore official competition data.
- `scripts/make_folds.py` — recreate the permanent grouped folds.
- `scripts/audit_gpt_gemini_external.py` — reproduce the 2,503-row external audit.
- `scripts/run_external_ablation.py` — reproduce Phase 2.
- `scripts/run_frozen_encoder_ablation.py` — reproduce Phase 3.
- `scripts/train_modernbert_cv.py` — reproduce Phase 4.
- `scripts/run_style_baseline.py` — reproduce Phase 5.
- `scripts/run_strong_sparse_anchor.py` — reproduce Phase 5b.
- `scripts/run_strong_sparse_external.py` — reproduce Phase 6.

## Validation contract

Do **not** use a random row split. Rows sharing a `prompt_id` must stay in the same fold. The competition test contains unseen prompts, so prompt-grouped CV is the local source of truth.

External data is admitted stage by stage. An uplift under one representation or weak baseline does not automatically authorize the same source elsewhere.

## Next experiment

Do **not** globally add the 2,503-row GPT/Gemini corpus again, and do not tune its sample count or class weights on these permanent folds.

If this source is revisited, the next defensible hypothesis is a **separate external-trained specialist** that leaves the official-only Phase 5b anchor unchanged and is combined only through a predeclared, untuned mechanism. Before building that, first compare its expected value against higher-diversity directions such as a genuinely different model family or newly audited external coverage for the currently weak/uncovered classes.

## Data attribution

The competition Data page lists the official dataset under **CC BY-SA 4.0**. See `data/README.md` for source and attribution. External source attribution and provenance are recorded in `artifacts/phase2/`.
