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

A fixed fold-local three-view baseline was evaluated without external data:

| View | OOF balanced accuracy |
|---|---:|
| Char TF-IDF + LogisticRegression | 0.32967 |
| Word TF-IDF + LogisticRegression | 0.28669 |
| Numeric style + LogisticRegression | 0.29008 |
| Equal geometric-mean blend | **0.34821** |

Blend fold scores: `0.34444 / 0.32037 / 0.37037 / 0.35741 / 0.34860`.

The blend improves Phase 1 by **+0.02891** and improves recall over the char-only view on 12 of 18 labels. It still trails independently observed public prompt-grouped sparse references around `0.374–0.376`.

**Decision:** keep the multiview/style feature families, but do not keep the equal three-way LR blend as the final CPU anchor. The next test should change only the combination/classifier strategy.

See:

- `artifacts/phase5/CONTRACT.md`
- `artifacts/phase5/style_baseline.json`
- `artifacts/phase5/RESULT.md`

### Maniceet 7,753-row source

`maniceet/name-that-llm-openrouter-openai-generations` remains blocked on resource-specific Kaggle dataset access/consent. It has not passed our raw-row audit and must not be treated as verified training data.

## Canonical workflow

The repo source of truth is reusable code plus committed evidence, not inline experiment code hidden inside CI YAML.

```text
data/                    Official competition data and audited external snapshots
scripts/                 Reproducible audit / fold / experiment code
artifacts/phase1/        Frozen validation audit and baseline evidence
artifacts/phase2/        External-data reconnaissance and ablations
artifacts/phase3/        Frozen encoder contract, result, and decision
artifacts/phase4/        Trainable encoder contract, result, and decision
artifacts/phase5/        Style/multiview contract, result, and decision
.github/workflows/       Execution wrappers plus historical diagnostics
```

Key scripts:

- `scripts/bootstrap_data.sh` — restore official competition data.
- `scripts/make_folds.py` — recreate the permanent grouped folds.
- `scripts/audit_gpt_gemini_external.py` — reproduce the 2,503-row external audit.
- `scripts/run_external_ablation.py` — reproduce the sparse external-data ablation.
- `scripts/run_frozen_encoder_ablation.py` — reproduce Phase 3.
- `scripts/train_modernbert_cv.py` — reproduce Phase 4.
- `scripts/run_style_baseline.py` — reproduce Phase 5.

## Validation contract

Do **not** use a random row split. Rows sharing a `prompt_id` must stay in the same fold. The competition test contains unseen prompts, so prompt-grouped CV is the local source of truth.

External data is admitted stage by stage. An uplift under one representation does not automatically authorize the same source for another representation.

## Next experiment

Run one **official-only strong sparse anchor** under the same permanent folds, not a hyperparameter grid:

1. char TF-IDF view;
2. word TF-IDF view;
3. boundary-char and punctuation-aware views;
4. deterministic style features with a smaller contribution;
5. one balanced `LinearSVC` over the fold-local feature union;
6. fixed prior-art-backed configuration, then compare honest OOF against Phase 5 `0.34821` and the public ~`0.374–0.376` reference range.

Only after this official-only sparse anchor is established should external data or neural ensembling return to the main line.

## Data attribution

The competition Data page lists the official dataset under **CC BY-SA 4.0**. See `data/README.md` for source and attribution. External source attribution and provenance are recorded in `artifacts/phase2/`.
