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

### Phase 5b — canonical official-only strong sparse anchor

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

**Decision:** keep **0.36709** as the canonical official-only base anchor.

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

### Phase 7 — gated 8-class external specialist

Phase 7 isolates the same audited external corpus in a separate 8-class specialist. The official-only Phase 5b anchor makes the covered-vs-uncovered decision first; the specialist may replace the exact label only when the anchor already predicts one of the 8 externally covered GPT/Gemini labels. No threshold or blend weight is tuned.

| Condition | OOF balanced accuracy | Δ vs Phase 5b |
|---|---:|---:|
| Reproduced Phase 5b anchor | **0.367089** | — |
| **Phase 7 gated specialist** | **0.370052** | **+0.002963** |

The structural gate behaved as intended:

- covered 8-label mean recall: `0.40833 → 0.41500` (**+0.00667**)
- uncovered 10-label mean recall: `0.33409 → 0.33409` (**0.00000**)
- all 10 uncovered per-model recall deltas: exactly **0**
- anchor predicted uncovered on 1,470 rows; all 1,470 remained untouched
- specialist changed 454 rows, correcting 110 anchor mistakes while breaking 102 anchor-correct rows: net **+8** correct rows

**Decision:** promote Phase 7 under the frozen contract. **0.370052 is the current best CPU OOF system**, while Phase 5b remains the canonical official-only base anchor.

Do not tune the gate, covered-label set, external sample count, class weights, or SVC C on these permanent folds. The gain is valid but modest; the next useful experiment must add genuinely different information or representation.

See:

- `artifacts/phase7/CONTRACT.md`
- `artifacts/phase7/gated_specialist.json`
- `artifacts/phase7/RESULT.md`

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
artifacts/phase5b/       Canonical official-only strong sparse anchor
artifacts/phase6/        Global strong-sparse external transfer rejection
artifacts/phase7/        Current best gated external-specialist CPU system
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
- `scripts/run_gated_specialist.py` — reproduce Phase 7.

## Validation contract

Do **not** use a random row split. Rows sharing a `prompt_id` must stay in the same fold. The competition test contains unseen prompts, so prompt-grouped CV is the local source of truth.

External data is admitted stage by stage. An uplift under one representation or weak baseline does not automatically authorize the same source elsewhere.

## Next experiment

Freeze Phase 7 at **0.370052**. Do not squeeze the same gate with thresholds or weight sweeps.

The next experiment should pursue a genuinely different source of signal, in this order:

1. newly audited external generations that cover the currently uncovered/weak Claude, GLM, Kimi, Qwen, and Gemini-3.7 labels; or
2. a materially different representation/model family whose errors are demonstrably complementary to Phase 7 under the permanent folds.

The blocked Maniceet source stays out until its raw rows and provenance can be audited directly.

## Data attribution

The competition Data page lists the official dataset under **CC BY-SA 4.0**. See `data/README.md` for source and attribution. External source attribution and provenance are recorded in `artifacts/phase2/`.
