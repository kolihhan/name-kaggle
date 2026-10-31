# Name That LLM — Kaggle

Working repository for the Kaggle competition **Name That LLM**: identify which of 18 language models generated each passage.

Competition: https://www.kaggle.com/competitions/name-that-llm

## Current state

### Permanent validation contract

- Official training data: **2,699 rows / 150 prompts / 18 models**.
- Validation: deterministic **5-fold `GroupKFold(prompt_id)`**.
- Metric: **balanced accuracy**.
- Prompt overlap between train and validation: **0 in every fold**.
- Official train SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- Fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.

Do **not** use random row splits. The competition test contains unseen prompts; prompt-grouped CV is the local source of truth.

### Phase 1 — char baseline

Char TF-IDF + balanced LinearSVC: **0.31930 OOF**.

### Phase 2 — first external-data ablation

The audited 2,503-row GPT/Gemini source helped the Phase 1 sparse model:

| Experiment | Balanced accuracy | Δ |
|---|---:|---:|
| Official only | 0.31929 | — |
| + all 2,503 external | **0.34116** | **+0.02187** |
| + balanced 1,120-row slice | 0.33597 | +0.01668 |

This was representation-specific evidence, not general authorization to use the source everywhere.

### Phase 3 — frozen ModernBERT representation gate

| Condition | Balanced accuracy | Δ |
|---|---:|---:|
| Official only | **0.30224** | — |
| + 2,503 external | **0.29041** | **-0.01183** |

The external source failed this gate; recall fell for all 8 labels it directly covers.

### Phase 4 — trainable ModernBERT

Official-only end-to-end fine-tuning on Kaggle GPU: **0.25781 OOF**.

The run passed all data/fold checks and completed without NaN, OOM, runtime failure, or prediction collapse. The configuration was rejected rather than expanded into a blind hyperparameter sweep.

### Phase 5 — honest multiview style baseline

A fixed official-only CPU candidate combines:

- case-preserving char TF-IDF 1–5;
- word TF-IDF 1–2;
- boundary `char_wb` 3–5;
- word-or-punctuation TF-IDF 1–2;
- 142 deterministic style / typography features;
- balanced `LinearSVC(C=1.5)` with style weight 0.2.

The exact Phase 1 sanity baseline reproduced to within `7.46e-06` before the multiview result was accepted.

| Candidate | OOF balanced accuracy | Δ vs Phase 1 |
|---|---:|---:|
| Phase 1 sanity | 0.31929 | — |
| **Phase 5 multiview style SVM** | **0.36004** | **+0.04074** |

Fold scores: `0.36852 / 0.33148 / 0.36481 / 0.38148 / 0.35390`.

Recall improved for 13 / 18 labels. **Phase 5 is now the canonical CPU anchor.**

See:

- `artifacts/phase5/CONTRACT.md`
- `artifacts/phase5/honest_style_baseline.json`
- `artifacts/phase5/RESULT.md`
- `scripts/run_honest_style_baseline.py`

The GitHub Actions artifact retains the full OOF CSV.

### Maniceet 7,753-row source

`maniceet/name-that-llm-openrouter-openai-generations` remains blocked on resource-specific Kaggle dataset access/consent. It has not passed our raw-row audit and must not be treated as verified training data.

## Canonical workflow

```text
data/                    Official competition data and audited external snapshots
scripts/                 Reproducible audit / fold / experiment code
artifacts/phase1/        Initial validation and baseline evidence
artifacts/phase2/        External-data audits / sparse ablations
artifacts/phase3/        Frozen encoder gate
artifacts/phase4/        Trainable encoder baseline
artifacts/phase5/        Multiview style CPU anchor
.github/workflows/       Thin execution / verification wrappers
```

Key scripts:

- `scripts/bootstrap_data.sh`
- `scripts/make_folds.py`
- `scripts/audit_gpt_gemini_external.py`
- `scripts/run_external_ablation.py`
- `scripts/run_frozen_encoder_ablation.py`
- `scripts/train_modernbert_cv.py`
- `scripts/run_honest_style_baseline.py`

## Next experiment

Re-test the already-audited 2,503-row GPT/Gemini external source on the **new 0.36004 multiview anchor**:

1. A — official-only multiview anchor;
2. B — official folds + all 2,503 external rows;
3. C — official folds + the class-balanced external slice;
4. validation remains official rows only on the permanent folds;
5. compare aggregate and per-model deltas, especially the 8 directly augmented labels.

Only after this stronger sparse-data gate should we decide whether to package a CPU Kaggle submission or move compute to LoRA / larger attribution models.

## Data attribution

The competition Data page lists the official dataset under **CC BY-SA 4.0**. See `data/README.md` for source and attribution. External source attribution and provenance are recorded in `artifacts/phase2/`.
