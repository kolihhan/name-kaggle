# Name That LLM — Kaggle

Working repository for the Kaggle competition **Name That LLM**: identify which of 18 language models generated each passage.

Competition: https://www.kaggle.com/competitions/name-that-llm

## Current state

### Phase 1 — frozen validation contract

- Official training data: **2,699 rows / 150 prompts / 18 models**.
- Validation: deterministic **5-fold `GroupKFold(prompt_id)`**.
- Prompt overlap between train and validation: **0 in every fold**.
- Metric: **balanced accuracy**.
- Phase 1 char TF-IDF + balanced LinearSVC anchor: **0.31930 OOF**.
- Permanent fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`.

Do not replace these folds with a different grouped split just to match an external CV number.

### Phase 2 — audited external data

The first admitted external source is a 2,503-row GPT/Gemini snapshot:

- 430 independent prompts;
- 8 exact competition labels;
- 0 duplicate normalized texts;
- 0 exact overlap with official train;
- max word TF-IDF similarity to official train: 0.166;
- provenance `model_seen` blank rows: 0.

On the Phase 1 sparse model it improved honest official OOF:

| Experiment | Balanced accuracy | Δ |
|---|---:|---:|
| Official only | 0.31929 | — |
| Official + 2,503 external | **0.34116** | **+0.02187** |
| Official + balanced 1,120-row slice | 0.33597 | +0.01668 |

This authorizes representation-specific A/B testing, not unconditional use of the source.

### Phase 3 — frozen ModernBERT gate

| Condition | Balanced accuracy | Δ |
|---|---:|---:|
| Official only frozen ModernBERT | **0.30224** | — |
| Frozen ModernBERT + 2,503 external | **0.29041** | **-0.01183** |

**Decision:** reject the external source for the encoder path. Recall fell for all 8 directly covered labels.

### Phase 4 — official-only trainable ModernBERT

| Model / representation | OOF balanced accuracy |
|---|---:|
| Phase 1 char TF-IDF | 0.31930 |
| Phase 3 frozen ModernBERT | 0.30224 |
| Phase 4 trainable ModernBERT | **0.25781** |

The five-fold Kaggle GPU run passed official-data SHA, fold fingerprint, row-count, label-count and no-external-data checks.

**Decision:** reject this full-fine-tuning recipe. Do not respond with a blind epoch / learning-rate sweep.

### Phase 5 — style multiview baseline

Official-only, fold-local three-view baseline:

| View | OOF balanced accuracy |
|---|---:|
| Char TF-IDF + LR | 0.32967 |
| Word TF-IDF + LR | 0.28669 |
| Numeric style + LR | 0.29008 |
| Equal geometric-mean blend | **0.34821** |

**Decision:** the views are complementary, but equal probability blending leaves performance on the table.

### Phase 5b — strong sparse anchor

The same signal families were combined into one fold-local feature union and classified with a fixed balanced LinearSVC:

- char TF-IDF 1–5, max 180k;
- word TF-IDF 1–2, max 150k;
- boundary `char_wb` 3–5, max 100k;
- punctuation-aware token TF-IDF 1–2, max 100k;
- deterministic style branch × 0.2;
- `LinearSVC(C=1.5, class_weight="balanced")`.

Result:

- OOF balanced accuracy: **0.36709**
- folds: `0.38333 / 0.34444 / 0.37593 / 0.38148 / 0.35032`
- Δ vs Phase 5 blend: **+0.01888**
- Δ vs Phase 1: **+0.04779**

**Decision: this is the current canonical CPU anchor.**

### Phase 5c — reference-style reproduction

To explain the apparent gap to a public `0.37560` result, only the style extractor was replaced with the narrower 108-feature public RichStyle definition; every sparse branch, `C=1.5`, style weight and permanent fold remained fixed.

- Phase 5b: `0.3670892`
- Phase 5c: `0.3670967`
- Δ: **+0.0000075**

The style extractor does not explain the gap. The public repository evaluates with shuffled `StratifiedGroupKFold`, while this repo intentionally preserves the deterministic Phase 1 `GroupKFold(prompt_id)` mapping. Those CV numbers are therefore not apples-to-apples.

**Decision:** keep Phase 5b as the canonical CPU anchor and stop sparse micro-tuning on official-only data.

### Maniceet 7,753-row source

`maniceet/name-that-llm-openrouter-openai-generations` remains blocked on resource-specific Kaggle dataset access/consent. The account token is valid and can access the competition API, but this resource returns HTTP 403. It has not passed our raw-row audit and must not be treated as verified training data.

## Canonical workflow

The repo source of truth is reusable code plus committed evidence, not inline experiment code hidden inside CI YAML.

```text
data/                    Official competition data and audited external snapshots
scripts/                 Reproducible audit / fold / experiment code
artifacts/phase1/        Frozen validation audit and baseline evidence
artifacts/phase2/        External-data reconnaissance and ablations
artifacts/phase3/        Frozen encoder gate
artifacts/phase4/        Trainable encoder baseline
artifacts/phase5/        Multiview LR baseline
artifacts/phase5b/       Canonical strong-sparse CPU anchor
artifacts/phase5c/       Reference-style reproduction check
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
- `scripts/run_reference_style_repro.py` — reproduce Phase 5c.

## Validation contract

Do **not** use a random row split. Rows sharing a `prompt_id` must stay in the same permanent fold. The hidden test uses unseen prompts, so prompt-grouped CV is the local source of truth.

External data is admitted stage by stage. Validation rows are always official rows only; external rows never enter the metric denominator.

## Next experiment — Phase 6

Run one direct A/B on the **canonical strong-sparse anchor** under the exact same permanent folds:

1. A: official fold-training rows only;
2. B: the same official fold-training rows + all 2,503 audited GPT/Gemini external rows;
3. fit each condition's TF-IDF vocabularies and style scaler independently inside that fold's training data;
4. score only the untouched official validation fold;
5. report aggregate BA, every fold, per-model recall delta, and covered-vs-uncovered label effects.

The question is narrow: **does the external source still help once the official-only sparse baseline is strong?**

Do not add the blocked Maniceet source or return to neural tuning before this A/B is known.

## Data attribution

The competition Data page lists the official dataset under **CC BY-SA 4.0**. See `data/README.md` for source and attribution. External source attribution and provenance are recorded in `artifacts/phase2/`.
