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

**Decision:** promote Phase 7 under the frozen contract. **0.3700521999 is the current best CPU OOF system**, while Phase 5b remains the canonical official-only base anchor.

Do not tune the gate, covered-label set, external sample count, class weights, or SVC C on these permanent folds. The gain is valid but modest; the next useful experiment must add genuinely different information or representation.

See:

- `artifacts/phase7/CONTRACT.md`
- `artifacts/phase7/gated_specialist.json`
- `artifacts/phase7/RESULT.md`

### Phase 8 — conservative external coverage audit

The previously targeted `maniceet/name-that-llm-openrouter-openai-generations` source was tested through both anonymous and authenticated Kaggle routes and returned HTTP 403. It remains access-blocked, not merely parser-blocked.

A public replacement corpus, `olegzholobov/name-that-llm-response-corpus`, was audited instead. Its frozen archive covers all 18 labels. The short view contains 24,025 mechanically clean response cells with zero exact or normalized-text overlap against the official 2,699 training answers.

A conservative provenance rule was frozen before any model run:

1. source rows are 100% license-known;
2. source rows are 100% output-terms-clear;
3. source rows have 100% exact prompt provenance;
4. the retained source/prompt bank is independently evidenced as public before the competition opened on 2026-10-14;
5. only labels not covered by the Phase 7 specialist are retained.

That produced **2,921 unique responses** across five labels:

| Label | Rows |
|---|---:|
| claude-opus-4-6 | 1,934 |
| glm-5.3 | 200 |
| glm-5.3-flash | 284 |
| kimi-k3 | 317 |
| qwen3.8-max | 186 |

Official train does not expose prompt text, so accidental prompt identity between these public pre-competition prompts and hidden official prompt groups cannot be mathematically ruled out. The defensible statement is narrower: the retained sources/prompt banks predate competition launch and their answer texts have zero exact/normalized overlap with official training answers.

**Decision:** Phase 8 is a reusable provenance/data-audit result, not a promoted model.

See `artifacts/phase8/PHASE8.md` and `artifacts/phase8/response_corpus_audit.json`.

### Phase 9 — pre-competition five-class gated specialist

Phase 9 tested the 2,921-row Phase 8 candidate exactly once under a frozen contract. It stacked a five-class specialist on top of the reproduced **Phase 7 final OOF**. The second gate could only change predictions already inside these five labels:

- `claude-opus-4-6`
- `glm-5.3`
- `glm-5.3-flash`
- `kimi-k3`
- `qwen3.8-max`

No threshold, class-weight, `C`, feature, label-set, row-selection, or gate sweep was performed after seeing the result.

| Condition | OOF balanced accuracy | Δ vs Phase 7 |
|---|---:|---:|
| Reproduced Phase 7 | **0.3700521999** | — |
| Phase 9 gated specialist | 0.3659781258 | **-0.0040740741** |

The safety invariants held:

- all 13 labels outside the five-class specialist retained exactly the same recall;
- Phase 7 predictions outside the gate were never changed;
- 2,699 OOF rows / 2,699 unique IDs / zero nulls;
- gate changed 182 rows, rescuing 27 mistakes and breaking 38 correct Phase 7 predictions: net **-11** correct rows.

Per-label recall delta inside the gate:

- `claude-opus-4-6`: `-0.0200`
- `glm-5.3`: `-0.0267`
- `glm-5.3-flash`: `-0.0133`
- `kimi-k3`: `+0.0200`
- `qwen3.8-max`: `-0.0333`

**Decision: REJECT Phase 9.** Clean, pre-competition external data did not transfer strongly enough under this five-class sparse specialist. Do not rescue this route with post-hoc tuning on the permanent folds. **Phase 7 remains the current best system at 0.3700521999.**

See:

- `artifacts/phase9/CONTRACT.md`
- `artifacts/phase9/precompetition_specialist.json`
- `artifacts/phase9/RESULT.md`

## Canonical workflow

```text
EVALUATION_POLICY.md      Frozen future candidate-selection / transfer rules
data/                     Official competition data and audited external snapshots
scripts/                  Reproducible audit / fold / experiment code
artifacts/phase1/         Frozen validation audit and baseline evidence
artifacts/phase2/         External-data reconnaissance and ablations
artifacts/phase3/         Frozen encoder contract, result, and decision
artifacts/phase4/         Trainable encoder contract, result, and decision
artifacts/phase5/         Style/multiview contract, result, and decision
artifacts/phase5b/        Canonical official-only strong sparse anchor
artifacts/phase6/         Global strong-sparse external transfer rejection
artifacts/phase7/         Current best gated external-specialist CPU system
artifacts/phase8/         External provenance and coverage audit
artifacts/phase9/         Rejected pre-competition five-class specialist
.github/workflows/        Execution wrappers plus historical diagnostics
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
- `scripts/phase8_audit_response_corpus.py` — reproduce the Phase 8 response-corpus audit.
- `scripts/run_phase9_precompetition_specialist.py` — reproduce the frozen Phase 9 negative experiment.

## Validation contract

Do **not** use a random row split. Rows sharing a `prompt_id` must stay in the same fold.

The permanent prompt-grouped 5-fold OOF remains required for historical comparability, but after repeated adaptive use it is **not the sole future promotion oracle**. Phase 19's three alternate assignments are also spent diagnostics; repeatedly minting more splits of the same 2,699 rows does not create a new hidden distribution.

All new candidate selection is governed by `EVALUATION_POLICY.md`: freeze the candidate before evaluation, preserve fold isolation, require one predeclared transfer/robustness check beyond re-splitting official rows, and never use Kaggle Public LB as a tuning loop.

External data is admitted stage by stage. An uplift under one representation or weak baseline does not automatically authorize the same source elsewhere.

## Next experiment

Keep Phase 7 frozen at **0.3700521999**. Phase 9 showed that simply adding a second sparse specialist for the newly audited Claude/GLM/Kimi/Qwen data is not enough.

Do **not** respond by tuning Phase 9's threshold, class subset, SVC C, external sample weights, or gate on the permanent folds.

The next experiment should require a genuinely different source of information, for example:

1. new trustworthy data for the still-uncovered weak labels (`claude-fable-5`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`, `gemini-3.7-flash`); or
2. a materially different representation/model family with complementary OOF errors, frozen and evaluated under `EVALUATION_POLICY.md`.

The blocked Maniceet source stays out until its Kaggle access state changes and its raw rows can be audited directly.

## Data attribution

The competition Data page lists the official dataset under **CC BY-SA 4.0**. See `data/README.md` for source and attribution. External source attribution and provenance are recorded in `artifacts/phase2/` and `artifacts/phase8/`.
