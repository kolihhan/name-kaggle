# Name That LLM — Phase 2: External Data

Date: 2026-10-31

## Objective

Admit external generations only when they improve the **frozen official `prompt_id` GroupKFold OOF** without label/provenance/leakage problems. Random-row CV is not accepted as evidence.

## Admission rules

An external source must pass the applicable gates:

| Gate | Requirement |
|---|---|
| Label identity | Exact competition label / verified exact model snapshot |
| Provenance | Generation route/model/provider documented; row-level preferred |
| License | Publicly usable; attribution recorded |
| Prompt independence | New prompts preferred |
| Duplicate safety | No exact overlap; near-similarity audited against official train |
| Protocol match | Prefer single passage, roughly 40–250 words, no tool/CoT traces |
| Coverage | Per-label counts recorded |
| Honest validation | Improvement on the frozen official grouped folds |

External generations derived from official train prompts must not be globally injected into every CV fold. The default is to reject them during model selection unless their prompt lineage can be fold-isolated.

## Executed source — GPT + Gemini generations

Source snapshot: `RohanExploit/kaggle-name-that-llm-top20/data/extra_gpt_gemini/ntl_generations.csv`, byte-pinned by Git blob SHA `8dccb6d8f8b8388b134a6aa8f8ea42e99df08c10`.

### Raw audit — VERIFIED

- 2,503 rows
- 430 prompt IDs
- 8 exact competition labels
- 0 duplicate normalized texts
- 0 exact normalized-text overlap with official train
- maximum TF-IDF word 1–2 similarity to official train: 0.166
- 0 blank `model_seen` rows
- 98.4% of passages are 40–250 words
- raw identity/provenance gate: PASS

Evidence: `gpt_gemini_raw_audit.json`.

### Frozen-CV ablation — VERIFIED

The Phase 1 baseline was reproduced at **0.319289**, only `7.46e-06` from the frozen historical score.

| Experiment | Balanced accuracy | Δ |
|---|---:|---:|
| Official only | 0.31929 | — |
| Official + all 2,503 rows | **0.34116** | **+0.02187** |
| Official + balanced 1,120-row slice | 0.33597 | +0.01668 |

Evidence: `gpt_gemini_char_ablation.json`.

**Decision: PROMOTE** this source into the first encoder-stage A/B experiment. This does not promote TF-IDF as the final model; TF-IDF is only the cheap admission gate.

## Maniceet OpenRouter + OpenAI corpus — BLOCKED, NOT AUDITED

Source: `maniceet/name-that-llm-openrouter-openai-generations`.

Public metadata reports 7,753 passages, 10 exact competition labels, row-level generation provenance, train-overlap filtering and CC BY 4.0. Those metadata claims remain useful for prioritization, but the raw dataset has not been independently inspected in this repo.

Access diagnostics are now resolved:

- `KAGGLE_API_TOKEN` exists and `kagglehub.whoami()` succeeds;
- the same token can access the official `name-that-llm` competition API;
- the Maniceet dataset returns HTTP 403 from both Kaggle CLI dataset endpoints and `kagglehub.dataset_download`.

**Decision: HOLD** as `blocked_resource_access_or_consent`. Do not stall the modeling path on it and do not treat the reported 7,753 rows as locally verified.

Evidence: `maniceet_kaggle_access_check.md`.

## Other candidate sources

### Extra API Passages — targeted Claude source

Promising because sibling Claude models remain a major confusion area. Public evidence describes new prompts and grouped-CV use. Still requires our own raw audit before training use.

**Status:** Tier B / next targeted source after the encoder baseline if needed.

### Response Corpus — broad 18-label candidate

Potentially useful full-label coverage, but original licensing/provenance/train-overlap details still require direct verification.

**Status:** Tier B / hold.

### 254k aggregate pool

High coverage but high lineage risk: mixed upstream datasets, licenses, aliases/snapshots, prompt origins and duplicate policies.

**Status:** Tier C / do not train yet.

### Qwen3.8-Max Distillation 50K

Rejected for Phase 2: teacher snapshot does not exactly match the competition label, protocol is reasoning/tool heavy, and licensing/terms are heterogeneous.

## Repo execution contract

Experiment logic belongs in `scripts/`; GitHub Actions are execution wrappers only.

Canonical Phase 2 scripts:

- `scripts/audit_gpt_gemini_external.py`
- `scripts/run_external_ablation.py`

## Current checkpoint

Phase 2 has done enough to leave reconnaissance mode.

**Next:** first encoder A/B under the frozen folds:

1. official-only fold training;
2. official + promoted 2,503 external rows;
3. same encoder/config/seed/folds;
4. compare official validation balanced accuracy and per-model recall;
5. only then decide whether to tune the encoder, move to LoRA, or audit another external source.
