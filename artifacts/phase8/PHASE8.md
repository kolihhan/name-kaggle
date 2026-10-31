# Phase 8 — external coverage audit

Date: 2026-10-31

## Goal

Find new external signal for labels not covered by the promoted Phase 7 specialist without weakening the permanent official `GroupKFold(prompt_id)` validation.

**Phase 8 is audit-only. No new external corpus is authorized for training yet.**

## 1. Maniceet candidate: access blocked

Target: `maniceet/name-that-llm-openrouter-openai-generations`.

GitHub Actions run `41197148926` tested both routes:

- anonymous Kaggle download -> HTTP 403;
- authenticated Kaggle CLI -> HTTP 403, `You don't have access to this resource.`

This is an access gate, not a workflow/parsing failure. Do not retry without a changed Kaggle access state.

## 2. Response corpus: raw access passed

Pivot target: `olegzholobov/name-that-llm-response-corpus`.

Frozen archive SHA256:

`8e238480b2d53b821f35eed847dca4538f91ee2fcff4f104370f896bdfc83ece`

Short-view parquet SHA256:

`15a67d022315b2508af8edd749a7c041d42cc37e266106028c73e405659c400f`

The short view contains **24,025** non-empty model-response cells and all 18 official labels.

## 3. Mechanical/provenance audit

- mechanically clean cells: **24,025**
- license-known cells: **14,998**
- license-known + output-terms-clear cells: **4,463**
- cross-label normalized duplicate groups: **0**
- official answer exact overlap: **0**
- official answer normalized overlap: **0**
- exact-prompt provenance fraction: **99.91%**

A strict source must be 100% license-known, 100% output-terms-clear, and 100% exact-prompt provenance. Restricting those sources to the 10 Phase 7-uncovered labels produced **2,992 rows / 2,992 unique responses**.

## 4. Pre-competition gate

A second frozen gate requires the source or prompt bank to be independently evidenced as public before the competition opened on **2026-10-14**.

Retained sources:

- `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` — public/updated 2026-05-01;
- `greghavens/kimi-k3-coding-and-debugging-traces` — public/updated 2026-07-31;
- `Zek-Takai/glm53-flash-harvest` — harvested 2026-08-30 to 2026-09-02, updated 2026-09-03;
- `zlab-princeton/SWEeper-Bench-traces` — benchmark/prompt bank publicly documented 2026-10-08.

Excluded despite passing the mechanical strict-source rule:

- `evaleval/EEE_datastore` — dynamic datastore updated after competition launch; exact 57 rows were not pinned strongly enough to a pre-launch state;
- `Met4physics/claude-opus-4-8-xhigh-reasoning-8.7k` — only 14 rows and no sufficiently strong authoritative timestamp was pinned, so it is not worth carrying the uncertainty.

Real-data GREEN run: `41198660881`.

Final pre-competition candidate:

| Label | Rows |
|---|---:|
| claude-opus-4-6 | 1,934 |
| glm-5.3 | 200 |
| glm-5.3-flash | 284 |
| kimi-k3 | 317 |
| qwen3.8-max | 186 |
| **Total** | **2,921** |

Properties:

- **2,921 rows**
- **2,921 unique response IDs**
- official exact/normalized answer overlap: **0 / 0**
- covers **5 of the 10** Phase 7-uncovered labels

Still uncovered by this candidate:

- `claude-fable-5`
- `claude-opus-4-8`
- `claude-opus-5`
- `claude-sonnet-5`
- `gemini-3.7-flash`

## 5. Residual validation risk

Official train exposes `prompt_id` but not prompt text. Therefore released files cannot mathematically prove that no external prompt happens to equal a held-out official prompt.

Status: **residual prompt identity = UNKNOWN**.

What we can support is narrower and defensible: the retained external sources/prompt banks demonstrably predate competition launch, and their answers have zero exact or normalized-text overlap with official training answers.

Do **not** describe future external-data CV as "provably prompt-leakage-free."

## Decision

- raw access: **PASS**
- schema/18-label audit: **PASS**
- answer-text overlap gate: **PASS**
- strict provenance gate: **PASS**
- pre-competition provenance gate: **PASS**
- residual accidental prompt identity: **UNKNOWN / unobservable**
- training authorized in Phase 8: **NO**

## Next gate

Phase 9 must freeze, before training:

1. the exact validation statement acknowledging the residual unobservable prompt-identity risk;
2. exactly one no-tuning experiment using the 2,921-row pre-competition candidate;
3. a structural promotion rule that prevents another post-hoc parameter/threshold sweep on the permanent folds.
