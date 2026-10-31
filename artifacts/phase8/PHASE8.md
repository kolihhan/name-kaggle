# Phase 8 — external coverage audit

Date: 2026-10-31

## Goal

Find new external signal for labels not covered by the promoted Phase 7 specialist, without contaminating the permanent official `GroupKFold(prompt_id)` validation.

**This phase is audit-only. No new external corpus is authorized for training yet.**

## 1. Maniceet candidate: access blocked

Target: `maniceet/name-that-llm-openrouter-openai-generations`.

GitHub Actions run `41197148926` tested both acquisition routes:

- anonymous Kaggle dataset download -> HTTP 403;
- authenticated Kaggle CLI using the repository Kaggle token -> HTTP 403, `You don't have access to this resource.`

This is an access gate, not a parsing/workflow failure. Do not repeatedly retry it without a changed Kaggle access state.

## 2. Response-corpus candidate: raw access passed

Pivot target: `olegzholobov/name-that-llm-response-corpus`.

Anonymous Kaggle API access succeeded. Frozen archive SHA256:

`8e238480b2d53b821f35eed847dca4538f91ee2fcff4f104370f896bdfc83ece`

The short-view parquet SHA256 is:

`15a67d022315b2508af8edd749a7c041d42cc37e266106028c73e405659c400f`

The short view contains 24,025 non-empty model-response cells and covers all 18 official labels.

## 3. Audit results

Real-data audit run: `41198184154`.

- mechanically clean short cells: **24,025**
- license-known cells: **14,998**
- license-known + output-terms-clear cells: **4,463**
- clean cross-label normalized duplicate groups: **0**
- official answer exact overlap: **0**
- official answer normalized overlap: **0**
- exact-prompt provenance fraction: **99.91%**

The answer-text overlap gate passes.

## 4. Frozen conservative source rule

Do not use all 24,025 rows.

A source is eligible only when, across its mechanically-clean short rows, it is simultaneously:

1. 100% license-known;
2. 100% output-terms-clear;
3. 100% exact-prompt provenance.

Then retain only the 10 labels that were uncovered by Phase 7.

The rule is data-driven and frozen before any CV run. It yields six sources:

- `Met4physics/claude-opus-4-8-xhigh-reasoning-8.7k`
- `Zek-Takai/glm53-flash-harvest`
- `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`
- `evaleval/EEE_datastore`
- `greghavens/kimi-k3-coding-and-debugging-traces`
- `zlab-princeton/SWEeper-Bench-traces`

Resulting strict candidate:

- **2,992 rows**
- **2,992 unique response IDs**
- official exact/normalized answer overlap: **0 / 0**

Coverage:

| Label | Rows |
|---|---:|
| claude-opus-4-6 | 1,991 |
| claude-opus-4-8 | 14 |
| glm-5.3 | 200 |
| glm-5.3-flash | 284 |
| kimi-k3 | 317 |
| qwen3.8-max | 186 |

Still missing from the strict candidate:

- `claude-fable-5`
- `claude-opus-5`
- `claude-sonnet-5`
- `gemini-3.7-flash`

## 5. Remaining blocker

Official training data exposes `prompt_id` but not the actual prompt text. Therefore released data cannot prove that an external prompt is different from every held-out official prompt.

Status: **prompt-level leakage = UNKNOWN**.

This does not negate the verified zero answer-text overlap or provenance checks. It means we must not call an external-data CV run "provably prompt-leakage-free."

## Decision

- raw external access: **PASS**
- 18-label schema/coverage audit: **PASS**
- answer-text overlap gate: **PASS**
- conservative source subset: **PASS FOR PROVENANCE AUDIT**
- prompt-level leakage gate: **BLOCKED / UNKNOWN**
- training authorized: **NO**

Next gate: either obtain stronger evidence that the strict sources use independently constructed public benchmark prompts, or explicitly freeze a validation statement that accepts the residual unobservable prompt-overlap risk before running exactly one no-tuning external-data experiment.
