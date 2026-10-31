# Phase 13 — post-Phase-12 reconnaissance

Date: 2026-10-31

## Goal

After Phase 12 rejected the HealthBench Opus 5 / Sonnet 5 pair gate, return to **reconnaissance only**. Do not tune the rejected Phase 12 gate and do not spend another permanent-fold shot without genuinely new information.

Current promoted system remains Phase 7:

- OOF balanced accuracy: `0.3700521998508576`
- Kaggle Public LB: `0.369`
- verified real submissions: 1

## Candidate A — EEE BenchPress Gemini 3.7 Flash

Source: `evaleval/EEE_datastore`

Public provenance looked promising:

- dataset license: MIT;
- BenchPress ingestion commit: `0d402386b021136075b4462d3cec68337c5f2721`;
- ingestion date: 2026-09-10, before the competition opened on 2026-10-14;
- target directory: `data/benchpress/google/gemini-3.7-flash/`;
- exact model identity in every inspected record: `google/gemini-3.7-flash` / `Gemini 3.7 Flash`.

Because three of the five JSON records are around 40 KB, Phase 13 ran an immutable schema probe rather than assuming they contained raw completions.

Probe workflow:

- workflow run: `41221155060`
- artifact: `13049742718`
- artifact digest: `sha256:1858872271842f2676cfafeb5a00e142710fca5d942f2d3ed2885bd6d87bc4f6`
- scope: schema inspection only; no training and no permanent folds.

### Probe result

Five JSON files were present at the pinned revision. Each has the same top-level structure:

- `eval_library`
- `evaluation_id`
- `evaluation_results`
- `model_info`
- `retrieved_timestamp`
- `schema_version`
- `source_metadata`

The larger files contain 14 `evaluation_results`; the small files contain one. Recursive schema inspection found benchmark metadata, generation configuration, metric descriptions, scores, URLs and notes, but **no instance-level response/output/completion/assistant-message/sample/transcript text**.

The only response-like key match was `generation_config.additional_details.context`, a short metadata value, not model-generated text.

Decision: **REJECT AS TRAINING SOURCE.** This source is useful provenance/evaluation evidence for Gemini 3.7 Flash, but it does not contain classifier text.

Status: **AUDITED / AGGREGATE EVALUATION ONLY / NO RAW TEXT**.

## Implication

Do not spend a permanent-fold shot on EEE BenchPress Gemini 3.7 Flash and do not reinterpret benchmark notes or score metadata as model response text.

Next reconnaissance should continue looking for a genuinely pre-competition, publicly available, license-compatible Gemini 3.7 Flash raw-response corpus. If no such source survives provenance checks, stop external-data hunting for that label and require a materially different representation/routing hypothesis before another permanent-fold experiment.
