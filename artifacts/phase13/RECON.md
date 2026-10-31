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

## Candidate B — ProofJudge Gemini 3.7 Flash traces

Source: `SJCaldwell/proofjudge-eval-traces`.

This is the cleanest technical Gemini 3.7 source found in Phase 13.

Immutable timing was resolved from the Hugging Face commit history rather than page-relative timestamps:

- target file: `data/traces_gemini-3.7-flash.jsonl`;
- first commit containing the file: `0cb577b0dfef6d3fcb0398d65e1c32351c99f6e4`;
- first public timestamp: `2026-08-18T01:11:59+00:00`;
- latest pre-competition commit containing the file: `2fe14f4cb00d624094d9d248fc55012db71ce114`;
- latest pre-competition timestamp: `2026-08-18T01:14:11+00:00`;
- declared dataset license: MIT.

Audit-only workflow:

- workflow run: `41221809227` — SUCCESS;
- artifact: `13050805092`;
- artifact digest: `sha256:96bb1dd9125ffc6a35eabe2890f8952d341229a524395125b7c601c3943ba2eb`;
- no model training and no permanent folds.

### Response-only extraction result

At the pinned pre-competition revision:

- exact raw model identity: `gemini-3.7-flash` on all retained trajectories;
- 1,308 trajectories with non-empty agent-authored message text;
- 1,308 unique session IDs;
- 436 unique evaluation cases, with three independent replicates;
- 1,308 normalized-unique extracted texts;
- zero normalized duplicate groups;
- zero exact or normalized official-train overlap;
- zero exact or normalized overlap with the admitted Phase 7 external corpus;
- extracted text length: min 1,280 chars, median 2,222, mean 2,226.1, max 3,386.

Extraction is mechanically response-only:

- retain only non-empty `step.message` where `step.source == "agent"`;
- exclude system prompts;
- exclude user prompts;
- exclude observations/tool outputs;
- exclude tool-call arguments.

The raw traces contain 12,975 tool-call steps with empty agent message fields; those tool calls were excluded from classifier text rather than serialized into it.

Technical audit verdict: **CLEAN CANDIDATE**.

### Provider-output terms blocker

The technical audit is not sufficient to authorize training.

Google's current public terms prohibit using AI-generated content from Google services to develop machine-learning models or related AI technology. A model-attribution classifier is itself an ML model. The dataset's MIT declaration is strong evidence that the publisher intended the traces to be reusable research artifacts, but it is not enough for this repository to assume that it overrides provider-specific output-use restrictions.

The ProofJudge public artifacts do not establish a separate downstream-output license or provider route that clearly removes this restriction.

Decision: **DO NOT MODEL UNDER THE CURRENT CONSERVATIVE ADMISSION STANDARD.**

Status: **AUDITED / TECHNICALLY CLEAN / PROVIDER-TERMS BLOCKED**.

`training_authorized=false` is therefore intentional, not an audit failure.

## Other Gemini 3.7 candidates reviewed

### `bigdatamark/synthid-text-research`

Claims August 2026 Gemini 3.7 Flash and Sonnet 5 generations, but no verified pre-2026-10-14 public provenance was found. The Hugging Face publication path observed in reconnaissance is post-competition.

Status: **NOT ADMITTED / PUBLIC TIMING NOT PROVEN**.

### `kailasa-ngpt/gemini-3.7-flash-ocr-26-aug-2026`

Large raw teacher-output corpus with exact Gemini 3.7 provenance and an August date, but Hugging Face declares the license as `other`; downstream output-use rights were not established.

Status: **NOT ADMITTED / LICENSE-OUTPUT TERMS UNKNOWN**.

### `Rapidata/text-to-svg-human-feedback-benchmark`

A strong provenance corpus:

- pre-competition immutable August revision exists;
- Gemini 3.7 Flash appears explicitly in per-row model fields;
- raw SVG model output is stored;
- dataset annotations/prompts use CC-BY-4.0.

However, the dataset card explicitly separates third-party model outputs from the CC-BY-4.0 grant and leaves them governed by provider terms. This does not resolve the same Google generated-output restriction.

Status: **NOT ADMITTED / PROVIDER-OUTPUT TERMS AMBIGUOUS**.

### JetBrains SWE-TaskFlow trajectories / public syzbot trajectories

Both expose useful raw Gemini 3.7 agent/model text and pre-competition activity, but their output-use/licensing path is not cleaner than ProofJudge. They do not justify another audit while ProofJudge already provides a technically superior clean corpus with the same provider-terms blocker.

Status: **DEPRIORITIZED**.

## Phase 13 decision

Stop hunting additional Gemini 3.7 external outputs unless a source provides a materially clearer downstream-output grant than the candidates above. More raw Gemini text does not solve the current blocker.

Do **not** spend a permanent-fold shot on EEE, ProofJudge, Rapidata, OCR teacher outputs, JetBrains trajectories, or syzbot under the current evidence.

The next reconnaissance target is a genuinely different signal that does not depend on restricted third-party model outputs: public competition methods, representation changes, or routing structures that can be justified before touching the permanent folds.
