# Phase 10 — missing-label reconnaissance

Date: 2026-10-31

## Goal

After rejecting Phase 9, do not tune its five-class gate. Search only for genuinely new signal for the five labels that still had no admitted external coverage:

- `claude-fable-5`
- `claude-opus-4-8`
- `claude-opus-5`
- `claude-sonnet-5`
- `gemini-3.7-flash`

A permanent-fold experiment is authorized only after a candidate passes the full admission gate below.

## Candidate A — Claude Fable 5 direct generations

Source: `PawanKrd/claude-fable-5-code`

Public evidence:

- 603 coding/math prompt-response examples;
- response model is `claude-fable-5`;
- generated on 2026-06-10;
- Hugging Face repository updated 2026-06-17;
- prompts originate from `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`, a prompt source already used in the Phase 8 provenance work;
- no chain-of-thought is included in the responses.

Why it matters: Fable 5 is currently our weakest uncovered class (Phase 7 recall ~0.141), so even a modest clean source is potentially high-value.

Blocker: the Hugging Face metadata inspected on 2026-10-31 does not declare a dataset license. **Do not train on this source until license/output-use terms are resolved.**

Status: **HIGH VALUE / LICENSE UNKNOWN**.

## Candidate B — Claude Sonnet 5 direct generations

Source: `Essentialia/claude-sonnet-5-2500x`

The source was pinned and audited at immutable revision:

`9c9fe906b8a9420b6577cdb3db13ffa84fa4c559`

Observed raw state:

- 2,595 rows;
- raw columns: `question`, `answer` only;
- zero null/empty answers;
- 2,592 unique normalized answers;
- two normalized duplicate groups;
- zero exact official-answer overlap;
- zero normalized official-answer overlap;
- answer-only extraction is mechanically safe;
- prompt bank `Roman1111111/prompts-for-claude-opus-4.6` predates competition launch and is MIT licensed.

Two admission gates remain unresolved:

1. **Model identity: UNKNOWN.** The raw rows contain no row-level `model` field. The repository/card name is useful provenance evidence but does not satisfy the frozen Phase 10 requirement for exact raw model identity.
2. **License compatibility: AMBIGUOUS.** The dataset is `CC-BY-NC-4.0`. Because the Kaggle competition carries monetary prizes, this repo will not assume that the non-commercial restriction is compatible without stronger evidence.

Decision: **DO NOT MODEL.** The technical overlap audit passes, but `training_authorized=false`; no permanent-fold shot was spent.

See `artifacts/phase10/sonnet5_audit.json` and `scripts/phase10_audit_sonnet5.py`.

Status: **AUDITED / NOT ADMITTED**.

## Candidate C — EEE Opus 4.8 / Opus 5 records

Source: `evaleval/EEE_datastore`, MIT licensed.

The earlier reconnaissance described these as possible eval-response records. That assumption was wrong.

Pinned revision inspected:

`20380bab5d156501e1e10aabbb7a311cf7812577`

At that immutable pre-competition revision:

- `data/global-mmlu-lite/anthropic/claude-opus-4-8-default/` contains two aggregate `.json` records;
- `data/global-mmlu-lite/anthropic/claude-opus-5-default/` contains two aggregate `.json` records;
- model identity and run dates are explicit;
- no `.jsonl`, parquet, `_samples`, or other instance-level response files exist in either target directory;
- the aggregate records expose evaluation scores/metadata only;
- recursive inspection found no response/output/completion/assistant-message fields.

Decision: **REJECT AS TRAINING SOURCE.** The source proves that those models were evaluated before competition launch, but it does not provide classifier text.

Status: **AUDITED / AGGREGATE ONLY / NO RAW TEXT**.

## Candidate D — ARC Prize Claude Opus 4.8 raw traces

Source: `arcprize/arc_agi_v2_public_eval`.

Pinned immutable source state:

- revision: `026789c1c12a4c34580a32e84dcaf5630d7e8f31`;
- last modified: 2026-06-04, before competition launch;
- license: MIT;
- task files used after excluding each setting's aggregate `results.json`: high **120**, medium **120**, max **90**.

Raw trace identity/extraction:

- every admitted attempt has `metadata.model = claude-opus-4-8`;
- user messages contain the ARC puzzle prompt and are excluded;
- assistant messages contain the model-generated text and are the only classifier text retained;
- the structured ARC `answer` field is separate and excluded.

Full-corpus audit result:

- assistant responses: **868**;
- non-empty: **868 / 868**;
- unique normalized responses: **868 / 868**;
- high: **329**; medium: **329**; max: **210**;
- missing assistant responses: **0**;
- normalized duplicate groups: **0**;
- exact / normalized overlap with official train: **0 / 0**;
- exact / normalized overlap with the already-admitted GPT/Gemini external corpus: **0 / 0**;
- immutable pre-competition gate: **PASS**;
- exact model identity gate: **PASS**;
- MIT license gate: **PASS**;
- response-only extraction gate: **PASS**;
- `training_authorized = true`.

See `artifacts/phase10/arc_opus48_audit.json` and `scripts/phase10_audit_arc_opus48.py`.

Decision: **ADMIT FOR ONE FROZEN PHASE 10 PERMANENT-FOLD EXPERIMENT.** This is the first Phase 10 source to clear every predeclared admission gate.

Status: **ADMITTED / MODELING CONTRACT MAY NOW BE FROZEN**.

## Gemini 3.7 Flash

No new trustworthy raw-response corpus was admitted in this pass.

`maniceet/name-that-llm-openrouter-openai-generations` explicitly contains `gemini-3.7-flash`, but direct anonymous and authenticated Kaggle access remains HTTP 403 in our environment. Google also now marks Gemini 3.7 Flash deprecated/replaced by 3.8 Flash, so fresh API generation would not reliably reproduce the original model: current requests may route to 3.8.

Status: **STILL BLOCKED**.

## Priority order

1. Freeze and execute exactly one Phase 10 experiment using the admitted 868 ARC Opus 4.8 assistant responses; do not tune against the permanent folds.
2. Resolve `PawanKrd/claude-fable-5-code` licensing only after the Opus 4.8 shot is resolved.
3. Keep Sonnet 5 out unless both row-level model provenance and CC-BY-NC compatibility are resolved.
4. Treat EEE as provenance/evaluation evidence only, not as a text-training source.
5. Keep Gemini 3.7 blocked rather than fabricating fresh 3.7 data through a now-rerouted API.

## Admission gate

Before any new source can enter permanent-fold modeling, it must pass:

- exact raw model identity;
- immutable/pre-competition provenance;
- declared usable license/output terms;
- zero exact/normalized answer-text overlap with official train;
- no cross-label duplicate text;
- response-only extraction with no prompt/reference-answer leakage into the classifier text.
