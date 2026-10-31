# Phase 10 — missing-label reconnaissance

Date: 2026-10-31

## Goal

After rejecting Phase 9, do not tune its five-class gate. Search only for genuinely new signal for the five labels that still have no admitted external coverage:

- `claude-fable-5`
- `claude-opus-4-8`
- `claude-opus-5`
- `claude-sonnet-5`
- `gemini-3.7-flash`

No permanent-fold experiment is authorized by this reconnaissance.

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

## Candidate C — pinned EEE Opus 4.8 / Opus 5 eval responses

Source: `evaleval/EEE_datastore`, MIT licensed.

Pinned pre-competition evidence exists for `global-mmlu-lite` records:

- `claude-opus-4-8-default`: September 2026 ingestion, 36-record runs visible in the repository history;
- `claude-opus-5-default`: September 2026 ingestion, 36-record runs visible in the repository history.

Unlike the moving `main` branch rejected during Phase 8, Phase 10 should only use an immutable commit/revision that predates the competition. This converts the previous timing ambiguity into a pinning problem rather than a reason to use current dynamic state.

Expected volume is small, so this is supplementary evidence, not a standalone large training corpus.

Status: **NEXT / PIN IMMUTABLE PRE-COMPETITION REVISION AND AUDIT RAW RESPONSES**.

## Gemini 3.7 Flash

No new trustworthy raw-response corpus was admitted in this pass.

`maniceet/name-that-llm-openrouter-openai-generations` explicitly contains `gemini-3.7-flash`, but direct anonymous and authenticated Kaggle access remains HTTP 403 in our environment. Google also now marks Gemini 3.7 Flash deprecated/replaced by 3.8 Flash, so fresh API generation would not reliably reproduce the original model: current requests may route to 3.8.

Status: **STILL BLOCKED**.

## Priority order

1. Pin an immutable pre-competition `evaleval/EEE_datastore` revision for Opus 4.8 and Opus 5, then audit response-only raw records with exact model provenance.
2. Resolve `PawanKrd/claude-fable-5-code` licensing; if usable, audit its 603 direct generations.
3. Keep Sonnet 5 out unless both row-level model provenance and CC-BY-NC compatibility are resolved.
4. Keep Gemini 3.7 blocked rather than fabricating fresh 3.7 data through a now-rerouted API.

## Gate before Phase 10 modeling

Do not spend another permanent-fold shot until at least one candidate passes:

- exact raw model identity;
- immutable/pre-competition provenance;
- declared usable license/output terms;
- zero exact/normalized answer-text overlap with official train;
- no cross-label duplicate text;
- response-only extraction with no prompt/reference-answer leakage into the classifier text.
