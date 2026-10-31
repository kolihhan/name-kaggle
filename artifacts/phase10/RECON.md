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

Public evidence:

- 1K–10K size class, repository name/card states ~2,500 Sonnet 5 generations;
- Hugging Face repository updated 2026-08-21, before competition launch;
- license: `CC-BY-NC-4.0`;
- prompting source is explicitly documented as `Roman1111111/prompts-for-claude-opus-4.6`;
- card states no chain-of-thought.

This is a materially stronger candidate than trying to tune Phase 9 because it covers a label with zero admitted external data.

Before admission, raw rows still need model-label verification, duplicate/overlap audit, prompt provenance inspection, and confirmation that non-commercial licensing is compatible with the intended Kaggle use.

Status: **PROMISING / RAW AUDIT REQUIRED**.

## Candidate C — pinned EEE Opus 4.8 / Opus 5 eval responses

Source: `evaleval/EEE_datastore`, MIT licensed.

Pinned pre-competition evidence exists for `global-mmlu-lite` records:

- `claude-opus-4-8-default`: September 2026 ingestion, 36-record runs visible in the repository history;
- `claude-opus-5-default`: September 2026 ingestion, 36-record runs visible in the repository history.

Unlike the moving `main` branch rejected during Phase 8, Phase 10 should only use an immutable commit/revision that predates the competition. This converts the previous timing ambiguity into a pinning problem rather than a reason to use current dynamic state.

Expected volume is small, so this is supplementary evidence, not a standalone large training corpus.

Status: **SMALL BUT AUDITABLE / PIN IMMUTABLE REVISION FIRST**.

## Gemini 3.7 Flash

No new trustworthy raw-response corpus was admitted in this pass.

`maniceet/name-that-llm-openrouter-openai-generations` explicitly contains `gemini-3.7-flash`, but direct anonymous and authenticated Kaggle access remains HTTP 403 in our environment. Google also now marks Gemini 3.7 Flash deprecated/replaced by 3.8 Flash, so fresh API generation would not reliably reproduce the original model: current requests may route to 3.8.

Status: **STILL BLOCKED**.

## Priority order

1. Audit `Essentialia/claude-sonnet-5-2500x` raw rows and license/use compatibility.
2. Resolve `PawanKrd/claude-fable-5-code` licensing; if usable, audit its 603 direct generations.
3. Pin immutable pre-competition EEE revisions for Opus 4.8 and Opus 5 and extract only response text with exact model provenance.
4. Keep Gemini 3.7 blocked rather than fabricating fresh 3.7 data through a now-rerouted API.

## Gate before Phase 10 modeling

Do not spend another permanent-fold shot until at least one candidate passes:

- exact raw model identity;
- immutable/pre-competition provenance;
- declared usable license/output terms;
- zero exact/normalized answer-text overlap with official train;
- no cross-label duplicate text;
- response-only extraction with no prompt/reference-answer leakage into the classifier text.
