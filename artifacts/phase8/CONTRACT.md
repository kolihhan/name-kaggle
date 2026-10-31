# Phase 8 — Maniceet raw-data audit contract

Date: 2026-10-31

## Goal

Determine whether `maniceet/name-that-llm-openrouter-openai-generations` is admissible as a new external source before any training experiment.

Phase 8 is audit-only. No classifier may train on this source until the raw-data gate passes.

## Public card claims to verify from raw rows

Current Kaggle card reports:

- 7,753 passages
- 10 competition labels
- per-row provenance
- CC BY 4.0
- labels: `gemini-3.7-flash`, `glm-5.3`, `glm-5.3-flash`, `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`, `kimi-k3`, `qwen3.8-max`
- prompt bodies were filtered against competition-train passages and released passages were checked for train-text duplicates

These are hypotheses until the downloaded raw version is inspected.

## Stage A — immutable fetch evidence

Acquire the current public Kaggle dataset archive without modifying repository data.

Record:

- acquisition route
- archive SHA-256 and byte size
- extracted file inventory, sizes, and SHA-256 hashes

Upload the downloaded archive/extracted files only as a GitHub Actions artifact. Do not commit raw data automatically.

Anonymous public Kaggle download may be attempted first. `KAGGLE_API_TOKEN` is only a fallback; never print it.

## Stage B — raw-row gate

After Stage A reveals the actual files/schema, audit the selected generation table directly. Hard failures include:

- row count differs from the version/card being audited without an explicit version explanation
- model labels differ from the documented 10-label set or include labels outside the competition target set
- blank text/model identifiers
- duplicate `row_id` when `row_id` is present
- `row_id` does not match the documented first 16 hex digits of SHA-1 over whitespace-normalized text when that rule is present in the downloaded version
- exact whitespace-normalized passage overlap with official train
- provenance columns are absent to the point that generator/provider/source routing cannot be audited

Report, but do not automatically hard-fail solely on:

- duplicate prompts across rows
- missing `served_model` where the card explicitly allows it
- prompt metadata blanks explicitly documented by the card
- near-similarity counts to official passages

## Prompt/leakage audit

The audit must separately check:

- exact normalized passage overlap with official train
- exact normalized prompt-text overlap with any recoverable official prompt text available in this repository
- prompt IDs/counts and prompt-source distribution
- whether fold-wise exclusion is needed if external prompts overlap official prompt groups

Do not claim honest CV transfer if prompt overlap cannot be resolved.

## Coverage value

Report exact coverage of Phase 7's 10 uncovered labels. Public metadata suggests this source adds five: `gemini-3.7-flash`, `glm-5.3`, `glm-5.3-flash`, `kimi-k3`, and `qwen3.8-max`; raw rows must confirm this.

## Promotion rule

Promote the source to a later training experiment only if:

1. raw identity/provenance gate passes;
2. label mapping is exact and competition-valid;
3. official passage overlap is zero;
4. prompt leakage is either absent or can be excluded fold-by-fold under the permanent `GroupKFold(prompt_id)` validation contract.

No Phase 8 result may change the current Phase 7 best OOF score or model selection by itself.
