# STABLEPAIR-18 — MA-Bench candidate source gate (frozen before data processing)

Date: 2026-11-07. **Research metadata-only, no model inference/training/submission.**

## Source and reason to examine

[ScaleAI/MA-Bench](https://huggingface.co/datasets/ScaleAI/MA-Bench) released a documented public CC-BY-4.0 dataset of generated mission-analysis briefs. The dataset card lists **10 scenarios**, and both GPT-5.6 Luna and Sol under two conditions, `benchmark_capped` (3,000-word cap) and `uncapped` (no cap), medium effort where offered. `runs/briefs.jsonl` exposes `brief_id,condition,case_id,scenario,model,model_slug,seed,text`. This dataset was not in the previously audited Oleg/SWEeper, Zakir, Maniceet, 834-row proxy or Aless/Elliot cohorts. No inference about independence from external model-fitting corpora until overlap checks exist.

## Predeclared candidate eligibility tests

Run **CPU, read-only external-public dataset**, parse JSONL line-by-line and discard the raw `text` immediately after counting word lengths. Preserve only aggregate evidence; never commit passages or labels to the public repository.

1. Byte count and SHA256 of `runs/briefs.jsonl`, total valid row count, model/condition/case/seed coverage; exact data package SHA provenance.
2. For Luna and Sol, count **distinct case_id/scenario prompts** with both labels under capped, uncapped, and their intersection, plus matched `case_id,seed` coverage; repeats within a case are **NOT new independent prompts**.
3. Check whether the released per-response schema contains explicit `served_model`, `endpoint_version`, `temperature`, `request_max_words`, `system_prompt_hash`, and `timestamp` or equivalent fields. Absence means version/protocol certainty is not established by the response file; consult card to separate documented global settings from per-request proof.
4. Record response length distribution by label/condition (aggregate only), but **no signal fitting** or length-as-model heuristic.
5. **Pilot eligibility PASS** only if >= 100 unique case/prompt identifiers with both Luna and Sol under each of the two conditions, all schema checks pass, and both conditions' full request and exact served model/version are verified. **Fail/blocked** otherwise. A PASS is permission to freeze a later CPU experiment, not proof of protocol invariance or Kaggle promotion.
6. For the full STABLEPAIR-18 data Gate 0, a second independently sourced, non-overlapping matched corpus is additionally required. One MA-Bench source alone can never satisfy the overall gate.

## Expected issue (prediction before execution)

The *card* lists only **10 independent scenarios**; so we expect >= 100 independent prompt gate to fail regardless of row volume or 5 repeats. This explicit test still confirms whether the candidate can support a small exploratory demonstration and avoids expensive GPU/fitted-model evaluation. The domain is **long-form military planning**, not Kaggle's 40–250-word self-contained responses; even a full match would need an additional short-form source transfer before competition use.

## Output and stop

The only output is aggregate JSON, run ID, source SHA and PASS / NOT_ELIGIBLE / INACCESSIBLE. No model fit, GPU, 14B, Kaggle submission or revision of spent prior candidate.

## Executed result — 2026-11-07

**Verdict: NOT_ELIGIBLE_FOR_PAIRED_PROTOCOL_PILOT; full STABLEPAIR source gate NOT MET.**

- [Successful read-only CPU GitHub Actions run 42388976208](https://github.com/kolihhan/name-kaggle/actions/runs/42388976208): **3/3 synthetic gate tests PASS**, public MA-Bench `runs/briefs.jsonl` streamed and parsed, result verification PASS, aggregate-only artifact uploaded (**artifact ID 13580239635**, zip SHA256 `656c1cfd0854b04437d591a2e2fe20c938f5c41cc8435ee75903c365837a2cd2`).
- Source JSONL **1,560** distinct `brief_id` records, **47,916,491 bytes**, SHA256 `a9e28e99849153e3edf854fbf9abe5ce6650530bb5249fa1d8db332eccbd67c8`. No raw response text retained in GitHub; aggregated in-memory counts only. No Kaggle competition data, model inference, training, or GPU used.
- Luna/Sol under `benchmark_capped`: 30 rows each, exactly **10 matched distinct cases**, **30 matched (case,seed)** cells.
- Luna/Sol under `uncapped`: 30 rows each, exactly **10 matched distinct cases**, **30 matched (case,seed)** cells.
- Across BOTH capped and uncapped: **10 matched distinct cases**, **30 matched (case,seed)** cells, not 30 independent prompts. Ten is below the precommitted minimum 100, even though the full source includes 1,560 rows and dozens of other models.
- The individual response records have **none** of the requested complete per-row fields `served_model`, `endpoint_version`, `temperature`, `request_max_words`, `system_prompt_hash`, `timestamp`. The public card documents the high-level protocol (3,000-word cap vs uncapped and medium effort *where exposed*), but this is not the same as full per-response endpoint/version verification.
- Exploratory, **not causal and not submitted**: whitespace-token word-count medians for capped Luna **3,080** vs Sol **2,486**; uncapped Luna **7,177** vs Sol **7,951**. These are unpaired marginal medians over 30 rows per model/condition, not independent-scenario paired effects; whitespace-token counting need not match the benchmark's word-cap counting. The apparent direction reversal is a **warning about protocol dependence**, not a proven model fingerprint.
- The content is 2,000–9,500-word *military planning briefs* instead of Kaggle's **40–250-word self-contained paragraphs**. A single source with 10 independent tasks cannot validate generalization to unseen Kaggle prompts.

### Additional public-source triage (card-level; not downloaded or audited as row data)

- [Crownelius/GPT-5.6-Sol-Luna-Terra-Traces](https://huggingface.co/datasets/Crownelius/GPT-5.6-Sol-Luna-Terra-Traces): public card explicitly says Sol samples are Codex CLI coding traces from one upstream source, Luna samples are a different upstream persona-injected SFT source, and Terra has zero rows in its stated snapshot. **Label and collection protocol are perfectly confounded**; not suitable for matched-protocol short-passages. Counts reported by the source are per-row source claims, not an independently verified generation audit.
- [alexshpunt/explicit-edit-benchmark](https://huggingface.co/datasets/alexshpunt/explicit-edit-benchmark): 226 matched tasks, several model/harness configurations, but the published observations intentionally **do not include model output text**, according to its README. Cannot train a text-attribution scorer from its metrics tables.
- [aporthq/vault-benchmark-v1](https://huggingface.co/datasets/aporthq/vault-benchmark-v1): 4,371 attack conversations and large model output/transcript coverage, but underlying files require acceptance of gated access conditions and the card states returned model revisions were **not recorded** in released manifest. This is also an adversarial multi-turn banker-agent domain, not standalone short paragraphs. **No access bypass attempted.**
- [kotekjedi/stolen-thoughts](https://huggingface.co/datasets/kotekjedi/stolen-thoughts): public card reports many shared problems and multiple model/effort outputs but access is gated under non-commercial research terms and includes recovered proprietary reasoning. **Do not acquire or use without explicit authorized access and terms review**.
- Previously audited Maniceet OpenRouter/OpenAI dataset normal Kaggle access returned 403, so this attempt did not retry or bypass that source.

### Stop decision and next scientifically productive action

**STOP STABLEPAIR-18 training on this MA-Bench cohort.** No pointwise/pairwise/shuffled CPU classifier fitted, because the predeclared metadata/prompt/endpoint gate failed. No claim of protocol-invariant sibling evidence, source-disjoint transfer, or improvement over reproduced best Public BA 0.668.

To reopen the full research claim, the missing dataset must have at least **100 independent matched short-form prompts** for one exact sibling pair under two documented conditions, with exact served model/version and original request settings, plus a genuinely independent transfer cohort. If this is unattainable from licensed public records, a separately authorized new-data generation experiment with explicit budget/endpoint/terms constraints would be required; no paid API request was made.

**Research bottleneck now demonstrated:** data *identifiability* and short-form transfer, not a known 8B-vs-14B evaluator-capacity limit. The 0.668 comparison remains **UNTESTED**, not a reported improvement.
