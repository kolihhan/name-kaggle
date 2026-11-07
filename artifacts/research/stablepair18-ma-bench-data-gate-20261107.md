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
