# Kimi K3 Reasoning Protocol — strict 80-pair, no-training champion diagnostic

**Date: 2026-11-11, PREDECLARED before 0.679 champion inference on this data.**

## Corrected experiment (not the failed 144-pair contract)

The [original 144-pair gate](../ntl-kimi-protocol-pair/CONTRACT.md) for `calls.jsonl` was **STOPPED**, because 988/1000 same-Prompt Kimi pairs differed on `target_words`. [Field-by-field audit](https://github.com/kolihhan/name-kaggle/actions/runs/42973523636): actual model_id, served_model and served_provider **each matched in all 1000**, but target length matched for only 12. No GPU or model test was performed on those calls. Do not reopen that experiment by relaxing guards.

A **distinct pre-existing trial protocol bank** `setting_trials.jsonl` has **80 Kimi K3 high/off paired prompts**; every pair has the **same** non-null model_id, served_model, served_provider, target_words, `finish_reason=stop`, nonempty text and 40–250 actual words. Source dataset: `willrice/name-that-llm-openrouter-harvest` (CC BY 4.0, original prompt bank by `seyitkaangunes/ntl-extra-api-passages` CC BY 4.0). Source SHA256 `a38745dd0b648773204193d2a34a9b1ccd5ce9a41cfd01b51eea49ceed56d87e`. [Read-only source/schema evidence](https://github.com/kolihhan/name-kaggle/actions/runs/42972737341).

**Sampling:** all 80 high/off pairs, no selection, no filtering by classifier score or word length beyond the fixed pair criteria above. Require exactly 80 qualifying prompt groups (one high & one off each); sort by `prompt_id` deterministically; form 160 `id,text` rows. If a pair fails, abort, no threshold change. Validate 160 normalized text hashes are unique and none exactly match official training text (only inside private Kaggle). Keep all raw passages in temporary/private memory, never public Git.

## Frozen scoring

- Reuse our own **official 0.679 champion** (Kaggle submission `58210835`) from private Notebook `kolihhan/ntl-ozertuu-0683-guarded-no-train`.
- Fix executable code-cell SHA256 `28fd128b745f0d95b70f79ea9b2c546d981edf9be08207cbf71a2d9e80ee1c86`, original 20 data, 8 kernel and 6 model inputs, and all 3 fail-closed no-new-training guards. Use the same original 18-class model output (not a binary Kimi-only classifier). No fit, changing weights, blending, resampling, training or Kaggle competition submission.
- Build and run one **private** Kaggle Notebook, with protected temporary CC-licensed text capsule. Kaggle Github runner must retain only aggregate statistics and SHA fingerprints, not raw text or individual examples.
- The 80 pairs must have exactly same model_id, served_model, served_provider, prompt_id and target_words. However `prompt_id` and fixed target words do **not prove actual request strings, other sampling kwargs or served-weight revision are identical**; do not claim a randomized causal experiment.
- Primary: per-condition `kimi-k3` recall; `high correct, off wrong` vs `off correct, high wrong` paired discordant counts, difference in percentage points and exact two-sided binomial/McNemar p-value. Report model-family and most common wrong labels. Four source-order blocks of 20 pairs show stability.
- Secondary: same-prompt prediction flips, observed response length difference, reasoning token metadata.
- **Predeclared evidence criterion:** difference in Kimi recognition at least 15 percentage points, exact McNemar p-value below 0.01, and same direction in all 4 fixed 20-pair blocks => material protocol sensitivity. Otherwise mixed / insufficient.
- Result remains **12/18 unrelated to competition BA**, in fact is Kimi-only. No score promotion. No adaptation to this inspected set; a deployable 0.700+ model still needs new independent full-18-label data and same-row rescue-regression evidence.

## Identity vs protocol limits

This is **within Kimi**, one of six labels missing from last week's 347-row cohort. Observed high/off differences may reflect reasoning effort, text length, prompt rendering, provider routing or temporal model revision. Presence of matching served_model IDs does not verify exact weights. Distinguish data generation fingerprint from generator identity.

## Compliance

No new training; no Kaggle competition submission; don't upload official train/test or source texts to public GitHub. CC BY 4.0 attribution retained in private Notebook. One GPU run maximum after preflight; if preflight fails, stop. Existing `0.679` champion stays untouched.
