# Dual-Embedding + Tencent WeMM — frozen inference-only candidate

**Frozen at 2026-11-12 MYT, before any personal Kaggle score.**
**Status:** NOT ELIGIBLE FOR RESUBMISSION. Official `58244483` failed hidden runtime with no score; real-path CUDA-bootstrap v3 also failed. The fixture-shortcut defect is corrected in the builder with CPU regression coverage only. Full-model preflight and numerical advantage over 0.679 remain unverified. See [current evidence](SUBMISSION_20261112.md).

## Goal and model-class hypothesis

Our currently verified complete Public BA is **0.679**, `58210835`, from an attributed public Ozertuu reproduction. A new public dual-embedding lineage adds a **Tencent WeMM Embedding 4B** model to Qwen3 Embedding / multi-backbone prefit LoRA / pretrained sparse and other ensemble members. An independent embedding space might distinguish near-sibling generator styles missed by the incumbent. **No measured gain or complementarity is established.** The hypothesis could fail if it is a correlated embedding, same-source overfit, or distorts the 18-way balance.

This is a **new model-input/representation candidate**, not a minor Sinkhorn hyperparameter patch and not an original pretrained architecture. It is **not** the private 0.765 leaderboard leader's unreleased solution.

## Immutable source and permissions

- Canonical public Notebook source: `ozertuu/llm-dual-embedding-lora-qwen3-8-27b-gsq` by Ertuğrul Özer.
- Source Notebook container SHA changed after the initial audit from `e943174f...` to `8708ba43653f262f1be3fdec423ffcb463ea07affc765e69fab7c633a09af1a1`, but the executable code-cell digest stayed identical. We pin exact executable code and all input manifests; container/markdown changes cannot relax training guards. Verified [source preflight #43059463986](https://github.com/kolihhan/name-kaggle/actions/runs/43059463986).
- Executable cell SHA256: `a65e0115aaa56e3ce64391e9911fed7d3521c7ace3a1e990d92cc4f1b8a03a82`, lineage [#43057087250](https://github.com/kolihhan/name-kaggle/actions/runs/43057087250); exactly identical to `omkarpattanaik123/name-that-llm-v28-sovereign-titan-apex`. Do not mistake this fork for a separate improvement.
- Input manifest: exactly **21 dataset sources, 8 linked notebook sources, 7 model sources**, 1 competition source. Must include `dangkhoa2016/tencent-wemm-embedding-4b/Transformers/default/1`. Existing prefit SVC and OLMo assets must remain.
- Keep author citations and all source notices in the PRIVATE Kaggle fork. No public code redistribution, no unlicensed claim, no source text, datasets, official competition files, weights or Kaggle secrets in GitHub.
- All execution must be on Kaggle, internet off, NvidiaTeslaT4, compatible with Kaggle's 9-hour hidden scoring limit (not yet verified for this candidate).

## No-training constraints

- **No new supervised model, Gaussian, LoRA, adapter, sibling-style LogisticRegression, or neural head fitting**; use fixed source artifacts. Original source contains two `model.fit` supervised sibling-style sites, which this candidate intentionally replaces with an identity residual. This is an ablation of the published style component, not an exact reproduction. Allow only unsupervised test-time TF-IDF prompt grouping. No paid API or official-data download to public GitHub.
- The builder must fail closed on changed **executable source digest**, changed input counts or representation, changed original training inventory, unknown training paths, or failure to install exactly **four safety interventions**: the `gauss.fit` block, Atlas training-mode block, core worker `--train-final` block, and identity bypass for grouped sibling-style supervised fitting.
- `--source-preflight`: only builds and validates private Notebook source in ephemeral runner temporary storage, no Kaggle GPU, no publish, no competition entry.
- `--private-preview-once`: the builder now forces the real ensemble branch instead of the known-answer preview fixture. The original destination already exists and remains protected against overwrite. No new GPU run or submission is authorized by source checks alone; the current session requires all-model, same-path, schema/ID/row-count and increasing-size inference evidence plus expected advantage. Existing submission `58244483` must not be duplicated.
- Champion `58210835` and MGT-v1 `58233227` must not be touched. The MGT-v1 public score is still unknown as of last authenticated check. No public LB weight tuning, arbitrary new blend, or duplicate submission.

## Decision boundary

A successful preview only proves Kaggle source compatibility and output format, **not** competitive quality on the ~4,000-row hidden run. Subsequent scientific promotion needs an independent same-row 18-label/source-provenance-qualified benchmark or separately authorized one-shot exploratory competition result, with the outcome labeled honestly. Public BA >=0.700 is a goal, not a prediction or achieved score.

Keep negative evidence; do not relaunch if preflight fails without first understanding its cause.
