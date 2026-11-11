# Dual-Embedding + Tencent WeMM — frozen inference-only candidate

**Frozen at 2026-11-12 MYT, before any personal Kaggle score.**
**Status:** SOURCE PREFLIGHT ONLY; no new official competition submission is authorized under this contract.

## Goal and model-class hypothesis

Our currently verified complete Public BA is **0.679**, `58210835`, from an attributed public Ozertuu reproduction. A new public dual-embedding lineage adds a **Tencent WeMM Embedding 4B** model to Qwen3 Embedding / multi-backbone prefit LoRA / pretrained sparse and other ensemble members. An independent embedding space might distinguish near-sibling generator styles missed by the incumbent. **No measured gain or complementarity is established.** The hypothesis could fail if it is a correlated embedding, same-source overfit, or distorts the 18-way balance.

This is a **new model-input/representation candidate**, not a minor Sinkhorn hyperparameter patch and not an original pretrained architecture. It is **not** the private 0.765 leaderboard leader's unreleased solution.

## Immutable source and permissions

- Canonical public Notebook source: `ozertuu/llm-dual-embedding-lora-qwen3-8-27b-gsq` by Ertuğrul Özer.
- Public Notebook exact downloaded SHA256: `e943174f3cfe114eeb96d9f721c675826f6ef8a91e30ae055dd86e5c48257865`, authenticated public Kaggle read-only pull [#43056843211](https://github.com/kolihhan/name-kaggle/actions/runs/43056843211).
- Executable cell SHA256: `a65e0115aaa56e3ce64391e9911fed7d3521c7ace3a1e990d92cc4f1b8a03a82`, lineage [#43057087250](https://github.com/kolihhan/name-kaggle/actions/runs/43057087250); exactly identical to `omkarpattanaik123/name-that-llm-v28-sovereign-titan-apex`. Do not mistake this fork for a separate improvement.
- Input manifest: exactly **21 dataset sources, 8 linked notebook sources, 7 model sources**, 1 competition source. Must include `dangkhoa2016/tencent-wemm-embedding-4b/Transformers/default/1`. Existing prefit SVC and OLMo assets must remain.
- Keep author citations and all source notices in the PRIVATE Kaggle fork. No public code redistribution, no unlicensed claim, no source text, datasets, official competition files, weights or Kaggle secrets in GitHub.
- All execution must be on Kaggle, internet off, NvidiaTeslaT4, compatible with Kaggle's 9-hour hidden scoring limit (not yet verified for this candidate).

## No-training constraints

- **No new supervised model, Gaussian, LoRA, adapter, or neural head fitting**; use fixed source artifacts. Allow only original unsupervised test-time TF-IDF prompt grouping. No paid API, no official data download to public GitHub.
- The builder must fail closed on changed upstream source hash, input count or representation, changed expected fit inventory, unknown train paths, or failure to install exactly **three guards**: `gauss.fit` block; Atlas training-mode block; core worker `--train-final` block.
- `--source-preflight`: only builds and validates private Notebook source in ephemeral runner temporary storage, no Kaggle GPU, no publish, no competition entry.
- `--private-preview-once`: if source-preflight passes and destination doesn't exist, create only one PRIVATE version at `kolihhan/ntl-dual-embedding-wemm-frozen-v1`, check Notebook 3-row preview and output. Never submit to official competition in this command.
- Champion `58210835` and MGT-v1 `58233227` must not be touched. The MGT-v1 public score is still unknown as of last authenticated check. No public LB weight tuning, arbitrary new blend, or duplicate submission.

## Decision boundary

A successful preview only proves Kaggle source compatibility and output format, **not** competitive quality on the ~4,000-row hidden run. Subsequent scientific promotion needs an independent same-row 18-label/source-provenance-qualified benchmark or separately authorized one-shot exploratory competition result, with the outcome labeled honestly. Public BA >=0.700 is a goal, not a prediction or achieved score.

Keep negative evidence; do not relaunch if preflight fails without first understanding its cause.
