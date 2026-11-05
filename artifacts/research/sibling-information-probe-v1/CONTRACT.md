# Frozen Sibling Information Probe — v1 (pre-result contract)

Date: 2026-11-05 MYT
Status: PRE-RESULT / DATA ADMISSION PENDING
Scope: independent research only. NO Kaggle submission, NO LLM fine-tuning, NO public training/validation data committed.

## Decision and scientific question

Question: Does an existing, externally trained Qwen3-8B LoRA24 attribution representation already encode transferable *within-provider sibling* signal that an ordinary 18-class linear readout fails to exploit?

This is a deliberately low-cost representation diagnostic, **not a claim of original invention**. Published V47 already uses family-conditional experts. Demonstrating a readout improvement would establish a narrow mechanism only, not novel architecture or a path beyond copied Public 0.666.

The 2026-11-05 official Kaggle score 0.666 is a reproduced external notebook; the independent own LoRA24 scored Public 0.520. The other agent's Ozertuu direct submission 58022340 is separate; no results from it may select this probe.

## Frozen control and treatment

Checkpoint: pre-official Stage-1 A1 Qwen3-8B LoRA24, frozen exact lineage from artifacts/candidates/two-stage-lora24-v1/RESULT.md. Do not use final official-trained Stage-2 weights for external transfer.

Feature extractor is identical on both arms: existing eight-layer normalized pooled LoRA representation. Feature matrix is extracted once and held frozen; neither arm changes LoRA/backbone weights.

- Control: class-balanced multinomial LogisticRegression, C=1, max_iter=1000, lbfgs, random_state=20261105, all 18 eligible model labels.
- Treatment: reuse control family probability mass. For each multi-label family fit one class-balanced multinomial LogisticRegression with the *same hyperparameters* on only that family's rows. Replace within-family normalized conditional probabilities while keeping the control's total family mass. Singleton providers remain unchanged.
- Negative control: deterministically shuffle training sibling labels *within their true family*, refit family heads, and apply the same mass-preserving composition. This checks whether the treatment gains are genuine sibling information rather than conditioning alone.
- Do not tune blend weights, per-label biases, temperature, thresholds, or family definitions.

Predefined provider families: gpt-, gemini-, claude-, glm-, qwen, kimi. Source and prompt are prohibited features.

## Admission: fail closed before scientific scoring

Readout training corpus must have frozen external-only embeddings, correct checkpoint lineage and all official 18 label names. The source manifest must prove where the model outputs came from, how the text was generated, and that output labels reflect the **served** model rather than only a requested API alias.

Holdout must be at least two entire unseen sources, no prompts/text hashes overlapping readout training, no holdout source within the **Stage-1 backbone-adaptation corpus**, and source family coverage sufficient to evaluate >=3 multi-label families with >=20 heldout rows per scored label. Two source names invented for the same upstream data do not establish independence. Do not use 2,699 official training texts, the three v1 Gate-A selection sources, v2 contrastive selection sources, v3 selected transfer sources, or Kaggle public leaderboard results as *fresh* holdouts.

Important newly surfaced provenance risk: Google Gemini API 3.7 Flash requests after 2026-10-08 can be routed to 3.8 Flash. Require returned served_model / modelVersion and generation date for such records; exclude or mark unknown if unavailable. Apply analogous served-version audit to other providers. Check text duplication / near duplication separately before opening the transfer gate.

All files must be content-hash pinned before loading holdout labels/predictions. If independent data or frozen embeddings are unavailable, status is DATA_BLOCKED; *do not fake a new holdout or silently evaluate spent data as fresh transfer*.

## Metrics and gate (frozen)

Primary: macro recall / balanced accuracy over heldout labels with >=20 examples and whose multi-label family is represented by at least two labels. Report all-available-label BA separately; never call a partial-label metric official 18-way BA.

Secondary: heldout source-level macro BA deltas; family accuracy and within-family conditional accuracy; per-label recall; short vs long response lengths if word_count is available. Bootstrap unit is entire prompt group, fixed 5,000 resamples and seed 20261105, dropping resamples that omit a required class.

PASS only if all hold:
1. treatment minus flat positive, with 95% paired prompt bootstrap lower bound strictly > 0;
2. at least two genuinely disjoint heldout sources have positive eligible-label BA delta;
3. the negative control is worse than treatment on the same eligible metric;
4. no eligible label whose control recall >=0.25 loses >0.15 absolute recall;
5. no provenance, leakage or official-data boundary fails.

FAIL if admitted real paired experiment violates one of these. BLOCKED if data/lineage is unavailable. Do not change gates after seeing scores. Failure does not prove the frozen LoRA encoder lacks sibling information; only rejects this conditional readout under these conditions.

Resource ceiling: one read-only feature-extraction pass using existing checkpoint, <=6 GPU hours; <=1 CPU hour for readout and analysis. No Kaggle competition submission.

## Execution boundary

A CPU-only evaluator and synthetic unit tests may be implemented on a dedicated research branch before data admission. Their passing is engineering evidence only. A separate future audited extractor must export non-official source embeddings to private storage with checksum; source data and embeddings must not be committed to public GitHub. Any readout evaluation on a spent source must be clearly marked DIAGNOSTIC and not used for promotion.

## Stop / pivot trigger

If no genuine external holdout can be authenticated, improve data provenance before training any model. If admitted and FAIL, investigate representation or data quality rather than sweeping family thresholds. Do not launch another public-reproduction submission based on these readout scores.
