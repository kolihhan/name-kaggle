## Completed exact current 0.691 model error distribution (Oleg216, spent domain)

[Read-only existing-predictions Action #44119966899](https://github.com/kolihhan/name-kaggle/actions/runs/44119966899) **COMPLETED/SUCCESS**. Parsed the exact old private `kolihhan/ntl-oleg18-frozen0691-probe-nov16` prediction parquet, matched every one of 216 IDs to SHA-pinned 18-label Oleg truth without new inference, required the historical **89/216 correct** digest, and exported only an aggregate 18×18 confusion metadata artifact.

- Correct **89/216 = 0.412037**, errors **127**.
- Within-provider wrong **64/127 = 50.39%**; cross-provider wrong **63/127 = 49.61%**.
- Largest observed confusion: GPT-5.6 Sol → Luna **8**; GPT-5.6 Terra → Luna **7**; Gemini-3.8 Flash → Claude Opus 4.6 **7**; Claude Fable 5 → Opus 4.6 **7**. The last two are **cross-provider**.
- Compared to the older LoRA24 Stage-2 **701/1223 = 57.32% within-provider OOF**, this is a different model and a different domain; the two fractions are **not a causal paired contrast**.
- Hence a **Sibling Expert is not established as the dominant bottleneck of the current 0.691 candidate**. External Oleg has only 12 examples/class, and prefit overlap is unproven. Leave the hypothesis open without preemptively training or promoting a sibling intervention.
- [Live leaderboard full-pagination attempt #44121067184](https://github.com/kolihhan/name-kaggle/actions/runs/44121067184) returned no parseable CSV header; **no rank-percentile/medal score is established**, and 0.700 is **not a medal cutoff**. The older [authenticated Top-20 snapshot](ntl-public-leaderboard-top20-20261112.md) is dated November 12 (leader 0.765; #20 0.727) and is not the present or final Private board.

---

# NTL validation-scale audit — 2026-11-17

## What the historic data do and do not establish

| Candidate | Historical grouped OOF BA | Own authenticated Public BA | Lineage / comparability |
|---|---:|---:|---|
| Own Phase 7 gated specialist | 0.370052 | 0.369 / #57838845 | Genuine grouped OOF, official score; repeatedly examined historical fold |
| Own Phase 15 terms-clean | 0.405322 | 0.357 / #57861337 | Same original 2,699 official grouped rows as Phase 7; Public **reversed** the OOF preference |
| Own two-stage LoRA24 v1 | 0.546761 (Stage-2 plain OOF) | 0.520 / #57908683 | Own development stage OOF; different experimental lineage, NOT an independent transfer reference |
| Own CDA LoRA24 v2 | N/A (no attributable comparable same-model OOF here) | 0.530 / #58150689 | Separate training submission; do not substitute external Gate-A as official grouped OOF |
| Public-sourced Ozertuu reproduced | N/A (frozen public/pretrained system) | 0.668 | Attribution to external author; no genuine paired official OOF in this repo |
| Later externally sourced reproduction | N/A | 0.679 / #58210835 | Attribution to external author; no matched official OOF |
| Areeb8 frozen external pretrained | N/A | 0.683 / #58295802 | Inference-only public source; no matched official OOF |
| Current Lightning frozen 0.691 | N/A | **0.691 / #58375806** | Source SHA pinned; **no comparable official grouped 2,699-row OOF** with same candidate is verified |

Only Phase 7 vs Phase 15 form a directly matched historical grouped-OOF comparison: Phase 15's **OOF +0.035270** became **Public −0.012**. Other rows have different data representations, incomplete diagnostics, or are attributed public systems. **It is unjustified to compute a global OOF↔LB correlation from this table.**

Sources:
- [Phase7](../phase7/RESULT.md), [Phase15](../phase15/RESULT.md)
- [Original LoRA24 model & stage result](../candidates/two-stage-lora24-v1/RESULT.md)
- [CDA authentic score](../../NOW.md) / [current Lightning authentic score](../../NOW.md)
- [Areeb8 authenticated Public receipt](../candidates/areeb8-prefit-only-20261113/OFFICIAL_SCORE.md)

## Why the new external denominators differ

The metadata gate retained **45,598** rows from **13** asserted-good publisher sources (10 prospective train / 3 heldout), with **28,283** potential train and **17,315** potential heldout. This is **not** an independently served-model-authenticated benchmark.

A subsequent, stricter frozen training-prep script required `n_words` 40–250, original prompt/response hash disjointness and the same fixed heldout-source hash. Before the final sample limit it retained **12,961** train / **17,048** validation from **8** training / **3** holdout sources, and **44,900** pool rows eligible by that particular recipe. Then source-round-robin per-model sampling with **at most 360 training and 150 validation rows per class** produced **5,562 train / 2,700 validation**. This is a *nested source-quality/sampling transformation*, not a mislabeled repeat of the original full gate. Logs: [real T4 run #44116912517](https://github.com/kolihhan/name-kaggle/actions/runs/44116912517), [strict source gate #44109672785](https://github.com/kolihhan/name-kaggle/actions/runs/44109672785).

The broad 45,598 set contained **8,475** row-level exact-string served-ID matches across only 13 model classes; after the stricter training-prep recipe the corresponding publisher row-level match count was **8,338** across 13 classes. These are not independent API attestations. The larger source protocol's `served_model_check=yes` is a publisher claim, not verified same-version model ground truth.

## Current 0.691 champion failure hypothesis is not yet verified on competition-domain rows

A former **self-trained LoRA24 Stage-2** model had 1,223/2,699 wrong official OOF predictions, 701/1,223 (57.32%) within-provider sibling confusions, computed in [archived raw-OOF read-only audit](ntl-sibling-confusion-pivot-2026-11-04.md). **This is not the 0.691 Lightning model.** Do not promote Sibling Expert, label-targeted thresholds or features from this old confusion matrix.

The authentic 0.691 frozen champion has an existing **216-row Oleg18** private-blob inference panel, 12 rows/class and 89/216 correct, but this spent domain may differ materially from competition Public BA and is not a fresh test. New [current-model read-only confusion calculation](https://github.com/kolihhan/name-kaggle/actions/runs/44119966899) aims to quantify per-class/provider errors from that exact existing 0.691 prediction set, explicitly without new GPU/model runs. Even if successful, its family confusion is **diagnostic, not trustworthy hidden-test inference**.

## Actual new-training decision

Colab T4 [matched 250+250 optimizer-step real training run #44116912517](https://github.com/kolihhan/name-kaggle/actions/runs/44116912517) completed after owner refreshed Google ADC:
- C0 ModernBERT external source-disjoint 18-label BA **0.214444**
- C1 source-adversarial ModernBERT same-row BA **0.054444**
- Δ **−0.160000**, 82 rescues / 514 regressions, three holdout source deltas all negative; **REJECT this mechanism**. No promoted checkpoint, no official NTL submission.
- Prior char TF-IDF C0 **0.213704** / source-balanced C1 **0.177037**; frozen MiniLM **0.107407** on same external holdout. These describe weak transfer on this particular heterogeneous corpus, not expected Public 0.691 performance.

## Evaluation and submission rule

1. Before any new NTL candidate/submit: verify exact model and requested-vs-served revisions, licensed source, prompt/source disjointness; retain all 18 labels and actual paired predictions from **current champion** on the exact same rows.
2. Freeze source-independent and official prompt-grouped evaluation and effective independent unit count. Historical OOF has already been adaptively mined. Treat a single novel holdout as diagnostic until calibrated against the frozen champion and a second independent source/domain.
3. Do not claim positive Kaggle or medal gain from external BA, from a green training job, or from `0.700` numerical goal. Medal rank is defined by final Private placement, not a static score.
4. Do not issue speculative leaderboard submissions or retrain rejected adversarial same mechanism.
