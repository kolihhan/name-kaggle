# 2026-11-20 — Source-matched data hypothesis: real Phase15 CPU tests, live Open Pool and hard limits

## Objective and source
Follow-up to [Phase7→15 source diff](../phase7-phase15-source-diff-20261120/RESULT.md). The aim is to test the specific allegation that **Phase15 0.405322 official OOF vs 0.357 Public** was driven by direct near-duplicate/topic reuse from its external data, and to assess whether an existing public cohort of GLM/Kimi/Gemini examples provides **legally/source-admissible** candidates. This is **retrospective, adaptively spent historical official TRAIN**, not an independently calibrated hidden-test benchmark.

All results are authenticated Kaggle data downloads performed on GitHub-hosted **CPU**. **No GPU, no Kaggle submission, no training on newly added/unapproved 7-label pool data, no raw text or per-row prediction artifact export.** Two read-only runs and one bounded fixed retry used actual input snapshots. All reported effects below are historical data effects; no Kaggle Public improvement is claimed.

## 1 — Open Pool existence and current dataset snapshot

**Confirmed real** `crtproduction/name-that-llm-open-pool`, Kaggle dataset file `ntl_open_pool.parquet`.
- [2026-11-08 older, actual SHA-locked scan](https://github.com/kolihhan/name-kaggle/actions/runs/42477146980): 212,623 rows, 18 model classes, 60 source_dataset values, SHA-256 `1e4200aed80f52d655d4900a301cb427ca1341e06be31a8b7d5fde9ea3ef425f`. Already previously audited; there was no need to rediscover it from a README.
- [2026-11-20 current Kaggle API download](https://github.com/kolihhan/name-kaggle/actions/runs/44667515988): **231,036 rows**, **18 labels**, **66 source_dataset values**, file bytes **115,325,974**, SHA-256 `7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492`. `match_weight` is non-null on all 231,036, but not independently validated to measure host-distribution match. Search-engine third-party index reporting 254,561 passages was NOT the current downloaded file; prefer authenticated raw-file counts. Current metadata inventory job succeeded even though this combined multi-job run has a red overall badge due to the separately fixed nearest-neighbor script.

**All seven Phase15 external-unsupported labels exist in the current Open Pool**, but publisher-reported CC license strings are **not** original provider output rights; route/model revision and true prompt bank are not fully verifiable from row metadata.

## 2 — Nearest-neighbor hypothesis: size-matched exact original Phase15 corpus

Source: [successful corrected CPU Action #44667893709](https://github.com/kolihhan/name-kaggle/actions/runs/44667893709); [source code](../../scripts/ntl_phase15_size_matched_nearest_audit_20261120.py), aggregate JSON `ntl-phase15-near-matched-diagnostic-44667893709`.

- Official: 2,699 responses, 150 prompts, five disjoint official folds.
- Phase15 original **SHA-locked 75,783-row raw pool** mechanically rebuilt to **26,025 admitted external rows, 11 labels**, not the new 231k Open Pool.
- Unsupervised content-oriented **word unigram+bigram TF-IDF with English stopwords** fit on unlabeled pooled official+external text, cosine similarity of official validation responses to (a) all 26,025 external rows, (b) all ~2,160 official other-prompt fit rows, (c) **30 random external subsamples of exactly the same cardinality as the official fit** per fold. Same fixed features across comparisons. Never use a Public score or class prediction to tune the sampling.

| Statistic | All 26,025 external | Equal-size external (30 draws; per-row mean max) | Other official train prompts |
|---|---:|---:|---:|
| Median of per-response nearest cosine | **0.108914** | **0.079402** | **0.085954** |
| 90th percentile | 0.178236 | 0.113236 | 0.139005 |
| 99th percentile | 0.292990 | 0.152068 | 0.227722 |
| Fraction with cosine≥0.75 | **0/2,699** | **0/2,699** | **0/2,699** |
| Fraction closer than official training response | **74.435%** | **40.0889%** | Reference |

The prior [exact response hash / prompt-ID audit](../ntl-ruler-2699-prompt-oof-reliability-20261118.md) also found **0 normalized response exact external→official overlaps**, and no literal official prompt-ID collision.

**Interpretation:** raw 26k vs ~2k maximum-neighbor comparison is strongly distorted by candidate pool size. The apparent external-neighbor advantage **disappears under equal-size random sampling**. We find **no high lexical near-duplicate signal** under this chosen representation/threshold. This is a negative result for this *specific* leakage detector, **not proof of semantic prompt independence**: paraphrases, template/subject overlap, altered model serving and inaccurate source labels remain possible. Equal-sized external and official subsets still have different class/source compositions. No raw passage examples were published.

## 3 — Identical four-view SVC: source-only versus official-only

Source: [successful separate job in Action #44667515988](https://github.com/kolihhan/name-kaggle/actions/runs/44667515988/job/136060326847); [code](../../scripts/ntl_phase15_source_only_vs_official_4view_20261120.py), aggregate `ntl-phase15-source-only-aggregate-44667515988`.

| Training cohort | Validation/evaluation | 18-class BA | Important asymmetry |
|---|---|---:|---|
| **Official only** | real 5-fold GroupKFold by original 150 prompts, 18 classes | **0.374857** | every model class supported, same frozen four-view TF-IDF/LinearSVC |
| **Phase15 external only** | all 2,699 official responses completely excluded from this model's fit; only 11 external classes | **0.300890** | 7 missing class recalls forced to ZERO; covered 11-class macro recall **0.492365** |
| **Original Phase15 official + external** | original historical 5-fold grouped OOF, 18 classes | **0.405322** | existing published prediction/result, not re-fit here |

The **official-only → historical combined** result is +**0.030465** OOF under the same model *recipe and group folds*, but additional training examples can move all class boundaries; no new paired CI was computed here. Historical folds were reused, and the official hidden Public score of Phase15 remains 0.357. This is evidence that external content gives the classifier signals useful on **official TRAIN**, not that these signals transfer to hidden prompts/host version.

External-only on supported 11 labels has BA **0.492365**; the 18-class BA **0.300890** is mechanically bounded above by 11/18 due to missing labels. Do not conclude source-only is 0.074 worse in an apples-to-apples 18-class challenge.

Some source-only supported class recalls on official TRAIN:
- Claude Fable5: **0.402685** external-only vs **0.114094** official-only
- Claude Sonnet5: **0.686667** vs **0.426667**
- GPT5.6 Luna: **0.480** vs **0.313333**
- Qwen3.8 Max: **0.400** vs **0.493333** (external-only worse)

This pattern is consistent with *some* transferable signatures but no support for seven classes. It does not identify why the final combined Public drops below Phase7.

## 4 — Current 231k Pool: weak7 metadata-only source risk census

Source: [successful live metadata scan #44668450000](https://github.com/kolihhan/name-kaggle/actions/runs/44668450000); [script](../../scripts/ntl_open_pool_weak7_metadata_screen_20261120.py), aggregate `ntl-pool-weak7-metadata-only-44668450000`, SHA consistent with latest current snapshot.

The following last column counts **only** rows with nonempty publisher source-license metadata, a recorded `gen_served_model`, and **none of the existing heuristic risk flags**. **They are NOT certified correctly labeled, legal, independent or host-matched.**

| Missing label | Current public rows | Distinct publisher source names | Preliminary metadata-complete/unflagged |
|---|---:|---:|---:|
| Gemini 3.1 Pro Preview | 10,513 | 41 | 867 |
| Gemini 3.6 Flash | 18,447 | 48 | 1,834 |
| Gemini 3.7 Flash | 3,541 | 5 | 1,534 |
| Gemini 3.8 Flash | 23,542 | 54 | 2,929 |
| GLM 5.3 | 12,450 | 17 | 2,630 |
| GLM 5.3 Flash | 12,745 | 17 | 2,720 |
| Kimi K3 | 10,516 | 14 | 4,628 |
| **Sum** | **91,754** | not additive / sources overlap | **17,142** |

Among the source rows, metadata for **actual served model** is sparse: Gemini3.1 only 2,724/10,513, Gemini3.6 4,688/18,447, Gemini3.8 6,918/23,542, GLM5.3 2,630/12,450 and GLM Flash 2,720/12,745. Many large original source groups have **zero** `gen_served_model` fields. A flag count of 0 (e.g. Gemini3.7) means the publisher's heuristic did not flag the rows; it does NOT authenticate the model route.

Possible sources to manually inspect for original provider rights and true generation protocol (NOT automatically admitted): GLM `willrice/name-that-llm-openrouter-harvest`, `arxitekton/ntl-claude-gen-v1`; Kimi `willrice/name-that-llm-openrouter-harvest`, `arxitekton/ntl-claude-gen-v2`; Gemini3.7 `alperen5252525/ntl-vertex-gemini-proxy-gpt-generations`, `zeminglii/ntl-own-exact-route-generations`. Source display names are not evidence that all generated passages came from the named vendor, or correct original terms.

**Explicit compliance caution:** The repository's [prior Phase15 legal/provider-terms review](../../phase15/TERMS_AUDIT.md) intentionally excluded Gemini, GLM and Kimi-labelled *external* output for provider terms/unknown output rights. Public availability, a CC BY/CC BY-SA metadata string from the dataset publisher, and the Kaggle competition's external-data public/free condition are three different issues. A source licence claim does not automatically supersede the original provider's terms. Neither the current dataset nor this audit independently clears that barrier. **DO NOT TRAIN a new Phase14/15 challenger on these seven categories yet.**

## Assessment and decision

**Tested and unsupported:** the strong claim that simple lexical near-duplicate contamination of Phase15 external data explains its OOF/Public reversal. No lexical high-sim matches found; apparent nearest similarity advantage mostly reduces to unequal candidate pool sizes.

**Observed:** original legal-filtered external-only data has clear measurable signal on the known 11 supported labels. Official-only 4-view SVC 0.374857; Phase15 combined historical 0.405322, but 0.048322 OOF−Public gap remains unexplained. The top priority shifts from repeated old-fold model tuning toward identifying **source/prompt provenance and served-model authenticity**, with independent adjudication rather than assuming any pool is host-matched.

**Blocked not scientifically rejected:** adding weak7 pool data to the same SVM, because original rights and served-model identity are unverified and earlier provider-terms exclusion is binding. First verify original source READMEs, license/output-right chain, serving route/model version and independence. If a legal, source-distinct weak7 subset is proven, pre-register an equal-recipe C0/C1 test with source-separated heldout prompts *before* using official OOF again, and keep exactly one candidate after that.

No new Public BA, no champion change (0.702), no new Kaggle submission. No GPU.
