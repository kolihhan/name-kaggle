# NTL 2026-11-23 — Public high-rank notebook mechanism census (25 real pulled sources)

## Scope and evidence contract

- [Read-only verified GitHub Actions #45142007503](https://github.com/kolihhan/name-kaggle/actions/runs/45142007503), using authenticated Kaggle API, `kaggle kernels list --competition name-that-llm --sort-by scoreDescending --page 1 --page-size 50 --csv`, then pull and inspect the **first 25** refs. Script: [`scripts/ntl_public_top25_mechanism_census_20261123.py`](../../scripts/ntl_public_top25_mechanism_census_20261123.py).
- Ranking response: 50 public notebooks but only CSV fields `ref,title,author,lastRunTime,totalVotes`; it **omits numeric score**. Thus this is **a sample from public source discovery sorted by Kaggle's requested scoreDescending option, not a verified numeric top-25 standings**, and certainly not techniques of the private-code top-10 teams.
- Completed pulls: **25/25 ranked** plus 3 exact additional comparisons (Estopadilla pinned V5, Sanpier pinned V3, own pinned 0.715), **28/28 total**. No source exceptions.
- Own 0.715 baseline scored-source SHA `202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8` verified before comparison in same run. Public notebook version metadata is not returned; explicit URLs/Vn requested but no claim of cryptographically certified version response.
- This is a code-presence and imported-asset survey, NOT evidence each code path executes at official hidden rerun. Notebook title scores are **claims**, not verified current official receipts.
- No training, GPU, runtime inference, model/dataset weight download, competition data download, test predictions, or submission. Public source transient; no notebook source copied to the repository.

## Actual quantitative observations

Denominator **25 pulled public high-ranked notebooks**:

| Property | Count | Interpretation |
|---|---:|---|
| **9 model mounts** | **25/25** | All use nine pre-existing model resource attachments; number of mounts is NOT nine independent classifier heads or an inferred paid-API cost |
| **Identical complete input manifest to own scored 0.715** | **24/25** | Same sorted model, dataset and kernel attachments. Remaining one differs at least one mount; earlier direct audit showed Estopadilla public V5 has 11 kernel sources vs own 10, while model source count remains 9 |
| **Nine common core scoring modules exactly identical to owner** | **17/25** | These notebooks have nine exact common-module body hashes; outer notebook/orchestration changes may still matter |
| **Distinct full notebook code SHA** | **20/25** | Several notebooks are byte-identical in code; distinct full-code SHA is NOT proof of distinct scoring algorithm |
| Pretrained/externally bundled multi-expert inference code | **25/25** | Core scoring sources such as `infer.py`, `atlas_infer.py`, `native_q8.py`, `comp_stack.py` |
| GLM causal likelihood code | **25/25** | `v1_stage`, `glm53_scores`, `glm53-segs`, `ll_model`, etc. |
| Stylometry features | **25/25** | `ntl660_readout`, n-grams, style/shape/punctuation evidence |
| Family-aware residual code | **25/25** | `pngy_residual.py`/family normalization |
| Soft prompt group inference | **25/25** | `gauss.py` and covariance/grouping infrastructure |
| Prefix/Fable routing marker | **22/25** | `prefix_decisions`, `fable_return`, `prefix_variant_stage` |
| Hungarian/hard matching marker | **2/25** | May be diagnostic/unused code; *not* proof hidden inference uses hard global assignment |
| `.fit(` marker | **25/25** | This can be small style/calibration head fitted during notebook execution; **NOT** evidence 25/25 fine-tune a foundation model |
| EDA/training markers | **25/25** | Not meaningful for detecting separate EDA-only methods because shared bundled code contains train-side data use |

**Key sample bias:** The high-ranked public list is dominated by a common strong upstream lineage; that is the measured phenomenon. Do not extrapolate these percentages to all independent high-performing entries, all paid data collection, or private-code 0.76–0.80 leaders.

## Notebook exemplars / source genealogy

- [Areeb — 0.719 Preserve GLM-to-Fable Corrections](https://www.kaggle.com/code/areebbb/0-719-preserve-glm-to-fable-corrections): 0.719 is notebook title; code confirms shared inherited backbone plus `prefix_decisions.py:fable_return`, protect GLM→Fable route. In top 25 has same 9 core modules and same input mounts as 0.715.
- [Dennis — GLM User Prefix Guarded Fable Return](https://www.kaggle.com/code/dennis6886/glm-user-prefix-guarded-fable-return): explicit GLM user-prefix variant stage, compute budgeting, protected Fable return, core nine modules exact inherited. New inference-orchestration logic, not independently trained new general 18-class backbone.
- [Asterios — 0.718 Error Map and Ideas](https://www.kaggle.com/code/asteriospanoras/0-718-which-llms-look-alike-error-map-and-ideas): descriptive analysis cells plus shared highscore scoring stack; 8/9 inherited core modules, identical input mounts; code marker hard matching could be in EDA; score belongs to actual scoring code, not necessarily author's added analysis. Notebook title != independently verified score.
- [Sanpier — 320-Token GLM Likelihood + Qwen Ensemble](https://www.kaggle.com/code/sanpier/0-709-320-token-glm-likelihood-qwen-ensemble): Kaggle publicly displays **Best Score 0.717 V3**. Requested V3 source SHA differs from latest (previous [Actions #45136005059](https://github.com/kolihhan/name-kaggle/actions/runs/45136005059)); includes own GLM segmented likelihood processing, budget and GLM/Fable evidence protection, same nine model mounts, 4/8 common core module hashes matching own. Title's `320` does not prove number of observed suffix tokens per scored passage; actual source uses `--max-text-tokens 160` in examined cells.
- [Lightningv08 — 0.717 User-Prefix GLM + Centering + Qwen3.8-27B](https://www.kaggle.com/code/lightningv08/0-717-user-prefix-glm-centering-qwen3-8-27b): 7/8 common scoring module hashes exact; same input mounts; GLM variants and soft grouping, not a new standalone fingerprint mechanism verified here.
- [Estopadilla — 0.716 Robust GLM Likelihood User Prefix](https://www.kaggle.com/code/estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix): its requested public V5 code and our **verified official 0.715** code were identical SHA `202bf1bf...` in previous read-only audit. Numerical 0.716 is title; do not treat own 0.715 clone as a separately invented model.
- [Evgen — Deep Forensic EDA](https://www.kaggle.com/code/evgendvorkin/name-that-llm-deep-forensic-eda-with-analogie): additional prompt/style exploratory EDA, downstream inherited strong inference code; 9/9 common core exact. EDA alone does not certify incremental accuracy.
- [Ranjeet — Sibling Odds](https://www.kaggle.com/code/ranjeet258/name-that-llm-sibling-odds): explicit additional sibling-agreement final stage; 8/9 common modules matching own; additional code does not have verified same-row rescues.

## Leading team vs public-code access gap

- Public competition [leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard) uses only approximately 40% of hidden test, the private 60% may reorder winners.
- [External 2026-11-22 board snapshot](https://autoresearch-arena.github.io/contests/name-that-llm/) reports team best **0.807** and publicly shared notebook ceiling around **0.719**, with title scores not independent score receipts. The top team's architecture is **not public in this audit**. Do not say private leaders rely on paid API data or QLoRA; those are speculations. The contest offline rerun [requires no Internet and <=9h](https://www.kaggle.com/competitions/name-that-llm/overview/description), so external API queries at scoring are not the direct method. Extra training data may be collected before submission, but costs/methods cannot be inferred.
- The code sample's single-lineage dominance is not proof that other hidden methods do not work. Nor does a tiny 0.715–0.719 Public change prove generalization or Public overfitting.

## Our proposed Generator Fingerprint — adversarial comparison

| Previously suggested element | Exists in 25-source public-code family? | Real differentiation |
|---|---|---|
| word choice, char n-grams, punctuation, sentence length, formatting fingerprints | **Yes**, stylometry 25/25, much of it in our 0.715 | **None.** Repackaging is not original research |
| existing contextual embeddings/LoRA or pretrained strong backbone representation | **Yes**, multiple expert scoring code in 25/25 | **None** from merely calling embeddings “fingerprints” |
| family-specific classification, soft grouping, output postprocessing | **Yes**, 25/25 | **None** |
| model-specific conditional next-token likelihood | **Yes for GLM in 25/25**, specialized segments | A new all-class likelihood route would be much more expensive; not appropriate to no extra paid API/budget requirement, and no independent availability guarantee |
| **Prompt/genre/source-disjoint stable higher-order response/decision dynamics**, controlling for topic and format with a frozen common representation | Existing features and soft grouping partially overlap. **Not validated** as an independent 18-class information source from this static census | **Research question only**, not novelty claim. Must prove extra same-row signal beyond 0.715, source-disjoint and at fixed cost |

**Negative prior experiments in own repo — do NOT omit:**
- GTF-v1 simple class-conditioned transition model: **0.1892916** vs unigram **0.2444743** (same historical prompt-heldout TRAIN OOF): transition representation **worse**. [Results](../../docs/research/ntl-generative-transition-v1-result-20261123.md).
- Frozen pretrained token-head GCTH-v1 **0.1826547**, independent 4-view sparse **0.3670892**; a weak isolated generative score is not competitive. [Results](../../docs/research/ntl-gcth-v1-result-20261123.md).
- MPCD-v1 matched-prompt column CE **0.3014815** vs row CE **0.3007407**, no meaningful robust improvement. [Results](../../docs/research/ntl-mpcd-v1-result-20261123.md).
- These are historical OOF against their own controls, **not** apples-to-apples numbers to Public 0.715; nonetheless falsify simplistic “token transition fingerprint already works” and “more paired prompts necessarily helps” narratives.

## Research verdict and next action

**The earlier 'Generator Fingerprint' hypothesis as phrased is not a differentiated algorithm; it mostly duplicates the 25-source public highscore codebase and own 0.715.** Do NOT start another small-model style classifier, token bigram model, QLoRA or paid target-model API calls simply to prove it.

The only potentially novel question is to find genuinely **independent**, **source-invariant** author signal in higher-order response structure that existing strong embeddings/stylometry do not already capture, **and** demonstrate it on same-row matched source/prompt-disjoint evidence against a frozen strong baseline within the same inference cost. At present that is a falsifiable research question, not a validated approach. Design a strict cheap availability/predictive-increment gate first; if original 0.715 final scores on independent rows are unavailable, report **BLOCKED**, not “promising breakthrough.”

**This turn's concrete result: the 25-notebook source-code census succeeded; no model trained, no new Public BA, no official submission, no paid API calls.** Owner verified best remains **0.715**.
