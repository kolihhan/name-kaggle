# CURRENT NTL — 2026-11-19 13:31 MYT

**Official Public BA remains 0.702**, #58514818; target >=0.720 (+0.018). No new GPU, fitting or competition submission this round.

## Decision: reject the free 363-row GPT proxy as NEW data
- [Authoritative GitHub CPU proof #44465398268](https://github.com/kolihhan/name-kaggle/actions/runs/44465398268), matched **pinned** proxy CSV SHA256 a8f61b647b5f2b2f99b73f083b7358445d93c3ebbff8607b44423850e9fb5583 to **pinned** current OpenPool SHA256 7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492.
- **355/363 qualified GPT outputs (97.8%) are normalized-exact text duplicates already in the current OpenPool**; same 355 are >=0.90 char-5gram cosine. Only **8/363** not exact near matches above cosine .80 in the OpenPool.
- Paired structure: 145 distinct Prompt IDs; **36 complete five-GPT same-prompt groups (180 rows)**; 72 single-label groups, 11 two-label, 15 three-label, 11 four-label; zero duplicate prompt-model pairs. Pairing exists, but responses are overwhelmingly not new.
- Source publisher states the prompt bank comes from `seyitkaangunes/ntl-extra-api-passages` and `prvsiyan/prvsiyans-extended-name-that-llm-dataset`—**both already mounted in the 0.702 champion**. Actual prefit prompt exposure isn't proven from mounts alone.
- `request` is plain text, **0 JSON-parsable request objects**, hence no independently authenticated actual served-response model ID. The CPU attempt's **0 prompt text hashes compared** cannot be interpreted as zero prompt overlap. The exact response collisions alone reject the novel-data hypothesis.
- **Scientific conclusion:** The previous "not listed among 66 source names" finding only proves the published source slug was absent; it does **not** establish disjointness. The 363-row proxy is not independent, not a clean holdout, not a new full-stack dataset, and not a reason to spend more NTL training budget. Do not perform further tests on this source. Next work must begin with source/data novelty verification rather than labels/feature tuning.

---

## 2026-11-19 11:33 MYT — Five-class GPT source candidate found; no score-facing promotion

**Actual official NTL best remains 0.702**, Kaggle #58514818, target >=0.720 (gap +0.018). **No training, Kaggle GPU, or new official submission this round.**

- Exact Nov17 OpenPool trainer version SHA `5a3dc82288328abedf49b7dd5e866e5c69eb922f9d6008e6fd78cf5153c12332` vs latest SHA `7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492`. Same **8,338** strictly source/license/protocol/length/risk-filtered exact-id-matching rows across **13/18** labels, and **zero qualified** for `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`. Latest raw 231,036-row OpenPool has 17/18 unfiltered exact labels, **not training-eligible 17/18**. [Authenticated CPU metadata proof #44451612939](https://github.com/kolihhan/name-kaggle/actions/runs/44451612939). New dataset version **not admitted to training**.
- Found new externally published [Kaggle Benchmarks free proxy dataset](https://www.kaggle.com/zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations), CC BY 4.0 per author, file SHA `a8f61b647b5f2b2f99b73f083b7358445d93c3ebbff8607b44423850e9fb5583`. Read-only [source coverage #44451906235](https://github.com/kolihhan/name-kaggle/actions/runs/44451906235): 834 rows incl **430 across all missing GPT labels** (gpt5.5 81 / Luna 79 / Sol 73 / Terra 64 / Astra 133).
- [Strict 5-class format/provider metadata audit #44452104227](https://github.com/kolihhan/name-kaggle/actions/runs/44452104227): 363 GPT examples after daily-run, reasoning-low/temperature1, 40–250 words, one paragraph, normal stop filter: **68 / 64 / 58 / 53 / 120** by label; **145 prompt IDs** from only one prompt bank. The exact dataset slug is **not listed among OpenPool's 66 source names**, but a later normalized-text check established **355/363 identical responses already in OpenPool**; this source is **not novel**. **Model IDs are request-side Kaggle proxy slugs, not independently recorded returned backend IDs**; author-reported labels only. Do not call 363 independently validated true identities, nor complete 18-class transfer evidence.
- Existing Nov17 ModernBERT adversarial Colab [Action #44116912517](https://github.com/kolihhan/name-kaggle/actions/runs/44116912517) trained real C0/C1 under fixed 250-step source-disjoint contract: **BA 0.214444 → 0.054444, Δ−0.160000**, 82 rescues / 514 regressions; mechanism **REJECTED**. Do not rerun.

**Research decision superseded by newest overlap audit above.** H_data_gap (OpenAI sibling training has no qualified independently verified served-ID examples in existing pooled data) remains plausible, but the free 363-row resource is **REJECTED as novel/independent data**. Prefer verifying bank/identity independence and a same-row scored-strong control rather than running a weak 5-class proxy, training with unapproved provenance, or another Kaggle submission. No new program pipeline or Model/GPU expenses.

---

## 2026-11-19 — Strong-stack comparison: NO NEW PROMOTION

- **Official Kaggle champion 0.702 #58514818** (+0.011 vs old .691). Target >=0.720; gap +0.018. No GPU, no official submission in this round.
- [Actions #44448660771](https://github.com/kolihhan/name-kaggle/actions/runs/44448660771): exact source and model-asset comparison of 5 public alternatives to our scored Apex 0.702. All use identical 9 model, 26 dataset and 9 upstream Notebook mounts. Soft Typography and Alexchilton differ in only 1 of 20 shared helper modules; **reject as redundant**.
- Areebbb adds spec_worker.py, Ozertuu adds a 95-line sibling residual head, Tensura replaces several inference/stage modules and adds 934-line comp_stack.py. [Actions #44448962381](https://github.com/kolihhan/name-kaggle/actions/runs/44448962381) confirmed signatures: each has fit-related pathways but static presence does NOT prove executed training. No validated same-row evidence they outperform 0.702, and their existing model/data inputs do not add independent author-generation information.
- **Hypotheses:** H_A new attribution information vs H_B reaggregation/supervised re-fit of the same prefit experts. Available evidence favors H_B as source-level explanation, **not a scored effectiveness test**. No new qualified strong-stack candidate. Preserve champion. Do not spend Kaggle on same-source mixture variants.
- **Single live requirement before next GPU:** distinct, legal, original-served-ID-backed data/pretrained expert or clean same-row comparison to 0.702 with an independent mechanism. Do not return to used weak proxy cohorts.

---

# NOW — Name That LLM
2026-11-19 10:46 MYT · AUTHENTICATED NEW CHAMPION

## Verified official result

- **Best Kaggle Public Balanced Accuracy: 0.702**, submission **#58514818**, scored 2026-11-18. Former #58375806 **0.691**. **Official +0.011 BA (+1.1 percentage points)**. New score confirmed by authenticated [read-only Kaggle ledger GitHub Action #44446729620](https://github.com/kolihhan/name-kaggle/actions/runs/44446729620) on 2026-11-19 02:46 UTC: scored_total 18, top submissions #58514818 0.702 / #58375806 0.691 / #58295802 0.683; no pending; quota_available 5 at that snapshot. Private BA remains unknown.
- Scored candidate = **attributed reproduction** of [Lightning Apex Style 0.702](https://www.kaggle.com/code/lightningv08/0-702-apex-style-15-glm-qwen3-8-27b-lora), **not our original model improvement** or guaranteed Private medal. Source SHA256 `f9948b633b22bd5e63b8253e4d8027c073b44f57eb85bc60b775857e396b38c2`; code SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`. No supervised fitting or optimizer steps added, original code untouched.
- The one owner-authorized Kaggle T4 run completed correctly. [Action #44321758700](https://github.com/kolihhan/name-kaggle/actions/runs/44321758700) SUCCESS: private Notebook [version 2](https://www.kaggle.com/code/kolihhan/ntl-0702-apex-pinned-nov18) COMPLETE with `submission.parquet`, accepted official submission ref **58514818**. Version 1 was Quick Save only. No second GPU run or duplicate official entry.

## One next step
**Goal official BA >=0.720 (remaining +0.018).** Pursue ONE **materially independent strong full-stack candidate** with verified source/valid offline assets and distinct representation/data signal; no 0.702 style weight tuning or weak proxy tests. Compare strong-stack overlap and code integrity first. Do not run another GPU experiment without a distinct score-facing hypothesis and permitted Kaggle quota. No claim of medal from Public alone.

## Historic detail
- [Pre-result handoff](https://github.com/kolihhan/name-kaggle/blob/1b5ca58ee9f1a4ac83a0e7d5ec7d847c599d3633/NOW.md)
- [Research decisions](BREAKTHROUGH_PLAN.md) · [Rules](RULES.md)
