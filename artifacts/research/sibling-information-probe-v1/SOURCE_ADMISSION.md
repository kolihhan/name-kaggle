# Independent source admission reconnaissance — 2026-11-05

Status: NO INDEPENDENT HOLDOUT ADMITTED. No raw rows, source hashes, served-model responses, LoRA embeddings, or full provenance were downloaded or scored. These are candidate sources, not available evaluation data.

This reconnaissance is driven by the frozen CONTRACT.md. In particular, the external Stage-1 A1 LoRA24 checkpoint has already seen much of the public corpus, and some archived source splits have already been used for candidate selection. A new dataset name is not itself an independent source.

## Candidate leads

1. [olegzholobov/name-that-llm-response-corpus](https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus): public discussion mentions an 18-model-wide response corpus. Candidate for a source-level audit only. Row counts, exact generation dates, upstream source identities, served-model IDs, overlap with Stage-1 pool, and licenses are **not verified**. Do not admit without raw source-level reconstruction.
2. [Name That LLM Open Pool: 255k Passages, 18 LLMs](https://papersdatacode.com/datasets/43ef5b24-name-that-llm-open-pool-255k-passages-18-llms): third-party listing claims 254,561 deduplicated passages merged from 59 open datasets. This may be substantially overlapping with existing Stage-1 inputs. The aggregate must NOT be treated as one independent evaluation source; inspect original 59 sources and provenance first.
3. [maniceet/name-that-llm-openrouter-openai-generations](https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations): dataset card claims 7,753 rows / 10 labels, includes model/request routing and served_model fields (some blank). This source family was already used in v2 or other project audits; it **cannot** be counted as new independent transfer evidence without demonstrating unseen upstream partitions. The available labels do not independently cover three multiple-sibling families.
4. [zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations](https://www.kaggle.com/zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations): 834 GPT/Gemini entries, protocol details from public dataset card. Zakir was already a Stage-1 v1 heldout selection source; not fresh for this experiment, and only two multi-label families.
5. [Crownelius/GPT-5.6-Sol-Luna-Terra-Traces](https://huggingface.co/datasets/Crownelius/GPT-5.6-Sol-Luna-Terra-Traces): public source reports Sol/Luna traces, zero Terra. Different length/task protocols and inadequate model-family coverage. Not suitable as an admitted holdout.

## Generation-label provenance risk

Google Developers Forum staff confirmed that from 2026-10-08 the Google Gemini API endpoint named gemini-3.7-flash routes requests to gemini-3.8-flash. See https://discuss.ai.google.dev/t/gemini-3-7-flash-requests-are-answered-as-gemini-3-8-flash/187817 and the Google API change log. This does not prove all OpenRouter/private upstream provider paths behave the same way.

An external record carrying requested_model=gemini-3.7-flash alone is therefore insufficient to certify generator identity when served after that date; require timestamp, API route and served_model/modelVersion. This is especially material for a sibling-discrimination experiment: wrong labels can manufacture an apparent irreducible sibling-confusion ceiling.

## Next executable step and decision boundary

- Acquire exact versioned source manifests and raw record metadata for each *underlying* independent dataset; compare upstream IDs, prompt hashes, text fingerprints and source timing against Stage-1 train and all historically spent validation sources.
- Admit at least **two** truly independent heldout source lineages with >=3 multi-sibling families; verify legal use and model-serving IDs. Freeze canonical source IDs and file hashes before any model inference.
- Extract Stage-1 A1 frozen LoRA features once for private, audited fit and holdout data. Save embeddings privately, never commit to public GitHub.
- Only then run scripts/run_sibling_information_probe.py against a content-hash lock and issue an honest PASS/FAIL.

This is **DATA_BLOCKED**, not an ML failure. The CPU synthetic test result verifies code behavior only. Do not fill the missing data gap by recreating fake sources, resplitting the old 2,699 official rows, or claiming 0.666 external model predictions as independent component evidence.

---

## Executed source verification (2026-11-06 MYT)

**Status: REAL DATA AUDITED; STRICT TWO-SOURCE HOLDOUT NOT YET ADMITTED.**
No official competition text, LoRA inference, training, or formal Kaggle submission was involved. All raw public corpus files were downloaded only into transient GitHub runners; only aggregate metrics were uploaded.

### Immutable public data evidence

- crtproduction/name-that-llm-open-pool, ntl_open_pool.parquet: 212,623 rows / 60 source identifiers / 18 labels; SHA256 1e4200aed80f52d655d4900a301cb427ca1341e06be31a8b7d5fde9ea3ef425f.
- Its sources.csv: 60 catalog rows; SHA256 0eb725083d61073afb459d4cd9190062388938c370b65c0d642d89550e9b7950.
- Frozen Stage-1 pool_v4.parquet: 75,783 rows; SHA256 37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec.
- **61,681 of the 212,623 new pool rows exactly match normalized Stage-1 text.** The remaining rows are not automatically independent; prompt and original-source overlap are separate questions.
- The source separately provides gemini37_unverified.parquet. Do not admit it as authentic 3.7-Flash data by filename.

Proof: [Kaggle inventory Action 42063920056](https://github.com/kolihhan/name-kaggle/actions/runs/42063920056), [source catalog 42064170317](https://github.com/kolihhan/name-kaggle/actions/runs/42064170317), [real pool overlap 42064722943](https://github.com/kolihhan/name-kaggle/actions/runs/42064722943).

### Zero-exact-text-overlap candidate sources

| Underlying source | Current pool rows | Stage-1 overlapping rows | Eligible multi-sibling families |
|---|---:|---:|---:|
| alessandrolair/ntl-kaggle-benchmarks-generated-passages | 3,701 | 0 | 3 |
| elliotthankinson/ntl-exact-gens-v3 | 13,359 | 0 | 3 (merged pool only) |
| aswathvanan/ntl-proxy-passages | 4,254 | 0 | 3 |
| trnhip/ntl-api-gen | 16,818 | 0 | 3 |
| kilupy/ntl-18-model-corpus | 3,359 | 0 | 3 |

**Not all pass provenance**. The 0-overlap measurement is lexical, not a guarantee of independent prompts or served model identity.

### Read the upstream generation protocols before deciding

[Kaggle source README audit 42065237791](https://github.com/kolihhan/name-kaggle/actions/runs/42065237791) and [follow-up 42065527178](https://github.com/kolihhan/name-kaggle/actions/runs/42065527178) verified:

- trnhip states returned served model IDs were not recorded (route=dify), and some prompts came from other public prompt banks. **Strict serving gate fails.**
- aswathvanan uses Kaggle Benchmarks proxy; lxd/seyit/arx prompt banks are reused. **Whole source not independent by prompt.**
- elliotthankinson claims pinned OpenRouter model IDs with fallbacks off and strict returned-ID matching, but API raw response logs were not independently audited. Original pool_v8j_add.parquet: 8,347 rows / 7 labels / SHA256 e8aeafcd75b9832c297324da360051a48fec3b58d1d8cfe251e03c2558f40185, zero normalized-text AND prompt_key overlap with Stage-1. **That original file alone covers only TWO multi-sibling families (GLM and GPT).** The merged pool's Claude coverage comes from other batches needing separate verification.
- alessandrolair generated new topic prompts using the Benchmarks proxy; original ntl_kbench_generated.parquet: 3,866 rows / 12 labels / SHA256 e98b1924269c30116f50623052f7fc4ea6148f7575d2fe483c9bd899b4ea2e21, zero Stage-1 exact text overlap. It contains proxy_model, prompt_group and run_date fields, but no independently retained backend served-model response.
- kilupy supplies original paired prompts for Claude / other-13 blocks, generated Nov 2; the blocks differ in prompts and per-row backend-served version is not established.
- zeminglii records model_seen and exact-provider route but explicitly reuses existing open-pool prompt banks, so an independently disjoint subset still needs proof.

[Original source raw parquet audit 42065925337](https://github.com/kolihhan/name-kaggle/actions/runs/42065925337) confirms the two original-source hashes, counts, and prompt-key overlap above. Only aggregate JSON was published.

### Decision boundary

The best provisional pair to inspect further is alessandrolair plus independently qualifying elliotthankinson sub-batches. **NO full strict holdout admission or PASS/FAIL claim yet.** Do not set model_serving_provenance_verified=true based solely on creator README assertions. Next: verify exact independent prompt coverage and true served-model traceability, then privately extract frozen Stage-1 A1 embeddings and run the already frozen readout test. GPU training and leaderboard submission remain unauthorized.


## 2026-11-06: Verified prompt leakage in a candidate holdout

GitHub Actions [42151858490](https://github.com/kolihhan/name-kaggle/actions/runs/42151858490) succeeded. It matched the 75,783-row frozen Stage-1 public pool (SHA256 37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec) to 212,623 new-pool rows using normalized answer text, then linked those matched answers to exact prompt_canon values. This is a **conservative detector**: 61,681 of 75,783 Stage-1 responses were mapped; 14,102 responses could not be mapped by this route.

Of 1,441 distinct prompts in zeminglii/ntl-own-exact-route-generations (2,162 total rows), **522 prompt texts** have a confirmed connection to a Stage-1 response. This affects **621 zeminglii rows**. Excluding those leaves **1,541 rows NOT PROVEN SPENT**, which must not be misrepresented as 1,541 PROVEN INDEPENDENT rows. Stage-1 pre-official adapter actually trained on only 30,133 of the 75,783 rows, so the 621 figure refers to overlap with the *broader Stage-1 public pool*, not a measured overlap with the exact adapter-training split. The stricter precise training-split overlap remains unverified.

The prompt catalog prompts.parquet contains 16,598 prompts, and all 1,441 distinct zeminglii prompts matched this catalog. The catalog contains no underlying original-source identifier.

Separate [Action 42151491857](https://github.com/kolihhan/name-kaggle/actions/runs/42151491857) confirmed zeminglii original passages.csv SHA256 4ff8c42f3c26b8ee0cfd1552253c3b7faa2ad4d111a8f8781a292e8d41e62afe: **2,162 rows all have nonempty model_seen containing the correct model label**. Specifically 319 Gemini 3.7 rows have returned model_seen=google/gemini-3.7-flash, OpenRouter route, Google provider; 416 Gemini 3.8 rows have model_seen=google/gemini-3.8-flash, Kaggle proxy route. This verifies recorded returned IDs, NOT the physical serving backend's identity.

Prior [Action 42151251888](https://github.com/kolihhan/name-kaggle/actions/runs/42151251888) found zero exact answer overlap between original alessandrolair and elliotthankinson. [Action 42151649815](https://github.com/kolihhan/name-kaggle/actions/runs/42151649815) also found zero exact normalized prompt overlap between alessandrolair and zeminglii, and zero direct text/ID overlap of zeminglii prompt values with Stage-1 opaque prompt IDs/keys. These **negative direct-ID tests do not supersede the positive Stage-1-to-prompt-canon reconstructed link above**.

**Decision: strict admission remains BLOCKED.** No frozen LoRA embedding export, readout scoring, GPU training, official Kaggle data, or Kaggle submission occurred. To unlock the frozen probe, account for the exact 30,133-row A1 training split and its 14,102 potentially unmapped broader-pool responses; verify source and prompt independence without treating ID formatting differences as proof. No evaluation criteria were changed.


### Exact frozen A1 training-split check (2026-11-06)

[GitHub Actions 42152215480](https://github.com/kolihhan/name-kaggle/actions/runs/42152215480) reconstructed the frozen Phase14 conservative subset and removed the original three Gate-A heldout sources/prompt keys. The resulting **30,133 A1 adaptation training rows** match the recorded Gate-A result exactly; a mismatch would have failed the job.

Joining adapter-training answers to the current public pool by normalized exact response text mapped **18,406** responses to a prompt_canon; **11,727** training responses remain unmapped. Among zeminglii's 2,162 candidate records, **125 rows are now provably tied to an actual A1 training prompt**. This is narrower than the earlier 621 rows linked to the *whole* 75,783-row source pool.

**Interpretation:** at least 125 zeminglii rows would violate a strict A1 prompt-disjoint holdout. Excluding only those 125 does NOT certify the remaining 2,037 rows because 11,727 actual A1 training responses lack a canonical-prompt mapping through this join. Nor can the entire 621-row broader-pool exclusion be described as adapter-training exposure.

**Next true gate:** resolve the 11,727 unmapped adapter-training responses to original prompt-bank lineage, or use a completely original, documented, source-disjoint and returned-model-verified external prompt corpus. Until then maintain NO ADMISSION, NO GPU INFERENCE, and NO scientific result. Do not relax the frozen contract.
