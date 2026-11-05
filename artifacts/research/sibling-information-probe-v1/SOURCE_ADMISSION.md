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
