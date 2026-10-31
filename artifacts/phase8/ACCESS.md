# Phase 8 — External-source access status

Date: 2026-10-31

## A. Maniceet OpenRouter + OpenAI Generations

Source: `maniceet/name-that-llm-openrouter-openai-generations`

Public Kaggle metadata currently reports 7,753 passages, 10 competition labels, row-level provenance, and CC BY 4.0. The source is attractive because it appears to add five labels that Phase 7's existing external source does not cover: `gemini-3.7-flash`, `glm-5.3`, `glm-5.3-flash`, `kimi-k3`, and `qwen3.8-max`.

### Fetch attempt

GitHub Actions run: `41197148926` (`Phase 8 fetch Maniceet raw dataset`).

Observed access behavior:

- anonymous Kaggle dataset download endpoint: HTTP **403 Forbidden**;
- repository `KAGGLE_API_TOKEN`: **present** in the Actions environment (masked by GitHub);
- authenticated Kaggle CLI request: HTTP **403 Forbidden**, `You don't have access to this resource`.

Therefore the source is **ACCESS-BLOCKED**, not audit-failed. No raw rows were obtained, no raw identity/provenance audit was possible, and no training is authorized from this source.

Do not reinterpret the public data card as raw-row verification.

## B. Next candidate: Oleg Zholobov Response Corpus

Candidate slug: `olegzholobov/name-that-llm-response-corpus`.

Current public indexing describes it as an 18-model-wide response corpus with shared prompts and external-response provenance. Because this coverage shape directly addresses Phase 7's class asymmetry, test raw access before spending more work on Maniceet.

The same rule applies: public metadata is reconnaissance only. Training is forbidden until raw files are fetched and audited.
