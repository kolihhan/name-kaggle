# NTL external source admission experiment, 2026-11-17

Inspected released model answers at [GlobeBench commit 5f29c08](https://github.com/patronus-ai/globebench/tree/5f29c086d4604fb63789918175076201b13a083b/runs). An open answer table for multiple models could close some of the exact model-ID gap. Competing explanation: apparent coverage comes from requested model labels, long machine-readable answers, or ranks without usable text.

Read each relevant subject shard and count full, nonempty `prediction` strings, excluding flagged error, refusal, and incomplete finish states. Use full responses of 40–250 whitespace-separated words. Require the separate returned provider model ID to equal the declared exact route. For matching, use item ID, renderer and sample index, then compare recorded request settings.

| Exact route | Clean 40–250-word answers | Separate exact returned ID |
|---|---:|---:|
| Claude Fable 5 | 25 | 25 |
| Claude Opus 5 | 62 | 62 |
| Gemini 3.7 Flash | 135 | 0 |
| GLM 5.3 | 172 | 0 |
| Kimi K3 | 137 | 0 |
| Qwen 3.8 Max | 278 | 0 |

Fable has 323 nonempty complete replies out of 1,632 total rows, of which 25 meet the length check. Fable and Opus have 23 identical item/render/sample/settings pairs in four domains. Seventeen pairs start with JSON-like or fenced output in both arms. Ten same-item groups cover the ten competing labels present if returned-ID checks are ignored. There are zero qualifying 18-model groups. The authors release outputs under Apache-2.0 within their rights and retain provider terms. No raw output was republished. The Hugging Face mirror exposes a normalized model key and served state, without a separate returned provider ID.

Second Fable source, [PawanKrd revision 4bf63f6](https://huggingface.co/datasets/PawanKrd/claude-fable-5-code/tree/4bf63f6009a984b50f5a7e07368e3fe24fa849aa), has 603 rows, 407 with an exact claimed model and a single assistant message, and 114 of those meet the length screen. It does not provide a separate provider-returned ID, other matched models, or a declared license. No holdout admission.

A fresh read of the [Vercel Gateway public model listing](https://ai-gateway.vercel.sh/v1/models) found an exact-name route for all 18 targets, including all six missing from the earlier Kaggle catalog. Listing canonical slugs does not verify actual served revisions. The queried gateway routes have metered unit prices. Vercel documents a free-credit tier, but the authenticated connector returned no accessible team. No usable zero-cost account/key or model generations were established. No fitting, inference, validation balanced accuracy or official submission.

Next obtain a permitted zero-cost exact-version source for all 18 models on the same original requests. Verify the provider-returned IDs before any full-panel generation or candidate evaluation. Official best `0.691` [#58375806](https://github.com/kolihhan/name-kaggle/actions/runs/43743764085).
