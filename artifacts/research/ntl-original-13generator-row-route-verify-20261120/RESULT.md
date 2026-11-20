# Original upstream 13-generator row-by-row requested/returned model check (2026-11-20)

**REAL completed CPU [Action #44765273054](https://github.com/kolihhan/name-kaggle/actions/runs/44765273054).** Code: [metadata-only checker](../../../scripts/ntl_original_generation_route_verify_cpu_20261120.py). Original publisher dataset: `lxdlxdlxd/ntl-synth-generations-v1`, upstream `generations.parquet` rather than 231,036-row aggregated Open Pool.

## Results

- Original upstream: **4,758 published generation records, 13 model labels**; publisher-requested and publisher-returned exact model identifier strings **4,758/4,758 match (0 blank/mismatch)**. This verifies the internal integrity of the public original record fields and reproduces the README's claim. It is **NOT independently authenticated server billing/API call logs**. All rows came from a source whose publisher claims first-party API generation; no model inference was made in this audit.
- Five Claude labels: 330 rows each, named `vertex_claude`, 330 distinct prompt IDs per label, producer `setting` has one distinct value per label. This may inform a *future* Claude-specific classifier if data rights, permission, and domain match are independently validated.
- Four Gemini labels: 390 rows each using `vertex_gemini`, 330 distinct prompt IDs per label and 3 settings per label; not a single known uniform generator protocol. Producer README independently reported strong Gemini style mismatch to competition official TRAIN (em dash 3–40% versus 0–1%).
- GLM-5.3: 406 rows, GLM-5.3 Flash: 413, both `zhipu` via publisher routing, 4 settings each. Z.ai output use policy remains a training gate, not waived by request/return identity matching.
- Kimi-K3: 369 rows, `dashscope`, 330 distinct prompts, 3 settings. **320/369** explicitly have `enable_thinking=false` in published settings; 49 do not state a boolean in parsed metadata. Original producer README described Kimi style mismatch under thinking-on and partial alignment in thinking-off.
- Qwen3.8-max: 360 rows, `dashscope`, 330 distinct prompts, 2 settings.

| Original upstream group | Row totals | Distinct label count | Reported generation route |
|---|---:|---:|---|
| Claude | 1,650 | 5 | vertex_claude |
| Gemini | 1,560 | 4 | vertex_gemini |
| GLM | 819 | 2 | zhipu |
| Kimi | 369 | 1 | dashscope |
| Qwen | 360 | 1 | dashscope |
| **Total** | **4,758** | **13** | Publisher-supplied route metadata |

**No raw generated text or original prompts were loaded into the Python analysis or exported**, although the public source parquet was downloaded ephemerally by the Action and then deleted. Only safe aggregated class and settings metadata was logged.

## Decision

**PASS: within-publisher request/return consistency of original 13-label source. FAIL: authenticated provider provenance/18-label coverage/host parity are still not established.** The previous 8,347-row train subset contained 4,270 rows from this producer because filters excluded other published examples; this report reviews the full original 4,758 and must not be conflated with the filtered training set.

Do not train all 8,347 external rows or submit a new 0.702 challenger solely from metadata consistency. Current evidence supports narrower **per-model-setting rights/host-style filtering**; legal original source-output rights, particularly GLM and routed Gemini/Kimi, remain independent blockers. Real original source [generator and README audit](../ntl-4270-upstream-generator-domain-shift-20261120/RESULT.md) and [provider terms review](../ntl-provider-output-use-terms-20261120/RESULT.md).

Official NTL best remains **Public BA 0.702 (#58514818)**. No model training, GPU or competition submission.