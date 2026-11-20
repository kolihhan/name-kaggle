# NTL — exact original upstream source audit for frozen PC-CL 8,347 training rows

2026-11-20 (MYT). **Actual GitHub Actions SUCCESS**: [#44749886505](https://github.com/kolihhan/name-kaggle/actions/runs/44749886505). Script [ntl_pcc_lawful_source_manifest_20261120.py](../../../scripts/ntl_pcc_lawful_source_manifest_20261120.py). SHA-pinned 231,036-row Open Pool; same 8,347 train and 1,800 held-source validation as original PC-CL. CPU-only, raw text ephemeral, no inference, no training, no official submission.

| Public original dataset source | Frozen eligible train rows | Labels | Non-empty row-level served-model claims |
|---|---:|---:|---:|
| lxdlxdlxd/ntl-synth-generations-v1 | **4,270** | 13 | 4,270 |
| tadeuhayashida/ntl-claude-fable-5-generations | 1,501 | 1 | 1,501 |
| bingli283/ntl-gen-pilot-v1 | 1,007 | 7 | **0** |
| danielandresmiranda/ntl-openrouter-glm-kimi | 802 | 3 | **0** |
| keshavatearth/ntl-gpt-new-prompts-v1 | 490 | 5 | 490 |
| prvsiyan/prvsiyans-extended-name-that-llm-dataset | 266 | 1 | 266 |
| davidpiln/ntl-vertex-gemini-gen | 6 | 2 | 0 |
| elliotthankinson/ntl-exact-gens-v3 | 5 | 1 | 0 |

**Critical finding:** 4,270/8,347=51.16% of all rows come from ONE source, lxdlxdlxd. Two 1-class sources add 1,767/8,347=21.17%. Four sources totaling **1,820/8,347=21.8%** have **no non-empty row-level reported served-model identifiers** despite their source_protocol.csv having `served_model_check=yes`. This contradicts any blanket interpretation that all 8,347 are authenticated at example level; the field is a publisher statement even where present.

All eight source metadata records claim CC BY 4.0. That is a statement by the dataset uploader about licensing; it is **not independently authenticated original-model API output-use rights**. No source here is automatically cleared for new training by this audit. The underlying source URLs/terms must be checked, especially for Gemini, GLM and Kimi that were excluded by previous TERMS_AUDIT.

## Decision (scope-limited)

1. **Do not train all 8,347 as if equal-confidence 18-class truth.** The 160-step PC-CL data control is deeply source-concentrated and model-route authenticity incomplete. Neither data popularity nor `match_weight` resolves original-provider rights.
2. Before any strong QLoRA, make a per-source allowlist from original public producer documentation and applicable competition/provider rules; exact served route/version should be shown at row level when available. Preserve rejected/unknown labels as UNKNOWN, not clean negative supervision.
3. When authorized full-18 coverage remains poor, prefer a fully licensed *official TRAIN-only* prompt-disjoint training experiment with an honest source-heldout component analysis, or obtain separately permissioned external examples. Do **not** silently reduce from 18 classes to 11 and call the resulting accuracy a full-class win.
4. For future host-match sampling experiment, **freeze** a source and label balanced comparison and allow both sampling methods the same training budget. The prior 2,095-row top-weight subset already FAILED strict 18-class sample sufficiency; use that finding rather than resweeping quantiles.
5. Competition best stays **0.702** (#58514818). No model BA improvement or Kaggle submission is implied here.

**Next decision:** Source-level authorization/identity proof for the 4,270-row large source, then choose full-exposure CE from an actually admissible pool. This is a blocker inventory with exact upstream names and counts, not another performance result.