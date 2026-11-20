# NTL — ORIGINAL generator + README audit: 4,270-row main source

## Scope / verified execution

2026-11-20. Real CPU GitHub Actions: [source code + provider routing #44756877019](https://github.com/kolihhan/name-kaggle/actions/runs/44756877019), [README host-domain evidence #44757193056](https://github.com/kolihhan/name-kaggle/actions/runs/44757193056). Earlier metadata structure [#44756551514](https://github.com/kolihhan/name-kaggle/actions/runs/44756551514). All SUCCESS. Inspected original public producer dataset **lxdlxdlxd/ntl-synth-generations-v1**. Downloaded **README.md and generator code only**, never generated passages, labels/prompt content, official test, weights or secrets. No GPU, inference, training, or Kaggle submission.

Original public dataset metadata: title *NTL synthetic generations v1 (13 labels)*, subtitle says 4.8k passages from 13 labels via official APIs; `dataset-metadata.json` nests metadata in `info` (initial v1 parsing missed this and must not be misreported as missing original documentation). Published dataset license metadata **CC BY 4.0**.

Original public generation code: `code/code/providers.py` (244 lines, sha256 `84ecb754c038d2cc9747336137adfdb9173e28e288a6a1f5a4fa185aaf2e6d5e`) defines `vertex_gemini`, `vertex_claude`, `openrouter`, `zhipu`, `_openai_compat`, `complete`; also `generate.py` (104 lines). This is **plausible producer routing code**, not authenticated execution transcripts. README says upstream `generations.parquet` includes `provider`, `requested_model`, `returned_model`, `settings`, prompt identifiers; publisher reports returned id always matched requested. **This is stronger original-source lineage evidence than aggregator `served_model_check=yes`, but remains unverified publisher claim.**

## Domain-gap evidence from original README (do not ignore)

Producer reports its own comparison against official competition TRAIN with a competition-trained classifier. Treat its self-ID metrics as producer observations, not our reproduced independent generalization:

- **Claude default:** punctuation style was relatively close to official train and self-identification near official OOF recall.
- **GLM5.3 default and Qwen3.8-max -0902:** punctuation reasonably close, self-identification near or somewhat below recall.
- **Gemini:** official competition Gemini passages had approximately **0–1% em dash**, original synthetic Gemini **3–40%**, depending on thinking settings, plus more hyphenation/curly quotes. Self-identification worse than on official train.
- **Kimi-K3:** thinking-enabled em dash **93%** vs official **52%**, with additional curly-quote mismatch. Thinking-disabled outputs better match several features, still less semicolon use.
- **Within-family incomplete coverage:** producer explicitly warns that adding data for some members of a family without siblings can pull predictions away from related classes.

These are **publisher-reported style ratios**, not our own recounted raw text or proof that such features determine hidden test success. Do not blindly train Gemini/Kimi synthetic passages merely because their label and API returned-model fields exist.

## Licenses and admission

Original README licenses its prompts and generated texts CC BY 4.0, but explicitly cautions that **individual model-provider output terms apply in addition**. Original provider permissions for competition fine-tuning/republication have NOT been independently confirmed. External 231k pool downstream CC claims cannot override them. Prior Phase15 excludes Gemini/GLM/Kimi external for provider terms reasons; this audit does not authorize reversing that exclusion.

## Decision / next experiment

1. **DISCARD** the proposed unfiltered 8,347-row full-data CE or blind match_weight finetune as a defensible score-facing experiment; 4,270/8,347 main-source share alone would magnify its synthetic style artifacts. Keep original official best **Public 0.702**.
2. **First** verify original provider output-use rules and which API settings/returned model IDs were recorded for each intended admissible family. If still uncertain, do not train that source. Separately the official Kaggle TRAIN 2,699 rows is a possible authorized group-disjoint baseline without external provider-output concerns.
3. **If** source allowed, freeze a label/source-controlled, model-family-aware CE-only comparison, explicitly **downweight/quarantine mismatched Gemini/Kimi settings** without tuning to Public; include control with producer metadata filtering. Honest source-disjoint validation, per-class recall, rescue/regression, 9h inference feasibility required before one scored submission.
4. Published source provenance is a useful lead, but no new true 0.702 strong-stack paired logits and no independent new hidden-label panel exist; a local external validation lift will not be called a Public improvement.

Authorship and permission evidence remains separate from classification quality. This result is a more precise DATA-GENERATION bottleneck diagnosis, not a new model score.