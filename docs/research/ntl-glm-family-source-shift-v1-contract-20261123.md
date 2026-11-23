# GLM Family-Recognition and Matched Source-Shift Audit — Frozen 2026-11-23

## Scope and rationale
This is a read-only, zero-fit, no-GPU source/protocol diagnostic. No Kaggle submission, source weight tuning, threshold optimization, or 0.715 champion code use. We never use external generated passages to fit any model.

Already-authenticated prior evidence: the historical 150-prompt official-train GLM-5.3 versus Flash native likelihood achieved 89.33% binary BA, but a distinct original-generator 64-pair external cohort (128 responses) showed **0.683 champion 88/128 exact-correct → native likelihood gate 77/128**, with **6 rescues and 17 regressions**. See GitHub Actions #43612072194 and the 2026-11-15 GLM64 decision report. Among 128 true GLM examples, 95 had been predicted as GLM-family and 88 of these 95 were ALREADY correct; only 7 eligible errors were repairable. 33 true GLM instances were not predicted GLM-family at all. This is an immediate **family-recall bottleneck**, not evidence to repeat the failed routing/threshold test. This historic 0.683 family gate must not be equated with the now-0.715 champion.

## Immutable input
- Official TRAIN SHA256 40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707; only glm-5.3 and glm-5.3-flash labels, exactly one per prompt among 150 prompts (300 responses).
- Public original generator CSV from lxdlxdlxd/ntl-synth-generations-v1 SHA256 8339081afee1d4aed0b806a686d91978f0d7815bc67fb8f620860477e0895675; 4758 original rows, select EXACTLY the same 64 matched pair prompts (128 responses) with fixed existing selector kaggle/ntl_glm_crosssource_private.py::select_pairs. Unique pid per prompt, same user message/target_words/setting for both generator labels, returned model matches label. No other cohort or tuning after seeing results.
- Only summary aggregates printed; never print/write passages, user prompts, text hashes, selected IDs, token IDs, trained heads, per-row predictions or private Kaggle assets to GitHub. External dataset downloaded only ephemerally, verified by SHA, then deleted.

## Fixed measures (all are response-level, unsupervised)
Ten features in order: (1) English word count via letters with internal apostrophes; (2) average word length; (3) word count / max(1, count sentence-ending ASCII periods/exclamations/questions); (4) ASCII punctuation ,.;:!? per 100 words; (5) U+2014 em dash per 100 words; (6) newline per 100 words; (7) Markdown markers: counts of **, # and backtick per 100 words; (8) ASCII digit characters per 100 words; (9) case-insensitive whole-word I/we/me/us/my/our per 100 words; (10) exact fixed function-word inventory from scripts.run_style_baseline.FUNCTION_WORDS per 100 words.
Also aggregate, WITHOUT exposing literal potentially prompt-containing setting fields, the external source cardinalities for target_words and generator setting (hashed categorical token counts only).

Two predeclared outcomes:
A. Source shift external minus official for each model and each feature: sample mean difference, domain/model median and cluster-bootstrap 95% CI by paired prompt group (1,000 replicates, seed 20261123; resample 150 official and 64 external prompt pairs independently).
B. Prompt-matched version fingerprint: within each domain for each prompt, feature(full) minus feature(flash), then external mean paired contrast minus official mean paired contrast; independent paired-cluster bootstrap 95% CI and descriptive sign flip detection. Fixed all 10 outcomes; don't selectively highlight favorable features and don't treat 10 intervals as multiple-testing-adjusted.

Additionally publish fixed prior 64-pair gate arithmetic: eligible GLM 95/128, correct within eligible 88/95, missed family 33/128, eligible repair ceiling 7/128, 6 rescues / 17 regressions / net -11. Only aggregate count is available: do NOT invent paired-cluster confidence intervals for historical predictions.

## Falsifiability and limits
A detectable distribution shift supports concern about host/protocol mismatch but DOES NOT prove it caused routing errors because row-aligned likelihood/predictions are unavailable and served model version identity/pretraining independence are unresolved. If absolute/pairwise style features are stable, do not claim no source shift at semantic/token level. Paired version contrasts adjust prompt content WITHIN source but not source/host differences BETWEEN sources. All official data and this 64-pair cohort are spent for model selection.
STOP after one audited CPU run. Do not train a family classifier, change decision threshold, re-submit or fit on external rights-uncertain outputs. Our 0.715 Public Champion remains unchanged.
