# NTL PRO — actual 18-class outside-source test: 0.4120; hard domain shift and prediction concentration

**Date 2026-11-16 MYT.** This is a completed real inference test, not a simulated score.

## Real verified evaluation

- Frozen own Kaggle-scored `#58375806`, Public **0.691**, unchanged Lightning V2 weights, ensemble, GSQ and 37 model-stage modules; one genuine private Kaggle T4 run `kolihhan/ntl-oleg18-frozen0691-probe-nov16`, now **COMPLETE**.
- [Read-only authenticated exact terminal state + truth match #43804320931](https://github.com/kolihhan/name-kaggle/actions/runs/43804320931): **89/216 correct = 0.41203704** 18-class BA. 12 per class, 216 distinct byte-exact prompt groups. Real final prediction file verified to contain precisely the anonymous 216 input IDs. Original official train/test not used for this pilot, no extra fitted model/official submission.
- [Source-identity, word-count, error distribution CPU audit #43804722129](https://github.com/kolihhan/name-kaggle/actions/runs/43804722129) **SUCCESS**, no new GPU. Stratified 4,000-replicate bootstrap sampling interval **[0.3704, 0.4537]**, quantifies sampling noise only, not upstream prefit contamination or label accuracy.

## The key confound: all 216 passed strong evidence, but all were about sensitive topics

- **216/216** from domain prefix `sensitive_topics/`, across a range of sensitive-topic subdomains. All had exact user prompts, strongest provenance category `request_and_returned_model`, `possible_truncation=False`, and **`output_terms_review=True`**.
- Therefore this is a ***narrow, challenging topic-distribution pressure test***, not a representative cross-domain external dataset. `output_terms_review=True` means source-output rights review is flagged, not necessarily that a label is incorrect.
- Original 0.691 Kaggle Public and 0.412 outside-source benchmark are **different distributions and not a directly matched estimand**. This gap neither proves general-purpose BA fell by 0.279 nor by itself identifies overfitting.

## What we actually learned

| Measured subgroup | Correct / total | Accuracy |
|---|---:|---:|
| All | **89/216** | **41.20%** |
| 40–80 words | 14/59 | 23.73% |
| 81–140 words | 9/21 | 42.86% |
| 141–250 words | 66/136 | 48.53% |
| Anthropic family (5 true labels) | 40/60 | 66.67% |
| OpenAI family (5 true labels) | 33/60 | 55.00% |
| Google Gemini family (4 true labels) | 11/48 | 22.92% |
| GLM family (2 true labels) | 3/24 | 12.50% |
| Kimi-K3 (1 true label) | 0/12 | 0% |
| Qwen3.8 Max (1 true label) | 2/12 | 16.67% |

- Of **127** errors, **63** cross-family and **64** same-family sibling; both defects matter in this *particular external domain*, unlike the spent official-train difficulty subset which had predominantly cross-family selected errors.
- Major label-collapse symptom: predicted `claude-opus-4-6` **47 times**, `gpt-5.6-luna` **36**, `claude-sonnet-5` **30**, and `kimi-k3` **0**. These 3 labels alone account for **113/216** predictions. Note all true classes have exactly 12 instances; these frequencies are *not evidence of correct priors*.
- Largest single confusion `gpt-5.6-sol` → `gpt-5.6-luna`, 8 cases; several distinct families were incorrectly assigned to Claude Opus 4.6.
- Short length vs accuracy is **observational and potentially confounded** with true class, subdomain, source and answer verbosity. Do not add length-specific correction rules from 216 already consumed examples.

## Source lineage limitations

- The winner has 23 Kaggle data mounts and the external corpus catalog 31 upstream slugs with **0 exact name intersections** ([source identity audit #43786295997](https://github.com/kolihhan/name-kaggle/actions/runs/43786295997)). **This does not rule out duplicated response content, mirrored sources, or prior fitted checkpoint exposure**.
- No causal pre/post-GSQ ablation was performed. The old pilot was train-lineage-spent and the new one is a focused domain stress test. Neither can certify general unseen-prompt 18-class performance.
- Test is now **spent** for model selection; do not optimize or submit on these 216 outcomes. Never use model-family/label-specific hardcoded fixes learned from the error list.

## Prospective decision

**Retain official 0.691**, **do not submit** a hand-adjusted candidate based on this one niche corpus. Split competing hypotheses before more GPU:

1. **Source/domain-shift hypothesis:** conditional generation channel, safety/postprocessing and topic have greater bearing than model-identity features. Test on separate provenance-strong, diverse domains with exact 18-class support; study source-specific quality and confidence *without* fitting on this dataset.
2. **Model-identity representation hypothesis:** learned ensemble over-selects just a few classes across domains, possibly from calibration/weight representation; seek **another independent full-18 source** and compare the **unchanged** predictions and source-level confusion. The present 216 are not reusable for tuning.
3. **Label/source-provenance hypothesis:** rows with request-vs-served model identity mismatches or output wrappers may distort truth. Use author response-level provenance and alternative channels to audit, not new supervised fitting.

First next step (cheap, decisive): source/domain eligibility census on the remaining public Oleg corpus, **excluding sensitive_topics**, and report which labels retain honest served-ID provenance and known licensing. If all 18 cannot be certified, stop the false external-Benchmark project rather than broadening to unverified labels and claiming a comparable 18-class BA.

No training, new GPU, or official Kaggle submission.
