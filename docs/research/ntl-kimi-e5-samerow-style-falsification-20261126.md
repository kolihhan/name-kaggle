# NTL Kimi/GPT frozen E5 — real same-row complementarity versus style controls
**Date:** 2026-11-26 MYT. **No official Kaggle submission, GPU, model fine-tune, or new 18-way Public score.** Authenticated incumbent remains **0.718** (official #58755304). Confidence Fusion stays retired.

## Real execution and immutable protocol
- Frozen source experiment: [successful GitHub Action #45744290403](https://github.com/kolihhan/name-kaggle/actions/runs/45744290403).
- Exact same-cohort lightweight style control: [successful Action #45744821355](https://github.com/kolihhan/name-kaggle/actions/runs/45744821355).
- **Crucial same-row E5 vs style rescues/regressions audit**: [successful Action #45744952794](https://github.com/kolihhan/name-kaggle/actions/runs/45744952794).
- Independent 335M frozen encoder: `intfloat/e5-large-v2`, pinned revision `d94960066016d2064f2a3a2dbc65107a2b48230d`. No encoder gradient updates or weight fitting. The binary readout **does use supervised class centroids fitted from 70 official train prompt pairs**, so “no training whatsoever” would be inaccurate.
- Samples: 70 official train prompt pairs for class means, 40 **disjoint official** prompt pairs to check, 80 external `prvsiyan` `split in {new,new2}` matched prompts and first-party `Moonshot AI` Kimi K3 vs `OpenAI (first-party API)` GPT Luna. Exactly 380 paragraphs encoded for the frozen E5 test; source and target text not republished. External matches have zero exactly normalized text overlap with the selected official train and validation responses.
- This diagnostic deliberately measures **2-class Kimi K3 vs GPT Luna**, not competition full 18-class balanced accuracy, not output similarity to the existing 0.718 final stack. External provenance is publisher-recorded, not independently signed by provider.

## Frozen E5 original result
| Metric | Result |
|---|---:|
| Official 40-prompt held-out 2-class BA | 0.6500 |
| External 80-prompt held-out 2-class BA | **0.7000** |
| External Kimi recall | 0.7375 |
| External GPT Luna recall | 0.6625 |
| Paired Kimi-vs-GPT ranking | **0.8625** |
| 95% prompt-bootstrap external BA interval | [0.64375, 0.75625] |
| Frozen single-feature word count baseline BA | 0.5000 |
| Frozen shuffled train label control BA mean | 0.5038 |
| Predeclared randomized control one-sided tail | 0.00498 |

The source-specific 2-class evidence establishes a transferable frozen-vector linear readout, **not incremental utility over the incumbent**.

## More important negative control: same frozen 80 prompt IDs and 160 responses
Style-only controls use the exact 70 official prompt pair centroids, not fitted in response to the E5 result. The ten features are em-dash density, curly apostrophes, curly quotations, semicolons, commas, colons, digits, long-word fraction, mean whitespace-token length, and word count.

| Control | External binary BA | External pair ranking | E5-only correct | Style-only correct | E5-minus-style BA 95% paired prompt-bootstrap |
|---|---:|---:|---:|---:|---|
| Em dash alone | **0.70000** | 0.6500 | 30 | 30 | [−0.09375, +0.08750] |
| All 10 style properties | **0.61875** | **0.8750** | **38** | **25** | [+0.00625, +0.15625] |
| Word count alone | 0.50000 | 0.4750 | 53 | 21 | [+0.11875, +0.28141] |

**Paired confusion against em dash, N=160:**
- Both correct 82; E5 only correct **30**; em dash only correct **30**; both wrong 18.
- On each class independently: E5-only correct **14 Kimi + 16 GPT**; em-dash-only correct **14 Kimi + 16 GPT**. Thus **no net improvement in individual binary recall or overall BA** against this one-feature control. Source-specific complementarity of errors exists but no validated selection gate.
- E5's ranking improves over em dash by +0.2125, prompt-bootstrap CI **[+0.1000,+0.3250]**, yet that paired rank is not available for isolated hidden-test responses where the rival generator's answer is unknown.

**Paired confusion against ten style features, N=160:**
- Both correct 74; E5-only correct **38**; ten-style-only correct **25**; both wrong **23**.
- **Kimi:** E5 rescues only 1 previously wrong style label and regresses 20 previously correct style labels. **GPT Luna:** E5 rescues 37 and regresses only 5. This reflects style-only overprediction of Kimi, not universal E5 superiority.
- E5's BA is +0.08125 higher on these 160 source-specific responses (paired prompt-bootstrap interval [+0.00625,+0.15625]), but **pair ranking is slightly worse** than 10-style (0.8625 vs 0.8750; interval for ranking difference includes 0).

## Hard-stratum check
Exact external matched prompts stratified **without tuning**:
| Style condition | Paired prompts | E5 2-way BA | Em-dash BA | E5 paired ranking | Em-dash paired ranking |
|---|---:|---:|---:|---:|---:|
| Neither response uses em dash | 11 | 0.5455 | 0.5000 | 0.5455 | 0.0000 (tie counted as not rank-correct) |
| Both responses use em dash | 29 | 0.7414 | 0.6034 | 0.8621 | 0.6552 |
| Only one response uses em dash | 40 | 0.7125 | 0.8250 | 0.9500 | 0.8250 |

The “no em dash” stratum is too small to establish deep non-typographic discrimination. Both-use style stratum supports some nontrivial E5 representation information, but lexical/protocol confounding remains. Pair rankings use strict `score(Kimi) > score(GPT)`: ties count as incorrect for that metric.

## Scientific conclusion and stop condition
1. **Do not promote a model.** Strong frozen E5 is better balanced than 10-style on this source, but **no better than a single em-dash feature in binary BA** and 30 gains exactly offset 30 losses on the same 160 rows.
2. The actual 0.718 full-stack incumbent already integrates substantial style and pretrained experts, including Open Pool. We have **no authenticated same-row 18-way incumbent predictions** on eligible disjoint external test rows; cannot demonstrate positive rescues minus regressions in the 18-class objective. Public BA +0.043 for top-20 cannot be inferred.
3. Historical exact-source overlap audit [#45743815844](https://github.com/kolihhan/name-kaggle/actions/runs/45743815844) found 173/239 Gemini3.7 responses in *current* Open Pool, and zero exact Kimi/GPT duplicates, but pretrained checkpoint training histories remain unknown; current Open Pool might differ from the November23 0.718 scored snapshot.
4. **Stop further isolated encoder or style rule sweeps** against this now-spent 80-pair panel. Before any score-facing hypothesis, obtain an *independent* allowed 18-class same-row incumbent prediction/logit panel, or a method that changes core scoring evidence with a matched end-to-end comparison. Do not launch a 9-hour GPU on unsupported fusion.
5. No Kaggle competition submission, no Kaggle notebook push, and no GPU were made in these experiments. The only paid computation was normal GitHub Actions CPU.

## Files
- `scripts/ntl_kimi_e5_frozen_transfer_20261126.py`
- `scripts/ntl_kimi_e5_style_falsifier_20261126.py`
- `scripts/ntl_e5_style_same_row_error_audit_20261126.py`
- Workflow: `.github/workflows/ntl-20261126-e5-style-samerow-rescues-cpu.yml`
