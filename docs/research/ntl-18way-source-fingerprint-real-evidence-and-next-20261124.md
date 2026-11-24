# Name That LLM: actual source-stable fingerprints versus sibling discrimination (2026-11-24)

## Bottom line
**REAL positive but non-deployable research signal.** At the prompt/model CENTROID level, source-relative lexical and typographic fingerprints correlate across multiple generator publishers. That correlation is strongest for curly quotes, robust across all audited datasets, but **within-family exact-version confusion remains substantial**. These analyses do not estimate hidden test BA or produce an original competition winner. We did not train any model, use GPU, alter the official scorer, or make any competition submission.

## Frozen actual studies
- [All18 official train vs Nischay 14 complete prompts Action #45423035348](https://github.com/kolihhan/name-kaggle/actions/runs/45423035348), SUCCESS. 149 official complete all18 groups (2,682 rows) and 14 external complete all18 groups (252 rows, 7 short / 4 median / 3 long). Original official train SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`, Nischay dataset `generations.csv` SHA256 `bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c`. Exact 14 frozen features, 2,000 prompt-wise label permutations (maxT family-wise error correction) + 1,000 prompt bootstrap.
- [Different publisher C1/C2 matched 12 labels Action #45424044547](https://github.com/kolihhan/name-kaggle/actions/runs/45424044547), SUCCESS. Exact same 14 style features; first restrict original official prompts to same 12 labels, then normalize each source by matched prompt ranks. C1 `beroli/ntl-benchmark-c1-generated-panels` SHA `8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177`, **30 fully matched 12-way groups / 360 rows**. C2 `beroli/ntl-benchmark-c2-generated-panels` SHA `bf25ef1e953d5c37b177f27b841a04098ae9b35f45fc43fa11036d91e8831e02`, **9 fully matched 12-way groups / 108 rows**, with 362 other rows unsuitable for matched-prompt test; NOT '50 all12 groups'. C1 and C2 are from **same** external publisher and not two independent provider replications.
- All source labels and provider versions are **publisher-reported**; no first-party serving log authentication. Official and external corpora have previously been used in other tests; **not a virgin independent hidden benchmark**. Following study was registered after observing the Nischay study; not a pristine pre-registration for all feature choices.

## Real model-centroid outcomes (privileged matched prompts; NOT per-passage prediction BA)

| Observable | Official vs Nischay (18 labels) | Official vs Beroli C1 (12 labels) | Official vs Beroli C2 complete groups (12 labels) |
|---|---:|---:|---:|
| Complete external prompt blocks | 14 | 30 | 9 |
| Model-centroid correct best match | **13/18** | **9/12** | **6/12** |
| Model-centroid true label among top3 | **15/18** | **12/12** | **12/12** |
| Prompt-wise label permutation p for best-match count | **0.0005** | **0.0005** | **0.0005** |
| Curly quote source-rank Spearman | **+0.9409** | **+0.9058** | **+0.8841** |
| Curly quote corrected maxT permutation p | **0.0005** | **0.0030** | **0.0045** |
| Em dash source-rank Spearman | +0.8836 | +0.8803 | +0.6032 |
| Em dash maxT adjusted p | 0.0005 | 0.0085 | 0.4168 |
| Mean word length rank Spearman | +0.8890 | +0.3077 | +0.6923 |
| Log word-count rank Spearman | **−0.2219** | **−0.0559** | **−0.3636** |

C1's only 3 model-template errors were **all within the same provider/family**: Claude Opus 4.8 → Opus 5; GPT 5.6 Sol → Luna; GPT 5.6 Terra → Sol. Thus 12/12 C1 centroid identities were assigned their correct broad provider (but only 9/12 exact models).

C2's six model-template mismatches include five same-provider errors and one across-provider (Gemini 3.6 Flash → Claude Opus 4.6). C2 provider-level correct 11/12, exact 6/12. Its **9 complete prompt banks** make estimates weak. C2 not promoted as equal support to C1.

## Distinguish source-invariant *relative ranks* from drift in raw rates
From Nischay publisher's entire synthetic `generations.csv` versus official train:
- Gemini 3.8 Flash: percent with em dash **0% official vs 18.69% synthetic**.
- Gemini 3.6 Flash: **0% vs 15.81%**.
- Gemini 3.7 Flash: **0.67% vs 15.84%**.
- Kimi K3 **52.00% vs 52.31%**.
- Claude Fable 5 **83.22% vs 81.13%**.
Both propositions can be true: source-level absolute rates drift, while the models' relative feature positions across all families remain similar. This is exactly why naive fixed punctuation vetoes have repeatedly failed.

## Mechanism interpretation
**Verified:** some stylometric identity-associated features (especially curly quote usage) have positive reproducible cross-source model-rank association, beyond 2,000 per-prompt label-shuffle nulls. In C1, model CENTROIDS from another publisher mostly retain **provider family** even though sibling versions swap.

**Not yet proven:** attribution for individual unknown passages; independence from publisher-specific prompt selection, service route/version and annotation; improvement vs **account best 0.718 Public BA**, which is a *reproduction of public code*. Matching identity centroids uses true class and complete known matched 12/18-response prompt sets. Hidden Kaggle test has **no prompt_id**, and its class proportions differ: we cannot compute these model-centroid rankings at inference time. It is not honest to call 13/18 a new BA or to claim it adds 0.05 points.

**Rival causal interpretations:** (1) valid model family fingerprints exist but siblings need more informative evidence; (2) prompt-format and publisher route produce stable broad-family shortcuts that fail per-passage. Separate genuine provider, prompt and sampling variables needed to tell them apart.

## Decision and next experiment boundary
`RESEARCH POSITIVE`, **not `PROMOTE`** and not `STOP because low`. This confirms a usable *research hypothesis* about preserving source-invariant family signal; it also isolates the **exact-version sibling discrimination** obstacle. Prior outdated public 0.721 shared ensemble and Fable guard are unlikely to close a >5pp top10 gap alone.

Next original architecture candidate must solve **prompt-FREE single-passage sibling identity**, with strong frozen high-capacity model evidence across **GPT 5.6 Luna/Sol/Terra**, **Gemini Flash 3.6/3.7/3.8**, **Claude Opus versions**, while preserving already good inter-family evidence. Pure punctuation/count/length rules are **insufficient** and should not be submitted.

Before any expensive or official scored run:
1. Find a genuine 18-way *same-row* inference outcome for the exact 0.718 champion or get one no-training pinned private cohort, including per-model actual error family split, and whether existing frozen strong experts can rescue its missed sibling rows. Pending 0.718 Fable 252 GPU cohort is separate.
2. Design **two frozen high-capacity inference-only evidence heads** on that same cohort: unchanged original 0.718 vs new *actual additional model evidence* per sibling (not mere reranking thresholds); count paired rescues and regressions with exact row IDs; controls must preserve all 18 predictions and runtime feasibility under 9h hidden cap.
3. Do not search new blend weights/thresholds on previously-used 2,699 training rows, C1/C2 or public leaderboard. No small-model training (owner rule). If no legal first-party high-capacity model or source-diverse qualified evaluation is available, state feasibility/cost blocker instead of inventing score lift.

All code, results and counted metrics are public repo **research evidence**, not a claim of winning solution.
