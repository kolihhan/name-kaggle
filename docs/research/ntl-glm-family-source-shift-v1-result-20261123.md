# GLM Family Recall + Matched Source Shift — Real Read-only Result

Date 2026-11-23 MYT. Authenticated GitHub Actions [#45127329258](https://github.com/kolihhan/name-kaggle/actions/runs/45127329258): terminal **SUCCESS**. Frozen preregistered [contract](ntl-glm-family-source-shift-v1-contract-20261123.md), scientific [source](../../scripts/ntl_glm_matched_source_shift_audit_20261123.py), original GLM source data and official TRAIN SHA checks, all self-tests PASS. **No training, no pretrained model inference, no GPU, no competition submission, no 0.715 model modification**. Runtime read/analysis 0.172s (dependency and public-source download excluded).

## Data and estimands
- Original official historical TRAIN 2,699 passages; subset 150 prompt pairs (300 responses), one GLM-5.3 and one GLM-5.3-Flash each. This dataset was already used many times and is *not independent*.
- Public publisher `lxdlxdlxd/ntl-synth-generations-v1`: pinned source hash, 4,758 original rows. Reuse unmodified original fixed 64-pair sample selector (128 records), distinct pid, exact matching user request/target_words/setting across the GLM sibling pair. Distinct values: setting=2, target_words=55. 0 identical normalized text overlaps with competition train. The source's actually served model-version identity and its upstream independence remain unproven.
- Exactly 10 response-level features fixed before seeing any outcomes, 1,000 bootstrap resamples over **matched prompt-pair groups** independently in each source. These CIs address observed prompt-cluster sampling only, not independence of served generators or any causal claim about errors.

## Historic 64-pair GLM routing failure anatomy (previously authenticated)
From [#43612072194](https://github.com/kolihhan/name-kaggle/actions/runs/43612072194), original scorer was **0.683** Public champion as of Nov 14, *not current 0.715*.
- 128/128 true source GLM. Champion recognized GLM provider family for only **95/128 (74.22%)** and missed family for **33/128 (25.78%)**.
- Among recognized GLM-family rows, **88/95 (92.63%)** already had exact model label correct; only **7/95** were wrong. Therefore even a hypothetically perfect *sibling-only gate* could rescue at most **7/128** while leaving 33 family misses untouched.
- Frozen likelihood override changed 23 answers, rescued 6, broke 17. Overall two-label correct **88→77/128**, BA **0.687500→0.6015625** (−0.0859375). This is a score **regression** on previously spent, not independently proven external test.
- These counts reveal a **family-recognition bottleneck** and a small reachable error pool for within-GLM routing. They do NOT reveal whether true native GLM full-vs-Flash likelihood is intrinsically incorrect on the remaining unseen family-missed rows: those rows were never passed to the GLM scorer. Previous 64-pair result does not provide raw per-row likelihood margins, so no invented conditional confidence diagnosis.

## New measured absolute source changes
All units for rate features below are per 100 English words. Differences are external minus official (the 95% CI is prompt-pair bootstrap).

| Feature | GLM-5.3 delta and CI | GLM-5.3 Flash delta and CI |
|---|---|---|
| Word count | +5.957 [−11.413,+23.769] | −3.202 [−22.036,+15.324] |
| Mean word length | **−0.1743 [−0.3143,−0.0405]** | **−0.1666 [−0.2974,−0.0366]** |
| Sentence words | −2.346 [−4.953,−0.139] | −2.555 [−5.810,+0.078] |
| Punctuation per 100 words | −0.171 [−1.050,+0.745] | −0.075 [−0.966,+0.834] |
| Em dash per 100 words | −0.212 [−0.506,+0.101] | −0.104 [−0.418,+0.186] |
| Newlines per 100 words | 0 [0,0] | 0 [0,0] |
| Markdown per 100 words | 0 [0,0] | 0 [0,0] |
| Digit characters per 100 words | **+1.000 [+0.164,+1.998]** | +0.787 [−0.003,+1.720] |
| First-person rate per 100 words | +0.335 [−0.392,+1.092] | +0.027 [−0.652,+0.719] |
| Function-word rate per 100 words | −0.193 [−1.360,+1.099] | −0.396 [−1.641,+0.894] |

**Interpretation:** some marginal text statistics shift between data sources even within each declared model, including shorter mean words for BOTH GLM models. These are ten exploratory marginals, not multiple-testing-corrected discoveries. The length setting of the external publisher has 55 distinct target_words values within only 64 pairs; source channel remains confounded. 0 newline/markdown in both cohorts means those particular measurements provide no discrimination and were reported without selection.

## Fixed matched version fingerprint contrast: Full minus Flash

This is the most important source-robustness check: for each prompt pair, compute (Full−Flash) *before* averaging, then compare those contrasts between sources. Every one of the predeclared **10 paired-change bootstrap CI95 overlaps zero**. Thus **no measured paired style feature has a statistically resolved cross-source shift** using this small cohort and these specific statistics.

| Feature | Official mean Full−Flash | External mean Full−Flash | External−official with CI95 |
|---|---:|---:|---|
| Word count | −8.7533 | +0.4063 | +9.1596 [−4.3202,+21.8429] |
| Mean word length | −0.0348 | −0.0425 | −0.0077 [−0.1100,+0.1022] |
| Sentence words | +0.0687 | +0.2780 | +0.2093 [−1.7146,+1.9835] |
| Punctuation /100w | +0.1353 | +0.0393 | −0.0960 [−1.0886,+0.8580] |
| Em dash /100w | +0.1382 | +0.0295 | −0.1087 [−0.4569,+0.2412] |
| Newlines /100w | 0 | 0 | 0 [0,0] |
| Markdown /100w | 0 | 0 | 0 [0,0] |
| Digits /100w | +0.0263 | +0.2391 | +0.2128 [−0.5173,+0.9019] |
| First-person /100w | −0.1725 | +0.1358 | +0.3082 [−0.1485,+0.7923] |
| Function words /100w | −0.0133 | +0.1893 | +0.2027 [−1.0053,+1.5021] |

Point estimates reverse sign for 3 of the 10 features (word count, first-person, function words), **but their intervals all contain zero**. Never call those sign flips a stable loss of version fingerprint.

## Verdict and next work

**Verified:** (1) the old GLM likelihood routing on independent-source candidate was negative on 64 matched prompts; (2) a significant fraction of true GLM rows were never eligible because of incorrect provider classification; (3) certain absolute language statistics shift between sources; (4) no simple matched Full-vs-Flash style contrast shows a resolved difference across the two studied source cohorts. These findings are not a causal identification of which factor drives GLM gate regressions.

**Current global bottleneck hypothesis:** generation-channel / source variation and provider-level identity extraction dominate the *recoverable* errors more than old GLM sibling ranking, especially since the champion already correctly classified 88/95 within its family-positive subset. This is a *hypothesis for subsequent genuine full-18 transfer*, not permission to engineer a new hard family router on a GLM-only dataset.

**STOP:** no more GLM ratio threshold experiments on the same 64 pairs, no iterative OOF or Public tuning, and no expensive real GLM full-scan again. The 64-pair cohort contains **zero non-GLM negatives**, so it cannot establish precision or false-positive cost of any family-recovery rule. A real next score-facing research experiment needs at least one rights-cleared, previously unused *full-18* source/Prompt/protocol-audited panel with genuine GLM and non-GLM text to measure both recall and false positives. Existing 18-class external aggregate sources remain uncertified. If that prerequisite remains unavailable, report BLOCKED, not a fabricated chance at Public >0.715.

**Competition status:** authenticated best Public BA **0.715**, no change.
