# NTL hard-negative identity research — REAL CPU results, 2026-11-26 MYT

**State:** NEW REAL EVIDENCE, **NO model training, no Kaggle GPU or competition submission, no measured Public BA gain**. Confidence Fusion remains **RETIRED**; incumbent authenticated score is **0.718**.

## 1. New publisher-declared 18-class paired source was actually downloaded

The dataset `prvsiyan/prvsiyans-extended-name-that-llm-dataset` exposes `passages_clean.csv` (7,519,791 bytes) with **5,070 rows / 18 exact labels** (none duplicated or empty). It records `model`, `prompt_id`, `split`, `provider_route`, `requested_model_id`, `served_model_id`, `served_provider`, creation time and generation configuration.

Fresh source audit: [GitHub Actions #45660827199](https://github.com/kolihhan/name-kaggle/actions/runs/45660827199). Exact new-prompt split: **new 4,158** + **new2 144** = **4,302 rows, 239 shared prompts × 18 models**. Excluded **252 train_regen** rows (14/model) plus **516 proxy_v1** from the new-prompt matched research panel. The publisher-reported `served_model_id` field alone is *not* a cryptographic provider signature.

[Pair eligibility audit #45661049279](https://github.com/kolihhan/name-kaggle/actions/runs/45661049279):
- Gemini 3.7: **239 new-prompt** with `served_model_id=gemini-3.7-flash`, `served_provider=Google Vertex AI express mode (google-genai SDK, first-party)`.
- Gemini 3.8: 239 new-prompt in total, but only **165** Google Vertex first-party; the remaining 60 opaque Kaggle Proxy, 14 Google AI Studio entries are excluded from the strict Vertex-to-Vertex pair.
- Kimi K3: **239** new-prompt with `served_model_id=moonshotai/kimi-k3`, of which **231** are `served_provider=Moonshot AI` and eight `Alibaba`. Restrict the experiment to 231 named-Moonshot rows. This does NOT independently prove the request bypassed a router.
- GPT 5.5 / Luna / Sol: 239 new-prompt each, **150 per label via OpenAI first-party API**, remaining 89 per label opaque Kaggle Proxy. Restrict comparisons to direct first-party rows.
- Matched external prompts after both-side route screening: **Gemini 3.7 vs 3.8 = 165**; **Kimi vs GPT 5.5 = 150**; **Kimi vs GPT Luna = 150**; **GPT Luna vs Sol = 150**.
- Old official TRAIN has 2,699 rows / 150 distinct prompt groups. The external panel and official train have different prompt-ID namespaces and are not assumed to be independently unseen by the full 0.718 pretrained ensemble. It is a real *source-transfer diagnostic*, not a clean incumbent OOF.

**Caution:** Simply grafting prvsiyan into another public recipe has negative recorded results: `crtproduction/name-that-llm-open-pool` latest Nov25 `dataq4_paired.csv` variant X ("plus prvsiyan extended rows") reported local BA **−0.0092** vs its own baseline (95% interval includes 0), and prompt-centred BA **−0.0130** (reported interval below 0). Do not infer this dataset automatically improves a strong stack.

## 2. Executed predeclared prompt-paired source-transfer test

[Successful GitHub Actions #45661298815](https://github.com/kolihhan/name-kaggle/actions/runs/45661298815); 4 attribution pairs × 10 predeclared text statistics, both official 150-prompt and source-disjoint 150/165-prompt cohorts. Within each shared prompt, compare A–B statistics and run an exact two-sided sign test; for each feature take the **maximum** of the two domain p-values, apply Benjamini–Hochberg correction across **40** comparisons, and require both domain mean differences to have the same direction.

**Result: 6/40 transfer-stable tests at joint BH q<0.05, all Kimi-vs-GPT; 0/10 for Gemini 3.7-vs-3.8, 0/10 for GPT Luna-vs-Sol.**

Transfer-stable tests:

| Pair | Feature | Mean difference (official / external) | Joint BH q |
|---|---|---:|---:|
| Kimi − GPT 5.5 | Em dashes per 100 words | +0.7253 / +1.0014 | <0.000001 |
| Kimi − GPT Luna | Long-word fraction | −0.0303 / −0.0219 | 0.000021 |
| Kimi − GPT Luna | Mean token length | −0.1988 / −0.2109 | 0.000164 |
| Kimi − GPT Luna | Em dashes per 100 words | +0.4110 / +0.6716 | 0.029292 |
| Kimi − GPT Luna | Curly apostrophes per 100 words | −0.2571 / −0.8554 | 0.030663 |
| Kimi − GPT 5.5 | Mean token length | +0.1383 / +0.0980 | 0.045678 |

**Interpretation:**
- There is **real, repeated Kimi-vs-GPT discriminatory statistical information** across the two observed prompt sources. Some signals also track provider, routing, word-count instructions, and preprocessing; neither the p-values nor the observed differences prove model identity is uniquely causal. Official train has been repeatedly mined in previous work; the 10 features overlap older explored stylometric tests, so this is descriptive **not independent preregistered discovery**.
- The absence of such confirmed features for Gemini 3.7-vs-3.8 or Luna-vs-Sol **does not prove they are intrinsically indistinguishable**; we only tested 10 surface-text features, no strong token-level or deep semantic representations.
- Old *spent* publisher-labelled 252 cohort showed **Kimi K3 correct only 1/14**, often predicted GPT, while Gemini 3.7 was 0/14. That juxtaposition makes Kimi cross-family attribution a plausible representation/gating weakness. It does NOT estimate a true hidden BA improvement.
- A one-feature heuristic targeting Kimi is explicitly **NOT** accepted: a threshold tuned to these two observed domains would repeat the overfitting problem. Likewise don't use official train as fresh OOF for the public pretrained stack.

## 3. Fresh third-party source data also checked

Actual [source inventory #45660699936](https://github.com/kolihhan/name-kaggle/actions/runs/45660699936) showed `crtproduction/name-that-llm-open-pool` had a Nov25 update. Its publisher-reported `removal_summary.csv` lists **11,001 rows `gemini37_route_unverified`**, and `per_model.csv` leaves **3,541 Gemini 3.7** vs **23,601 Gemini 3.8** rows in the current pool. Unverified rows are separately quarantined. These are publisher-derived curation values, **not our own independent served-model attestation**.

Author's two-seed `dataq4_paired.csv` reports:
- Vertex-only Gemini 3.7 selection (G) **+0.0093 local BA**, +0.0178 prompt-centred BA vs their B control, not versus our 0.718 stack;
- Typography-matched sampling (T) **+0.0263 local BA**, +0.0118 prompt-centred BA, not hidden transfer;
- Adding prvsiyan Extended rows unconditionally (X) **−0.0092 local BA**, −0.0130 prompt-centred BA.

The observed contrast strengthens the case that **data fidelity/source protocol matters more than blindly increasing dataset size**, while remaining method- and domain-specific.

## 4. Next model-class decision (not executed)

**Hypothesis H1:** A *new* prompt-robust representation-level Kimi-vs-GPT discriminator (frozen strong embedding / within-prompt hard-negative contrastive training) can recover misidentified Kimi without breaking existing GPT correct rows.

**Competing H2:** Differences are mostly response-protocol/host/vendor style and do not transfer to hidden test; another trained expert would regress despite strong local p-values.

**Shortest discriminating next experiment** (separate, carefully controlled; no direct LB tuning):
1. Confirm the exact 0.718 pretrained component lineage does not contain the `prvsiyan` samples or the underlying Open Pool versions; otherwise this panel cannot be used as independent incumbent evaluation.
2. Build prompt-disjoint *and* route-disjoint train/validation for Kimi/GPT and Gemini 3.7/3.8; verify rights for source data including API-derived texts (public CC licence is not sufficient by itself to prove competition permission).
3. Use **one strong frozen representation** and one rank/contrastive objective, with a scrambled-label/source negative control. Do not add a small stylometric classifier or tuning rule.
4. Freeze full 18-class decision-change contract: actual same-row incumbent predictions, rescues/regressions, all-class BA, runtime/cost. Kill if only within-source/Kimi pair metrics improve or if GPT regressions exceed rescues.
5. No new official Kaggle submission, no use of the retired Confidence Fusion notebook, and no GPU in this executed turn.

**Bottom line:** The prior "no credible Kimi or Gemini data" blocker is materially superseded by a verified **4,302-row, 18-class prompt-matched** public candidate. The hard-negative research hypothesis now has concrete diagnostic evidence, but **has not yet improved Public BA**.
