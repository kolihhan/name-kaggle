# NTL: next submission is a new candidate, not a 0.691 rerun

**Decision date:** 2026-11-18 MYT. Official incumbent: `#58375806, Public BA 0.691`. No new submission or experimental winner is established by this note.

## Scope and original owner constraint

- Do not spend Kaggle official entry re-submitting the frozen incumbent just to revalidate it. It already completed and scored **0.691**.
- The eventual next entry should change **one prospective mechanism** relative to the same SHA-pinned Lightning incumbent, satisfy the offline-code competition budget (<9 hours, internet OFF), retain all other predictions unchanged whenever the intervention is out of scope, and be accompanied by a **pre-registered interpretation/stop rule**.
- **Submission is an intended deliverable, not presently forbidden.** A diagnostic-probe submission is possible but must be declared and authorized as such before it consumes one daily slot. The previously written global `pause all NTL official submissions` is **not the owner's policy**; it was the assistant overcorrecting from uncalibrated experiments.

## Authenticated rule/budget evidence

- Official [Kaggle competition](https://www.kaggle.com/competitions/name-that-llm/overview/description) says final deadline **2027-01-13 23:59 UTC**, metric 18-class balanced accuracy, test ~4,000 passages, public leaderboard ~40% of hidden test; inference code must finish within 9 hours, internet disabled.
- Read-only authenticated submissions ledger [Action #44125532299](https://github.com/kolihhan/name-kaggle/actions/runs/44125532299) reported **0 NTL submissions made on 2026-11-17 UTC**, newest scored user submission #58375806 (2026-11-15 UTC) Public **0.691**. The exact competition daily upper limit/remaining count was **not** returned by that ledger; do not guess 5 from Kaggle defaults.
- Separate [Kaggle contest metadata read](https://github.com/kolihhan/name-kaggle/actions/runs/44126116564) was started for competition-specific daily cap. A completed read-only result is required before claiming an exact remaining count.

## Fixed incumbent failure diagnosis (same current 0.691 model)

Read-only [Action #44125532299](https://github.com/kolihhan/name-kaggle/actions/runs/44125532299) recreated the exact existing `kolihhan/ntl-oleg18-frozen0691-probe-nov16` 216-row predictions and pinned Oleg18 truth; **no new inference**.

- 216 Oleg responses, 18 classes, 12 true rows/class. Exact incumbent errors = **127**, correct **89**, external BA **0.412037** versus official Public BA **0.691**. Oleg is clearly not demonstrated to be representative of hidden Public distribution.
- **Observed within-provider errors 64/127 = 50.3937%**. Cross-provider **63/127**.
- Pure balanced-random *distinct-label-pair* same-provider baseline: `54/306 = 17.6471%` (class-family sizes 5 Anthropic, 4 Google, 5 OpenAI, 2 ZAI, 1 Kimi, 1 Qwen).
- Stronger *prediction-marginal-preserving* independence baseline: **21.3235%**. Across 15,000 fixed-seed label permutations (true class balance and actual predicted-label marginal preserved), null within-provider fraction conditional on a wrong prediction median **21.2560%**, 95% simulated range **[16.1765%, 26.6010%]**; exploratory one-sided Monte Carlo `p≈0.0000667` (limited by permutation count).
- Descriptively observed 50.3937% is **2.36×** prediction-adjusted null. This confirms that sibling/family confusion is a real residual pattern in this Oleg domain—not proof of hidden-test prevalence, nor proof that a sibling-specific classifier can correct these errors. It is natural for a classifier that often gets provider families right to make relatively more within-family errors; family-error overrepresentation alone does not demonstrate a complementary new predictor.
- Any future specialist should be **restricted to genuinely confused within-family pairs, with all predictions outside the predeclared pair gate bit-for-bit identical to Lightning**. Need actual paired test predictions, not just permutation rates.

## Before the next official submission

1. Freeze candidate **C1** and SHA-pinned incumbent **C0**. The two notebooks must make predictions on the **same frozen rows**, from the same model-serving/protocol and with no training/prompt leakage. Distinct prepared artifacts must be tested for input-output identity on non-target pairs.
2. Require a **paired 18-label BA delta**, per-source and per-class rescue/regression, effective independent prompt count, and a statistically honest uncertainty interval. Oleg216 alone and the old repeatedly consumed 2,699 public official train OOF cannot certify hidden-transfer uplift for the already-prefit Lightning model.
3. **Owner-set gate still missing.** Explicitly specify the minimum meaningful *paired BA uplift* and the permitted uncertainty/independent-domain criterion before any empirical candidate-dependent scoring. Neither the assistant nor a reviewer chooses it retrospectively.
4. If there is no admissible distinct candidate, say **NO CANDIDATE** and offer a clearly labeled diagnostic-only **PROBE** as a separate owner decision. Do **not** re-submit incumbent under the pretense that it is a new model.
5. At submission time: authenticate fresh Kaggle account quota; ensure no active duplicate for the same frozen kernel; CPU/private full hidden-sized smoke with no internet and <9h runtime; record source SHA and unique submission ref. Public outcome `>0.691` may identify a better *observed leaderboard package*, but not independently prove the targeted change caused uplift. If `<=0.691`, do not adapt thresholds on Public and recycle the same hypothesis; retain 0.691 as incumbent. Never treat `PENDING`/blank score as gain.

## Avoid invalid cross-generation comparisons

- Current champion 0.691 is an externally sourced prefit multi-module ensemble. Its original official 2,699 public train texts may have contributed to the prior provider modules; we **do not have a leak-free same-model prompt-group OOF** on them. Simply running Lightning on official train would be in-sample and not an honest local leaderboard predictor.
- Old LoRA24 701/1223 within-provider OOF is **different model**; ModernBERT 0.214 vs 0.054 is a **different source corpus**, not paired against incumbent.
- A new model that only improves Oleg may be a useful probe, but cannot be called a confirmed improvement over 0.691 Public before official score or a validated, independent proxy.
