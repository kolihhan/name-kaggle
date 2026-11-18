# NTL — Measure-the-ruler audit, not another model experiment (2026-11-18)

## Predeclared question, sample, detectability

**Question:** Did the old official GroupKFold split itself put the *same official prompt* across train/validation, or is the Phase 7→15 local/Public reversal better explained by external sampling/representation and source mismatch?

- Historical evaluation population: **N = 2,699 official responses; 150 prompts, 18 true classes; 5 frozen folds, 30 distinct holdout prompts each**. Phase 15 loads **26,025 external fit responses** in every fold.
- Historical observed effect to explain: Phase 7 → 15 grouped OOF **+0.0352697**, while official Public **−0.012** (Public delta may itself be noisy). Before any new model work, audit all prompt IDs and normalized response hashes across boundaries and recover the original paired OOF outputs.
- Detectability: a pure hash/ID contamination audit can enumerate exact duplicates with no statistical MDE. For predicting model delta, under independent-row assumptions and assumed discordance `q=0.2`, nominal MDE at n2699 ~**+0.024**; if only 150 prompt clusters were independent, ~**+0.102**. Both are illustrative: actual cluster dependence and class balance matter. A retrospective true paired prompt-bootstrap on the old model pair estimates actual, *still optimistic* 80%-power minimum detectable BA delta **+0.0261** (doesn't cure adaptive selection/source leakage). **An expected +0.01 change is below this ruler's effective sensitivity; do not launch a comparable new model experiment to prove +0.01.**

## Executed: actual pinned group/external exact leak census

[Authenticated GitHub CPU run #44213134245](https://github.com/kolihhan/name-kaggle/actions/runs/44213134245) **COMPLETED/SUCCESS**, official train and original pinned `nishantdahal/ntl-handwriting-blend` `pool_v4.parquet` SHA-verified, rebuilt the exact Phase 14/15 external subset; no model fitting/inference, no paid compute, no competition submission.

- Official `GroupKFold(prompt_id)`: 2,699 rows, 150 prompts, five 30-prompt heldouts, **0 train-validation prompt-ID collisions across all folds**.
- Official train-to-validation normalized response exact hash matches **0** (also no identical response string shared across distinct official prompt groups).
- Phase 15 **all 26,025 external rows are fitted in every fold**, 2,801 external prompt IDs, 11/18 model labels. Direct external prompt-ID/Prompt-key string collisions with holdout official prompt IDs **0**. Normalized **exact response** overlaps between external fit and official validation **0**, in all five folds.
- Thus **the claimed literal prompt split bug and literal duplicate-response leakage did not reproduce**. The audit does **not** exclude paraphrases/near-neighbors of heldout prompts, source/proxy/wrapper mismatches, model-version noise, or upstream prefit use of official data. Distinct prompt-ID namespaces make string mismatch only a narrow negative test.
- Historical local-vs-Public discrepancy remains real as *recorded outcomes*, but its cause remains unproven.

## Executed: the two original model OOF CSVs, aligned on the same official rows

[Authenticated original-artifact rerating #44213810962](https://github.com/kolihhan/name-kaggle/actions/runs/44213810962) **COMPLETED/SUCCESS**. Inputs are the original:
- Phase 7 [Actions #41195328534](https://github.com/kolihhan/name-kaggle/actions/runs/41195328534) artifact `phase7-gated-specialist`, matching the original 150-group fold fingerprint.
- Phase 15 [Actions #41270900644](https://github.com/kolihhan/name-kaggle/actions/runs/41270900644) artifact `phase15-terms-clean-frozen-shot`.

All 2,699 IDs, 18 ground-truth classes, prompt IDs, and fold IDs matched exactly, with no missing predictions. Actual same-row paired results:

| Item | Verified |
|---|---:|
| Phase 7 OOF BA | **0.370052** |
| Phase 15 OOF BA | **0.405322** |
| Delta | **+0.035270** |
| Rescues / regressions | **448 / 353** |
| Paired prediction correctness discordance | **0.296777** |
| Prompt-cluster bootstrap (8,000 reps) 95% interval | **[+0.016761, +0.053469]** |
| Retrospective optimistic 80%-power MDE | **0.026064** |
| Five per-fold macro BA deltas | **+0.03333/+0.04259/+0.02037/+0.02963/+0.05026** |
| Real Kaggle Public BA delta | **−0.012** |

The historical local gain is not a single-fold accident. **However, this is a reused historical OOF which repeatedly drove earlier experiment selection; resampling it does not make it a new blinded benchmark.** The observed Public reversal might stem from distribution and/or score noise; without genuine independently retrained current-champion CV cannot establish a transferable calibration curve.

## Direct error anatomy from the stored per-class OOF aggregates

Compared Phase 7 `artifacts/phase7/gated_specialist.json` vs Phase 15 `artifacts/phase15/terms_clean_sparse.json` (same fold fingerprint):

- **11 model labels with admitted external training data:** average per-label recall Δ **+0.128017**.
- **7 labels without admitted external data:** average per-label recall Δ **−0.110476**.
- Full 18-label macro Δ **+0.035270** is an offsetting mix, not uniformly better model attribution.
- Largest gains: Claude Sonnet 5 **+0.29333**, Claude Fable 5 **+0.22819**, Claude Opus 4.8 **+0.22667**.
- Largest regressions: Kimi K3 **−0.19333**, GLM 5.3 Flash **−0.18667**, GLM 5.3 **−0.16000**.
- **This is a measured internal representation/source-support tradeoff, not proof that external support CAUSED the delta**. Different Phase7 and Phase15 model recipes also changed; source support is confounded with architecture and sampling. Do not claim a causal mechanism from observational per-class deltas.

## Critical impossibility: incumbent Lightning cannot be given honest OOF by merely predicting on train.csv

Current `0.691` Lightning is a pinned, externally pretrained multi-module public system. It may have trained using official training samples or shared prompts. **No independently retrained fold-specific Lightning checkpoints and complete source provenance are verified in this repo.** Calling inference on the same 2,699 official training responses may be *in-sample*, not out-of-fold, even if the output is stored with `fold_id`. The existing old Phase7/15 fold predictions are honest with respect to *their official train split*, subject to unverified external prefit exposure; they are NOT directly comparable to Lightning pseudo-OOF.

Thus do **not** produce a fake “Phase7 / Phase15 / LoRA24 / Lightning all matched OOF” ranking by running a prefit model on historical training rows. A 4–5-point CV↔LB chart is blocked by unavailable matched retraining lineage. Until this is resolved, use existing Public scores as descriptive independent comparisons and label local CV as non-comparable.

## No new Oleg/external mechanism searches. Next decision

- Existing Oleg216 and mixed-source experiments are spent, don't launch more tiny targeted classifier/synthetic experiments on them. The recent `liuyihuileo/ntl-external-training-candidates-18` 18-label publisher source was read-only audited in [Actions #44212521956](https://github.com/kolihhan/name-kaggle/actions/runs/44212521956), but **only 1,381 / 3,946 balanced articles have directly matching requested-vs-reported-served model label strings (13 labels), 8 underlying sources, and independent raw provider attestations missing**. Qwen has only **30** rows in its larger publisher-selected training candidate table. Do not promote it as a reliable full-18 benchmark.
- For a future model hypothesis, require BEFORE running: **effective independent n**, **optimistic MDE and assumed paired discordance/cluster inflation**, and **expected improvement** in the same metric and distribution. If expected gain is below MDE, **NO RUN** unless the explicit question is an engineering/diagnostic failure mode with an exact counting oracle.
- Kaggle Public is legitimate limited external feedback. Adopt **at most one clearly preregistered exploratory probe in any seven-day interval** absent genuinely paired independent evidence, and never tweak using its score. This is an internal cost cap, not prohibition of properly evidenced new submissions. The owner's 0.691 is the fixed reference; Public uplift does not by itself prove a causal mechanism.
- Do not open a new modeling lane on the basis of this audit alone. The next valuable work is **independent provenance of the Lightning prefit modules and original fold-specific training lineage**, or an outside-source heldout that actually provides source/served-ID rights and can run the current Lightning exactly once under a frozen protocol. No such qualifying 18-class independent benchmark was established in this turn.

**Official competition best unchanged: 0.691; new submission count 0. Gemma stays low-investment, no new development GPU.**
