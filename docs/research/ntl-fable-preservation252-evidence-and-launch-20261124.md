# Fable Preservation 252: evidence audit + one real private inference launched

**As of 2026-11-24 21:01 MYT**; no new scientific accuracy or Public BA is verified yet. Distinguish a successful Kaggle **push** from a completed Kaggle **inference**.

## Authenticated read-only existing evidence
[GitHub Actions #45397740811](https://github.com/kolihhan/name-kaggle/actions/runs/45397740811) surveyed 6 exact existing owned Kaggle kernels: scored original 0.715, reproduced 0.718, 0.715 failed stage trace, older labelled 252 GLM reroute, external 128 GLM, strong Lightning1200 cache. NONE exposes a final aligned `final-logits.npz` plus `prefix-votes.npz` in the current downloadable kernel outputs. The 0.718 private kernel was COMPLETE but the accessible output is only its preview, not labelled final hidden decisions. **Do not claim the 0.718 score can already be replayed without native inference.**

Older pinned 252-row 18-class observational score output, from `kolihhan/ntl-glm-conditional-compare-nov14` (`glm-conditional-summary.json` authenticated download):
- Exact 0.683-era champion `58295802`, 150/252 originally correct.
- Conditional GLM-only reroute: 154/252, 13 changed, 5 rescues / 1 regression.
- True Fable `claude-fable-5`: 6/14 correct unchanged, **0 rescues/0 regressions**. True GLM-5.3: two additional rescues. True GLM-5.3-Flash: three rescues, one regression.
- This older **0.683** conditional rule does not reproduce the **0.718** `fable_return` mechanism. Different scorers, interventions and temporal provenance; do not pool into a single claimed marginal gain.

## One real private experiment
- [Frozen preregistered diagnostic contract](ntl-fable-preservation252-frozen-contract-20261124.md).
- Pinned private 0.718 source SHA256 `74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56`; original notebook 58 cells, all 9 model / 10 kernel / 26 original dataset dependencies retained; one public synthetic 18-label 252 panel added (source SHA pin and exact response overlap guard).
- [Successful Actions #45398866351](https://github.com/kolihhan/name-kaggle/actions/runs/45398866351) confirms CPU builder/algorithm smoke PASS, source/code pin and metadata PASS, GPU available **22.18h**, and **one private kernel actually accepted** at `kolihhan/ntl-fable-preservation-252-20261124`. First private worker state `QUEUED`.
- [Read-only status #45399086667](https://github.com/kolihhan/name-kaggle/actions/runs/45399086667): worker still `KernelWorkerStatus.QUEUED` as of 21:01 MYT, no new labelled result yet.
- Earlier attempts #45398532129 and #45398736153 failed prior to any GPU run due to Kaggle id/title slug mismatch. Fixed metadata, revalidated before acceptance. These failures are **engineering errors**, not negative evidence for the mechanism.
- Original 0.718 author inference and Apex run on same private 252 text-only cohort, no training new ML models. Per-row prediction/label metadata remain private. After actual Apex, replay old no-Fable-protection path on **same existing** in-memory `refined_lp` without model training or inference, then paired exact rescues/regressions, BA18 and protected-row count. Output aggregate only `ntl_fable_preservation_252_aggregate.json`.
- Bound: Kaggle private T4, **5.8h GLM native deadline / 6h runner** and no official Kaggle competition submit. It is a spent synthetic/self-reported external cohort, **not** a clean independent test.

## Exact interpretation once Kaggle finishes
- If output aggregate exists, validate same 18 labels / 252 rows and paired exact counterfactual (non-protected rows must not differ).
- If private worker ERROR, inspect first real failing cell and error class; do **not** characterize its BA as low or method failed.
- If V1 fallback or protected count zero, classify `NON_DISCRIMINATING`: this panel did not activate the mechanism; do not conclude it is ineffective.
- If `fable_return` activated, report actual wrong→right and right→wrong among protected rows, per-class changes and costs. Don't force `0.718+delta` because data distribution differs and donor code includes external prior-art mechanisms.
- Regardless of positive or negative results, do not tweak margins `0.30/0.25` on this spent panel or submit based only on it; analyze rival causes/source shift and seek qualified independent evidence.
