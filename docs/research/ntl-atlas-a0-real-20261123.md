# ATLAS A0 — Exact original bounded token readout, real GLM–Flash cached likelihood (2026-11-23)

**Status when committed:** real CPU [Actions run #45152458986](https://github.com/kolihhan/name-kaggle/actions/runs/45152458986) **QUEUED**, no score outcome. Run is triggered from repository workflow [here](../../.github/workflows/ntl-atlas-token-a0-real-20261123.yml), exact implementation [here](../../scripts/ntl_atlas_exact_token_readout_a0_20261123.py). Do not claim success from a queued job.

## Source and reproducibility

- Paper [Ko et al. 2026 ATLAS](https://arxiv.org/abs/2610.19583), author's exact formula in [src/atlas/model.py:_prod_readout](https://github.com/komseok0109/ATLAS/blob/main/src/atlas/model.py). Exact candidate k token t readout is exp(-sum_j relu(logp_jt - logp_kt)/T), averaged across scored suffix tokens and the full fixed T values {1,2,4,8}. No new LLM calls. This **only ports the readout**, not trained source prefixes, shared-A per-source-B LoRA, or five-model all-condition ATLAS.
- Original safe GLM token cache recovery `scripts/ntl_merge_public_glm_caches_20261122.py`: public Kaggle published versions v1 800 and v2 1899 rows, **2699 unique exact official IDs**, exact two-model aligned token IDs/offsets/prefixes; original real [Actions #45016238278](https://github.com/kolihhan/name-kaggle/actions/runs/45016238278) SUCCESS. The new workflow verifies all **eight exact published file SHA256 hashes** against that previous run BEFORE scoring, plus official train and permanent fold file SHA checks. Raw response data and token cache remain transient and deleted.
- 300 original official TRAIN responses are actually `glm-5.3` or `glm-5.3-flash`, 150 prompt groups, same frozen 5-fold assignment. Does **not** use independent hidden labels or new source data.

## Frozen C0/C1/Cstrong alternatives

| Arm | Scoring/learning | Scope |
|---|---|---|
| C_mean | Simple mean of each candidate's per-token log likelihood, then argmax | 2-way GLM, zero fit |
| C_prod | Paper's original multi-temperature bounded argmax-prod, then argmax | 2-way GLM, zero fit, **primary** |
| C_sign | Fraction of aligned tokens won by each source | 2-way GLM, zero fit, descriptive |
| C_global | Exact historical publisher-global 2x24 likelihood statistics + five pair-global features, fixed 5-fold StandardScaler + LogReg C=1 | Existing supervised **strong control**, expected 0.9433333 binary BA, reproduction gate |

Primary outputs: BA, per-sibling recall, same-row net rescues and regressions, C_prod against C_mean **and strong C_global**, prompt-cluster resample diagnostic conditional interval, by-fold delta, exact cache alignment and token counts. A positive score on spent official TRAIN **cannot justify leaderboard promotion** without external validation and 18-way comparison to official frozen 0.715.

**Prior negative evidence:** [Distributed-token window feature experiment #45016238278](https://github.com/kolihhan/name-kaggle/actions/runs/45016238278): C_global **0.9433333**, adding 8-token window features **0.9266667** and after token misalignment **0.9266667**. This is a **different experiment**: old linear feature fitting vs current paper's zero-fit bounded token readout. Do not pool observations or retune after loss.

## Current simultaneous paper work

- **POS syntax mechanism [#45146166103](https://github.com/kolihhan/name-kaggle/actions/runs/45146166103) completed:** actual source-shift 216 BA C0 0.106481, C1 0.125, order-shuffle C2 0.083333; historical strong old 0.691 outputs 0.412037. POS standalone **reject** as replacement. Official OOF C0 0.153351, C1 0.160040, shuffled 0.106692. Not actual current 0.715 comparator.
- **SupCon implementation [smoke #45149856087](https://github.com/kolihhan/name-kaggle/actions/runs/45149856087) completed success:** gradient/synthetic prompt pair mechanics PASS, **NO actual 18-class model-quality result**. Strong 0.715 source-path audit #45148589048 completed success, but notebook lists training-source references, *not verified already extracted, leak-free strong feature matrices on independent same-row source*. Do not pretend the SupCon research evaluation has run.

**No paid API, new model generation, new QLoRA, new foundational model inference or official competition submission. Champion unchanged: Public 0.715.**
