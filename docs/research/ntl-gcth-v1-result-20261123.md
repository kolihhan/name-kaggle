# GCTH-v1 — Actual 18-Class Generator-Conditioned Token Head Results

2026-11-23 MYT. Frozen contract established BEFORE observing scores at [`51920a5ce0092d9be0fa7c3aaec1b4ca5f2d07b2`](https://github.com/kolihhan/name-kaggle/commit/51920a5ce0092d9be0fa7c3aaec1b4ca5f2d07b2). **Real completed GitHub Actions [#45123925475](https://github.com/kolihhan/name-kaggle/actions/runs/45123925475) (SUCCESS)**, all input integrity and probability normalization checks passed. Program: [`scripts/ntl_generator_conditional_token_head_v1_20261123.py`](../../scripts/ntl_generator_conditional_token_head_v1_20261123.py).

## Research question and frozen scope

Does generator-conditioned **categorical next-token likelihood** learned on frozen pretrained DistilGPT2 causal context states actually improve source identity, beyond each model's next-token marginal frequencies? Main prediction: genuine context should beat both class-unigram priors and same-generator context-shuffled equal-capacity heads. All arms score each text alone, no Prompt ID at inference, no 0.715 original Champion predictions/weights/ensemble.

- Cohort: **2,699 official historical TRAIN rows**, **150 prompt-disjoint groups**, **18 exact classes**, 5 frozen grouped folds with no overlap; all inputs byte-validated before run.
- Frozen transformer: `distilbert/distilgpt2@f241065e938b44ac52db2c5de82c8bd2fafc76d0`, pretrained public reference, no backbone training, CPU only. Hidden vectors at timestep `t-1` predict `t` (teacher forced causal prefix), fixed random 64D projection.
- Source-specific proper probability is a **128-way token category** logsoftmax (127 most common BPE IDs among each fold's TRAIN texts, plus OTHER); it is NOT proprietary generators' full-vocabulary likelihood. Train-only model-specific empirical Bayes unigram priors.
- B and C: same 18 independent frozen-recipe small output residual heads, 3 epochs and identical optimizer settings. C sees same-source but different-response randomized contextual donors during **training**; heldout evaluation for both on true untouched context. Negative C's deliberately mismatched training distribution prevents strong causality claims based on B−C alone.
- D: identical original frozen 4-view Strong Sparse independent full-18 fold-local model; **NO blending**.
- Context extraction: **171,310 token positions**; **2,446/2,699 responses truncated at 64 BPE tokens** (~90.6%). Scientific wall time **219.3 s**; frozen reference extraction **66.6 s** and about 28–33 s per fold including Sparse. Fold-local learned head wall time per arm ≈0.3 s. No GPU, no external generated text or terms-uncertain data, no Kaggle competition submission or hidden tests.
- All class-conditional null shuffles use ≥98.954% different-response donor hidden vectors, exceeding the predeclared 95% control integrity minimum.

## Actual 18-way historical OOF

| Method | Balanced accuracy |
|---|---:|
| **A** model-conditional token unigram | **0.2022620** |
| **B** real pretrained context → source-specific token output head | **0.1826547** |
| **C** same-capacity source-specific token head learned with shuffled within-source contextual donor | **0.1452374** |
| **D** direct four-view Strong Sparse SVC, heldout same rows | **0.3670892** |

| Paired contrast | Delta BA | 95% Prompt-group bootstrap CI | Rescues (B correct) | Regressions (B wrong) |
|---|---:|---|---:|---:|
| **B−A** | **−0.0196073** | **[−0.0358866, −0.0047886]** | 240 | 293 |
| **B−C** | **+0.0374174** | **[+0.0199022, +0.0551944]** | 361 | 260 |
| **B−D** | **−0.1844345** | **[−0.2044625, −0.1633241]** | 250 | 748 |

Five fold BA A/B/C/D:
- 0: 0.2148148 / 0.1814815 / 0.1611111 / 0.3833333
- 1: 0.1703704 / 0.1740741 / 0.1388889 / 0.3444444
- 2: 0.1851852 / 0.1925926 / 0.1222222 / 0.3759259
- 3: 0.2074074 / 0.1870370 / 0.1648148 / 0.3814815
- 4: 0.2334610 / 0.1780332 / 0.1388889 / 0.3503193

- B beats A only **2/5** folds, B beats C **5/5**.
- Maximum B class recall regression versus A **0.1733333**, violating 0.15 safety veto.
- All-frozen conjunctive **mechanism PASS = FALSE**, **competitive vs Sparse = FALSE**.

## Interpretation and stop decision

**FACT:** Context-aware source output heads outperform identical heads trained on shuffled same-source context, and positive B−C bootstrap remains. However, B does NOT beat the simple unigram control: it performs materially *worse*, with negative paired CI and 53 net correct errors in the wrong direction. It also loses to a much cheaper independent discriminative Sparse source classifier by ≈18.44 percentage points. This directly falsifies the new claim that this specific source-conditioned categorical next-token modeling is a net improvement.

**ALTERNATIVE EXPLANATIONS (not causally isolated):** Only 64 BPE token prefix, 90.6% truncation, 128-token coarsened vocabulary with OTHER class, limited per-source training, 64-dimensional fixed random projection, and domain/host mismatch between the general reference LM and modern proprietary source models could all reduce contextual identity. The bad B−A effect and safety veto are sufficient to stop; do **not** post-hoc tune caps, vocab, projection, regularization, source weights or choose favorable source models based on the score. The randomized-context arm itself sees synthetic OOD contexts, so B>C is supporting but **not** proof of transferable identity mechanism. A prior raw FCS-v1 context summary did beat BOS baseline by +.02038, but its stand-alone BA .1952 was also far below Sparse .3671; neither result warrants an ensemble.

**DECISION: STOP GCTH-v1 exact recipe.** No official Kaggle submission, original authenticated Public champion **0.715 unchanged**, no inference from this OOF to Public BA. Keep open the *fundamentally different* source-specific **actual-generator likelihood** hypothesis, already showing limited GLM5.3 vs Flash binary evidence from original model weights. This is not permission to blindly retrain external GLM/Gemini/Claude/Kimi outputs: dataset provenance and service-output terms are not cleared, and the clean fresh full-18 external benchmark is still blocked. Prefer either (a) evaluate a genuinely available licensed and independent exact-label source panel under a frozen design, or (b) review actual paired full-vocabulary token evidence and strong anchor failure cohorts rather than another family of feature/ensemble variants on the same 2699 spent rows.

## Reproducibility links

- [Frozen contract](ntl-gcth-v1-contract-20261123.md)
- [Full audited GitHub execution, no raw data exported](https://github.com/kolihhan/name-kaggle/actions/runs/45123925475)
- [Frozen small generator-token-head source](../../scripts/ntl_generator_conditional_token_head_v1_20261123.py)
- [Earlier FCS-v1 actual result](ntl-fcs-v1-result-20261123.md)
- [Conditional Denoising external-data status](../../artifacts/candidates/conditional-denoising18-v1/CONTRACT.md)
