# Source-transfer view ablation — observed result

Date: 2026-11-06 MYT
State: **ONE-SHOT DIAGNOSTIC COMPLETED; STYLE-ONLY HYPOTHESIS NOT SUPPORTED; NO COMPETITION PROMOTION.**

## Run identity and precommit

- Frozen design: `artifacts/research/source-transfer-views-20261106/CONTRACT.md` (commit `3c70d647f7e671e4ace62f42cd2caa5ef05eb314`, before executing any models).
- Exact runnable implementation: `scripts/run_source_transfer_views.py`.
- Successful GitHub Actions CPU run: [42157902705](https://github.com/kolihhan/name-kaggle/actions/runs/42157902705). Run head `f6e88a02859ed343b5d6555031dbb0dfac752d4e`, conclusion **success**; all steps including actual fitted model run and aggregate JSON upload passed.
- Aggregate-only artifact: `13475960270`, `source-transfer-views-aggregate-only`, ZIP `sha256:ea2d07c5ececfb36657a4b218521ab72f911b7d9d1589562b4086b3f17dcd669`. Contains no responses or per-row predictions.
- Fixed 30,133 Stage-1 A1 training rows from Phase14 (all 18 labels). No official training/test content; no GPU; no Kaggle competition submission.
- Heldout source formats: alessandrolair 3,866 rows/12 scored labels and elliotthankinson 8,347 rows/7 scored labels. Both were previously inspected; **exploratory diagnostic only**, not genuinely fresh out-of-source validation.
- Frozen file SHA256: Stage1 `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`; Aless `e98b1924269c30116f50623052f7fc4ea6148f7575d2fe483c9bd899b4ea2e21`; Elliot `e8aeafcd75b9832c297324da360051a48fec3b58d1d8cfe251e03c2558f40185`.

## Observed eligible-label balanced accuracy

| View | Aless (12 labels) | Elliot (7 labels) |
|---|---:|---:|
| FULL char+word+shape+small | **0.464769** | 0.295586 |
| LEXICAL char+word | 0.457431 | **0.297183** |
| SURFACE shape+small | 0.388482 | 0.199968 |
| FULL minus LEXICAL | +0.007338 | -0.001597 |
| SURFACE minus LEXICAL | -0.068949 | -0.097215 |

These numbers are **within-source** comparisons for each frozen view. The two columns have different labels and class mixes, and their absolute score difference MUST NOT be interpreted as a measured source-shift penalty. Neither column is official 18-label Kaggle BA.

### Family-level mechanism detail

| Source / family | FULL | LEXICAL | SURFACE |
|---|---:|---:|---:|
| Aless Claude | 0.3964 | 0.3793 | **0.4135** |
| Aless Gemini | 0.5827 | **0.5898** | 0.4300 |
| Aless GPT | **0.4487** | 0.4405 | 0.3436 |
| Elliot Claude Fable only | 0.1309 | 0.1314 | **0.1354** |
| Elliot GLM | 0.3283 | **0.3312** | 0.2042 |
| Elliot GPT | 0.4144 | **0.4157** | 0.2667 |
| Elliot Kimi only | 0.0382 | 0.0392 | **0.0559** |

The shared models between the two sources are **only GPT-5.6 Luna/Sol/Terra**. FULL recalls on these exact three labels:
- Luna: Aless 0.5215, Elliot 0.6327.
- Sol: Aless 0.2767, Elliot 0.3568.
- Terra: Aless 0.2469, Elliot 0.2538.

This directly cautions against treating lower Elliot overall macro recall as uniform cross-source deterioration. The very low Elliot `kimi-k3` and `claude-fable-5` recalls, plus its different 7-label mix, drive a large part of apparent separation.

## Decision under the predeclared contract

1. **REJECT the simplistic style/surface-only advantage hypothesis** for this pair of diagnostics: surface-only underperforms lexical by 0.06895 and 0.09722 on their respective eligible-label metrics.
2. **FULL does not dominate LEXICAL consistently**: one modest positive delta and one essentially flat/slightly negative. Adding shape+function words to char+word has no compelling global gain in this test.
3. **No proof that semantic/topic content is more robust than actual stylometry.** Char TF-IDF can encode punctuation, formatting and style; shape+small also encodes function words and length artifacts. The two arms differ in information and dimensionality. This is an ablation of these exact representations only, not a clean intervention separating content from style.
4. Claude's source-specific family pattern suggests heterogeneity, **NOT permission to train a Claude-only gate** or tune family thresholds on these source results.
5. Source/prompt/served-model provenance risks and prior exposure remain. No bootstrap confidence intervals were precommitted for this diagnostic; do not call differences statistically significant or generalization proven.
6. **Retire the two evaluated sources for promotion decisions.** No subsequent source-conditioned hyperparameter sweeps, new splits, label filtering, or LB queries using these numbers.

## Research strategy update

The prior binary story of “stylistic fingerprints vs lexical artifacts” is too crude. There are probably interacting signal types and label/provider coverage effects. Importantly, existing earlier candidates already tested and rejected prompt-residual LDA (worse on 11/13 sources; `artifacts/candidates/prompt-residual-style-v1/RESULT.md`) and prototype contrastive v2 (source-transfer delta +0.00269 with 95% bootstrap including zero; `artifacts/candidates/prompt-invariant-contrastive-v2/RESULT.md`). Do not reinvent them as new breakthroughs.

A fundamentally different **already-frozen** research line is DCT temporal token-trajectory fingerprint v3, `artifacts/candidates/dct-temporal-fingerprint-v3/CONTRACT.md`. As of its RESULT.md snapshot it has only feasibility/pilot evidence, **no scored paired source-transfer result**; requires a legitimate CUDA execution path and exactly-once frozen v1/v3 comparison. Do not claim it passes or launch ad-hoc GPU runs. If the goal is competitive innovation, prioritize completing that existing controlled representation test when executable instead of opening another ensemble/style-tuning lane.

**Status:** this view-ablation lane is complete, with a meaningful negative result, and authorizes **no submission**.
