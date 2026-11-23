# NTL 2026-11-23: Real class-family vs sibling bottleneck (no QLoRA)

**Scientific run**: [#45140011226 — success](https://github.com/kolihhan/name-kaggle/actions/runs/45140011226). Pin-preserving, **historical** 0.691 Lightning V2 **complete** predictions from `kolihhan/ntl-oleg18-frozen0691-probe-nov16`; 216 publisher-labelled Oleg response rows on predeclared 12-per-class / 18-class selection; exact source SHA checked. 216 distinct prompt IDs; raw text/ID/predictions deleted after processing, aggregates only. Script: `scripts/ntl_real_external216_oracle_bottleneck_20261123.py`. **This is not the current 0.715**. All 216 belong to narrow **sensitive_topics**, and original label-route/version and training exposure cannot be fully authenticated.

## Real strong-model baseline & diagnosis

| Scope | Result |
|---|---:|
| 18-way true exact identity | **89/216 = 0.412037 BA** |
| Family top-1 correct (regardless sibling) | **153/216 = 0.708333 BA** |
| Errors where predicted family correct, sibling wrong | **64** |
| Errors predicted across family | **63** |
| Ideal **sibling-only** correction (true label oracle, impossible deploy) | **153/216 = 0.708333 BA** |
| Ideal **cross-family-only** correction (true label oracle, impossible deploy) | **152/216 = 0.703704 BA** |
| Spent 216 conditional row-bootstrap BA 95% | **[0.370370, 0.458333]** |
| Spent 216 conditional row-bootstrap family top-1 95% | **[0.666667, 0.750000]** |

**Crucial falsifier:** Neither a perfect sibling-only correction nor a perfect family-only correction could reach **0.76 on these exact 216**. They are *counterfactual truth oracle ceilings*, not novel model results, not limits on a different unseen dataset.

### Where real errors occur (actual 0.691, historical Oleg216)

| True family | Family right | Exact model right | Main bottleneck |
|---|---:|---:|---|
| OpenAI | **60/60** | **33/60** | Sibling, 27 |
| Anthropic | 54/60 | 40/60 | Sibling 14, family 6 |
| Google/Gemini | 31/48 | 11/48 | Sibling 20, family 17 |
| Z.ai/GLM | **6/24** | **3/24** | Cross family 18, sibling 3 |
| Moonshot/Kimi K3 | **0/12** | **0/12** | Cross family 12 |
| Alibaba/Qwen3.8 Max | **2/12** | **2/12** | Cross family 10 |

Model output collapsed to `claude-opus-4-6` **47**, `gpt-5.6-luna` **36**, `claude-sonnet-5` **30** (out of 216), even though actual balanced labels were exactly 12 each. This is a *source/domain stress-test symptom*, not a justified reason to force equal label counts in hidden evaluation.

Prior training-group OOF for a **different Qwen3-8B LoRA model** had **701 sibling vs 522 cross-family errors**. These two diagnostics and model versions cannot be pooled as a same-model effect-size comparison. They jointly falsify the assumption that all future inference failures are necessarily sibling-only.

## What happened when attempting the only direct 0.715 vs highscore comparison?

[Actions #45139584293 — source-only outputs](https://github.com/kolihhan/name-kaggle/actions/runs/45139584293) **SUCCESS as an availability audit**:
- Own scored 0.715 clone `kolihhan/ntl-estopadilla-glm53-0716-repro-20261122`: `kaggle kernels output` returned no `submission.parquet`.
- Sanpier requested V3, Dennis, Areeb-0.719-title, Hcluo-0.719-title: `submission.parquet` exists but **only 3 examples and 1 predicted label** in commit preview — **not scored hidden 4k predictions**.
- Therefore **ZERO valid matched 0.715-vs-0.719 hidden per-row comparison**; no claim about which predictions are repaired. The earlier [source lineage diff](ntl-0715-highscore-decision-lineage-20261123.md) is valid static code evidence only.
- The first YAML-triggered action #45139499447 was malformed due to an unquoted colon in `name:` and never ran any job; fixed version #45139584293 executed successfully.

## Actual research direction — source-aware dual failure modes, NOT another weak-model patch

**Mechanism hypothesis (unvalidated, no GPU or training run):** Independent, source/protocol-matched model-identity *evidence* should first recover wrong model families (notably GLM/Kimi/Qwen in the narrow external stress test), then discriminate siblings *without overwriting well-supported families*. Current 0.715 already performs GLM likelihood, sibling residuals and soft grouping; reusing those modules with other thresholds is **not** innovation.

A realistic differentiated method would be a **cost-controlled counter-evidence router**: score genuine source-model evidence on rows where independently existing model signals *disagree at family level*; preserve a strict no-change action if evidence unavailable; maintain a separate sibling-only decision that cannot change family mass. Rank evidence acquisition by expected **paired-error repair per unit inference time**, not solely per-row uncertainty. This is *a research conjecture*, and overlaps existing public entropy/budget-aware GLM work; novelty requires a demonstrably new, robust evidence source rather than just different queue order.

### Feasibility caveats

- Kimi's official [Kimi K3 API selection docs](https://www.kimi.ai/help/kimi-api/api-model-selection) and Alibaba's [qwen3.8-max official model ID](https://help.aliyun.com/en/model-studio/qwen3-8-max) document actual providers, but **availability of a model name does not authenticate exact version, access credentials, model logprobs, budget, output licensing, or permission to use generated data for this research**. Kaggle proxy's earlier six exact-route lookups returned HTTP 404; a direct vendor API is a different, untested service and may incur charges.
- Independent labelled false positives AND true family misses (not just GLM-only positives) are required before any family resurrection can be considered safe.
- Need the real frozen 0.715 complete end-to-end logits/masks/predictions on same-row external target data to evaluate. Current public Kaggle commit preview does not satisfy this.
- No 0.715-vs-0.719 hidden truth, no new GPU, no QLoRA, no official submission, no score gain this turn.

**Research decision:** Do NOT build a class-bias patch from already-spent Oleg216. Keep own official 0.715 frozen; if exact model outputs and source-control evaluation can be acquired, test source-aware family evidence before any Champion changes.
