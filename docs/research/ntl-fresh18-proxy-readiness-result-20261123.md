# NTL fresh-18 source acquisition — measured Phase 0 result (2026-11-23 MYT)

## Decision: **KAGGLE-ONLY BLOCKED; ALTERNATE GATEWAY CATALOG LEAD; NO MODEL-QUALITY RESULT**

This is an **executed data acquisition feasibility study**, not a scored classifier experiment. We have **no new OOF, independent transfer BA, or Kaggle Public BA**. Account's original **0.715 Public best remains frozen**.

## Authenticated execution, dated receipts

1. **[GitHub Action 45129347799](https://github.com/kolihhan/name-kaggle/actions/runs/45129347799) SUCCESS:** `kaggle benchmarks tasks models` + `kaggle benchmarks quota --format json` with the owner's GitHub secret. Matching exact names/approved aliases, **12/18 competition target labels listed** in catalog (39 parsed catalog entries overall). Missing six:
   - `claude-fable-5`
   - `gemini-3.7-flash`
   - `glm-5.3`
   - `glm-5.3-flash`
   - `kimi-k3`
   - `qwen3.8-max`
   
   At measurement: **daily $10.00/$10.00 remaining; monthly $100.00/$100.00 remaining**. These are **Kaggle Model Proxy credits**, not cash/free GPU, and may expire/change; do not promise they transfer to another gateway. No inference calls for this survey.

2. **[GitHub Action 45129637029](https://github.com/kolihhan/name-kaggle/actions/runs/45129637029) SUCCESS:** one bounded native-provider request for each missing model, with **max 256 output tokens per request**, daily and monthly quota preflight and an observed-spend cutoff of $0.75. On the attempted native paths, **all six HTTP 404**, zero verified exact-model responses. No generated response text logged or retained. Observed remaining quota still **$10 daily / $100 monthly**, measured debit **$0.00**. HTTP 404 proves these six specific native path/route combinations did not serve target replies, **not** nonexistence across every possible API route.
   
   Exact paths checked: Claude `/anthropic/v1/messages`; remaining five `/openapi/v1/chat/completions`. Requested provider prefixes `anthropic/`, `google/`, `z-ai/`, `moonshotai/`, `qwen/`, respectively. A catalog entry alone would still not certify a *returned* model version.

3. **Vercel AI Gateway live public catalog, 2026-11-23, read-only via connected Vercel tool**: named exact alternate model routes exist for all six:
   - `anthropic/claude-fable-5`
   - `google/gemini-3.7-flash`
   - `zai/glm-5.3`
   - `zai/glm-5.3-flash`
   - `moonshotai/kimi-k3`
   - `alibaba/qwen3.8-max`
   
   However **Vercel `list_teams` returned zero teams** to the current account connection; no authenticated gateway key, credit balance, hosted generation result, or actual provider-returned version is established. Model catalog is *discovery*, not measured availability or free quota. This would be metered if accessed; **zero Vercel paid/credit requests were made**. Prior source-use policy audit, including restrictions on training from certain Z.ai outputs, remains applicable.

## Why we did not train / score
- The exact 18 classes are required for full-class balanced accuracy; mixing unrelated 13/18 and 5/18 sources into an observational score creates channel==label confounding.
- Existing 64-prompt GLM transfer is GLM-only and **spent**. Historic 33/128 misses belong to old **0.683** classifier, not the current **0.715**. Old GLM-only likelihood gate actually regressed **88/128 to 77/128**.
- The prior Oleg 216-row full-18 diagnostic **89/216** belongs to an older frozen **0.691** model and a spent source. It is evidence of serious transfer instability but not the latest champion's accuracy.
- Source provenance, permitted use, provider-returned exact model IDs/versions, source-disjoint prompts, and at least two distinct acquisition channels are still unresolved.
- The **0.715** Kaggle Public is not directly comparable to Strong Sparse **0.3671 OOF** from historical grouped official TRAIN.

## Next decisive path — no hidden work promised
1. **If another route becomes authorized and actually reachable:** obtain exact returned IDs and legal use rights for the six missing classes, then one small, genuinely new *same-prompt* full-18 batch with generation metadata. Register and freeze source/prompt keys before seeing model-prediction results.
2. **If only one full-18 producer can be secured:** use it for single-source feasibility, not claim source-robustness; a second distinct producer is required to test H1 vs H2.
3. **Only after rights and data eligibility:** compare the **same actual 0.715 original scorer** against one fixed source-robust family-identity candidate on identical unseen rows. Primary score is full-18 BA; report class recalls and per-family true-positive recovery **and non-family false-positive capture**. Separate model-identity gains from class-prior and source/protocol effects. Require prompt-cluster paired uncertainty and source-stratified rescues vs regressions.
4. **If source acquisition stays blocked:** stop, do not propose a fifth weak proxy classifier, re-score spent OOF, spend GPU, tune GLM thresholds, or submit an unevaluated Kaggle candidate.

## Deliverables and spending
- [Frozen research hypotheses and intake contract](ntl-fresh18-source-readiness-20261123.md)
- [Workflow 1](../../.github/workflows/ntl-fresh18-proxy-readiness-20261123.yml)
- [Workflow 2](../../.github/workflows/ntl-six-native-route-probe-20261123.yml)
- [Bounded probe code](../../scripts/ntl_proxy_six_native_probe_20261123.py)
- Real GitHub Actions: **2 SUCCESS**; Kaggle Model Proxy actual queries **6**, **$0.00 observed**; Kaggle official submissions **0**; training **0**; new evaluation data **0**.
