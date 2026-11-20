# 2026-11-20 12:25 MYT — live Kaggle top 10 vs champion and diagnostic submission choices

## Authenticated observation — no submissions

[Read-only successful Kaggle leaderboard + own submissions Actions #44674191096](https://github.com/kolihhan/name-kaggle/actions/runs/44674191096). GitHub Actions ran Kaggle CLI with the owner's authorized token, read `kaggle competitions leaderboard name-that-llm --show --csv` and `kaggle competitions submissions name-that-llm -v`. It returned the **TOP 20 ONLY** via page token, so **cannot identify owner exact overall rank**.

- Top 10 Public BA: `[.776,.771,.770,.769,.769,.769,.769,.767,.765,.761]`.
- **Top 10 cutoff = 0.761**, top 1 = 0.776.
- Owner verified frozen best **#58514818 COMPLETE, Public 0.702** (submitted 2026-11-18).
- Absolute gap to current Top 10 **0.059**, not "near Top 10".
- Prior Phase7 **#57838845 COMPLETE, 0.369**, Phase15 **#57861337 COMPLETE, 0.357**.
- Current Kaggle submission listing: **23 owner submissions**, **0 pending**, no new official submission from this research call.
- Public ~40% test and Private ~60% can reorder places; one cannot predict exact private result, but do not assert ~0.702 is sufficient for Top10.

## Evaluate suggested "submit official-only and Phase14" causal diagnostics

**Potential value, but low decision value against the 0.702 champion.**
- Official-only four-view LinearSVC **0.374857** genuine 5-fold official-grouped OOF was actually trained/tested via successful CPU job [#44667515988](https://github.com/kolihhan/name-kaggle/actions/runs/44667515988/job/136060326847). Historical official+26,025-external Phase15 same-recipe **0.405322 OOF, Public 0.357**.
- Official-only Public is **UNKNOWN**. Phase7's 0.369 cannot be borrowed as its Public prediction: different architecture, training sources/gate and score variability. A score near 0.37 is only a hypothesis, not a reliable calibration interval.
- If an official-only Public diagnostic exists, its difference vs Phase15 still contains two separate scoring executions (and possibly stochastic inference/scorer variance), so it does not isolate a perfectly noiseless paired causal effect. The experiment can inform direction but may not distinguish Δ~0.01 from noise.
- Phase14 **0.417810 OOF** vs Phase15 **0.405322 OOF** is the clearest fixed-architecture *DATA inclusion/exclusion* retrospective mechanism. But Phase14's 43,781 external includes **17,756 Gemini/GLM/Kimi public outputs rejected by the repo's pre-registered provider-output-terms policy** ([terms audit](../../phase15/TERMS_AUDIT.md)). The competition accepts public/free externally accessible data, but this is not a warranty of original provider output rights; **do not upload Phase14 to Kaggle or train a new Phase14 copy solely for competition use without resolving provenance**.
- The two weak SVMs are ~0.36–0.42 OOF vs incumbent 0.702 Public and target Top10 0.761: their submission differences **cannot prove complementarity** to the currently deployed 0.702 ensemble. They are for source-distribution science only, not a direct competition-improvement test.
- Decision now: **no official-only or Phase14 competition submission** based solely on a quoted proposal. Preserve user scarce official runs; evaluate official-only once only if explicitly prioritizing scientific diagnostic over medal pursuit and after private smoke tests. No re-admission of terms-blocked external examples.

## Existing route and label risks

Current Kaggle Open Pool source SHA `7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492`, 231,036 rows/66 source labels and 18 classes, is an actual downloaded artifact, not third-party search snippet. Weak7 91,754 rows, only 17,142 pass **metadata-only** nonempty publisher-license + served-model + publisher heuristic flags. These are **not route-authenticated nor legally admitted**, no new training. Goldentom42's statement of 10,838 mislabeled route-unverified Gemini3.7 responses belongs to an **older different snapshot**; must not blindly transfer the count/identity to the now-current 3,541 Gemini3.7 rows. Need compare `gen_served_model`, `gen_route`, source-level evidence if deciding to train. Binary flag `flag_gemini37_routing_risk` is a publisher heuristic, not independent proof.

## Real top-10 focus

- Need **~0.059** Public BA improvement over existing frozen 0.702 to reach **current** top10 cutoff, but public top10 cutoff is moving and Private standings can differ.
- Avoid optimizing Phase7/14/15 to 0.40. Select a candidate whose novelty could materially affect **currently embedded 0.702 model-specific likelihood, latent classifier, calibrated routing**. Strong-component errors require genuine source-independent annotations / fold-clean models to claim conditional complementarity; no fabricated champion OOF from prefit train.csv.
- A **distinct, pre-registered high-effect mechanism** + controlled clean evaluation matters more than two additional noncompetitor Public draws.

## Limits

Only first page of LB returned, so owner exact ranking not known; all Public scores shown to 3 decimal places. No claim that a ~0.059 Public change is realistically achievable from current architecture; also no claim that Private cannot reorder. The original 0.048 Phase15 OOF/Public discrepancy remains causally unresolved.
