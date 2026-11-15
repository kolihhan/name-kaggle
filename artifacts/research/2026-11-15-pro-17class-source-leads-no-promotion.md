# PRO continued: investigate near-full external cohorts without false independent-validation claims

Date: **2026-11-15**. This is **metadata evidence**, no passages or model inferences. It is a decision on **eligibility**, not a score.

## Authenticated new finding

[GitHub Actions #43690975535](https://github.com/kolihhan/name-kaggle/actions/runs/43690975535) **SUCCESS**. We restricted the canonical `crtproduction/name-that-llm-open-pool` 65-source public `sources.csv` and `source_protocol.csv` catalog to exactly its three sources reporting **17** nominal labels. Cross-checked source references against the pinned 0.683 champion's direct Kaggle dataset dependencies.

| Upstream source | Rows (catalog) | Nominal models | License as catalogued | Direct champion mount | Served model ID audit |
|---|---:|---:|---|---|---|
| `drjcfmd/ntl-generated-v2` | 20,400 | 17 | CC BY-SA 4.0 | No | **unknown** |
| `zeminglii/ntl-own-exact-route-generations` | 2,000 | 17 | CC BY 4.0 | No | **unknown** |
| `earthmaker/ntl-own-prompt-generations` | 698 | 17 | CC BY 4.0 | No | **unknown** |

All three catalog records have route/sampling/post-processing field content, **but none establishes actual served model identity, original prompt independence, or disjointness from the champion's upstream prefit sources**. Source file counts are publisher metadata, not independently checked per-row truth. The prior 65-source [inventory #43619232564](https://github.com/kolihhan/name-kaggle/actions/runs/43619232564) found **zero single-source 18-label cohorts**. Direct-mount absence does NOT prove prefit absence.

## Research decision

**NO QUALIFIED full-18 independent transfer panel from these three leads.** Do not score them as 18-class BA or use them to optimize sibling rules; don't blend missing labels across mismatched providers and call it controlled transfer. The 17-label sources remain possible *exploratory negative controls* only if served IDs and prompt provenance can first be independently verified.

Priority when a source with complete, legitimate provenance is available: freeze a source- and prompt-disjoint same-row panel, run the exact 0.683 champion and the frozen Lightning candidate on it, report per-label support, paired rescues/regressions and runtime. Until then, retain Lightning300's 0.960279 **historical training-set** BA strictly as an in-sample result. Existing Lightning1200 scale inference is only a runtime study, never a score estimate.

No new training, model GPU, official submission or claims of 0.7 resulted from this metadata check.
