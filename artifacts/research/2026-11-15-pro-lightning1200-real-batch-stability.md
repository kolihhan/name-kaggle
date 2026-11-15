# Lightning 0.694: real 1200 historical-train score and batch-size robustness

Date: 2026-11-15. **All predictions were obtained from already completed genuine private Kaggle inference**, no rerun or new GPU.

## Frozen authenticated outputs

- [Existing 1200-row terminal checker #43691415910](https://github.com/kolihhan/name-kaggle/actions/runs/43691415910) confirmed 1200/1200 legal predictions spanning 18 predicted classes; GSQ execution **`applied`**, **240** selected rows, 14 stage logs, no detected supervised-fit-guard errors, exact parquet SHA256 `926dffc6ab0909f63767e8fcec7ca9488f7d1dc7c49656b0eaa8cf50064e0826`. The previous elapsed-time regex was fixed, but no reliable elapsed marker was found in emitted logs; do not fabricate runtime.
- [Actual two-size same-row accuracy/stability experiment #43691599507](https://github.com/kolihhan/name-kaggle/actions/runs/43691599507) SUCCESS, reusing the existing 300 and 1200 parquet outputs, both SHA-pinned. Truth is official competition **train.csv first 1200 rows**, known SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`; no independent validation.

| Condition | Right / total | Accuracy | 18-truth-class BA |
|---|---:|---:|---:|
| Old real 300-row run | 288/300 | 0.960000 | 0.960280 |
| Same first 300 predictions in 1200-row run | 290/300 | 0.966667 | 0.964637 |
| Remaining 900 rows of 1200 run | 859/900 | 0.954444 | 0.955436 |
| Full real 1200-row run | **1149/1200** | **0.957500** | **0.958200** |

- **Identical first 300 texts, two different run contexts:** 6 predictions changed, **4 old-wrong → new-right** and **2 old-right → new-wrong**. The other 294 predictions match. Such instability is a valid observed difference, **but NOT proof of a causal batch-size effect**: sampling/randomness, runtime and dynamic GSQ selection could also explain it. Do not tune heuristics on these 6 spent cases.
- In the 1200-run sample, Kimi-K3 was the weakest true class with recall **0.75758**, followed by GLM-5.3 **0.9** and Gemini 3.7 Flash **0.91781**. This is merely a historical in-sample error slice. It must not become a new Kimi-specific rule.

## Decision and evidence boundary

**KEEP as frozen feasible inference-only candidate; quality/generalization UNVERIFIED.** The high 95.82% BA is on *historical official training data*, and this source uses previously fitted third-party assets. **Checkpoint contamination or fit exposure cannot be excluded**. Our 0.683 official best is on different hidden rows, so comparing 0.958200 to 0.683 is invalid.

No qualified, unused 18-label independently source-and-prompt-disjoint benchmark is available; see [three nominal 17-label source leads](2026-11-15-pro-17class-source-leads-no-promotion.md). The Lightning source's author title '0.694' is **not our authenticated Kaggle score**.

Next falsifier: admission-qualified same-row 18-class external transfer with champion control, or a clearly scoped single research-bet official submission **only if competition quota, inference time, provenance and no-training guards permit**, recorded as a one-shot outcome (not an unbiased mechanism proof). No new Kaggle official submission launched in this audit.
