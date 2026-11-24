# NTL GPT sibling 14B ICL — launch receipt and preregistered interpretation caveat

2026-11-24 23:54 MYT. This is a status/implementation-integrity report, **not** a claim of measured BA improvement.

## Real execution evidence
- [Accepted one-shot GitHub Actions #45430337038](https://github.com/kolihhan/name-kaggle/actions/runs/45430337038): smoke tests passed, authentic 0.718 scored donor pinned SHA `74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56`, pretrained `qwen-lm/qwen-3/Transformers/14b/1` asset mounted (metadata ID), one fixed C1 external corpus and offline wheel dataset, remaining Kaggle GPU quota 21.03 hours **before** job launch. Private kernel `kolihhan/ntl-qwen14b-gpt-triad-icl-private-20261124` accepted at 23:51 MYT; no competition submission and no weight training.
- [Read-only checks #45430617075](https://github.com/kolihhan/name-kaggle/actions/runs/45430617075) and [#45430787454](https://github.com/kolihhan/name-kaggle/actions/runs/45430787454): `KernelWorkerStatus.RUNNING` 23:53–23:54 MYT, no authenticated aggregate yet, no caught runtime error classes. Kaggle logs have not yet emitted a retrievable model-loaded or progress marker; do **not** call this success/failure of model loading. Existing `kolihhan/ntl-fable-preservation-252-20261124` remains `QUEUED`.
- Running frozen comparisons are **ZERO**, **PAIR** (three same-prompt triplets, truthful names), **PLACEBO** (labels re-assigned by fixed index expression) on 72 official 3-model heldout outputs and 90 other-publisher Beroli C1 3-model outputs. This is **3-way classification on preselected true-family support**, NOT full18 BA nor same-row uplift vs 0.718. Source identity is publisher-claimed.

## Important independently identified fixed-code caveat — MUST disclose alongside results
Inspection of the **already-submitted running source**, `scripts/ntl_gpt14b_icl_inference_20261124.py`, found:
```python
shown=source_label if arm=="PAIR" else LABELS[(k+idx+1)%3]
```
For three demonstration triplets (`idx=0,1,2`), this yields rotations **+1, +2, +0**. Thus the third triplet **retains all three correct source labels**. The original [frozen contract](ntl-gpt14b-icl-triad-precommit-20261124.md) describes a wrong-label placebo too strongly. The actually running **PLACEBO is a mixed-label / two-of-three corrupted demonstration control**, not a fully corrupted or independent random-label null. Interpret `PAIR > PLACEBO` as advantage over **partially corrupted label evidence** only. A non-significant `PAIR–PLACEBO` delta does *not* falsify all source-aware in-context exemplar learning, especially if the uncontaminated triplet remains informative. Do not silently patch the running kernel or revise the precommit after results; any stricter control requires a distinct version and separately frozen prospectively independent test.

## Gates before any next action
1. Real private worker must be `COMPLETE`; download only `ntl_qwen14b_gpt_icl_aggregate.json` aggregate and verify exact 72+90 row counts, three arms and group bootstrap.
2. Report actual 3-way BA per source, PAIR–ZERO and PAIR–mixed-PLACEBO rescues/regressions, per-sibling recall, bootstrap CI and time per sample. If code failed, identify engineering vs model-method issue.
3. Full 4000-row runtime extrapolation must account for **all required arms/gate calls** and original 0.718 inference overhead. Any >9h pathway is not Kaggle-viable without a fresh speed mechanism. No leaderboard trial or claim of full18 +0.05.
4. Preserve original `0.718` owner-scored public Notebook reproduction; independent novel GOLD remains unproven. No small-model training.

**Interpretation:** The experiment provides evidence about an inference-only large-model *exact sibling discrimination* mechanism, even if its 3-way BA is low; initial low result requires per-class errors, source drift, token scoring, prompt length and availability diagnosis, not an automatic family-wide PASS/STOP.
