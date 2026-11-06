# V47 core grouping feasibility

**2026-11-07 MYT. Closed at the CPU feasibility check.**

## Hypothesis and evidence

V47 groups lexically similar passages before applying centered and Gaussian core votes. Mixing responses to different questions could distort those votes. The exact V47 submission scored **0.666** on Kaggle Public. That score reflects the whole pipeline. [Prior source audit](ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md)

## Experiment

Ran the exact V47 `topic_groups` and `capped_agglo` functions once on the **2,699** official training passages in file order. The grouper received only text. Scoring then read the **150** ground-truth prompt IDs. Original parameters were an 18-member cap and 0.15 similarity cutoff. The predeclared gate required **at least 5% of all passages** to be in a group containing more than one prompt before preparing a core-grouping ablation. The threshold is an exposure heuristic.

| Metric | Count | Rate |
|---|---:|---:|
| Passages in non-singleton groups | 2,580 / 2,699 | 95.59% |
| Predicted pairs that share a prompt | 17,610 / 17,654 | 99.75% precision |
| True same-prompt pairs recovered | 17,610 / 22,933 | 76.79% recall |
| Passages in mixed-prompt groups | 23 / 2,699 | 0.85% |

There were 375 inferred groups, including 119 singletons. Three groups mixed prompts, touching two true prompts and yielding 44 cross-prompt pairs. Grouping took 44.586 seconds on CPU. Pair counts were checked independently against scikit-learn's pair confusion matrix.

## Decision

Close the proposed grouping-off ablation at the 5% gate. This check shows rare prompt mixing on the official training batch, with some same-prompt pairs missed. The hidden prompt structure and the grouped votes' effect on exact-model accuracy remain unmeasured. The ablation and Kaggle submission were conditional on the feasibility gate.

## Provenance

- Kaggle version ID `365000313` downloaded as an IPYNB with SHA-256 `0930f6105b42c29d6f4e8a42a5536bd63e39e64b54a78a17c93ed391342dbd6f`. [Exact notebook version](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313)
- The joined source of its 37 code cells matched the prior authenticated V47 audit at SHA-256 `c50abf03f5d5472dc2404f62d000df659001e2b73def475c17b45d581c4199dc`. The web download and CLI pull serialized the notebook differently. [Authenticated source audit](https://github.com/kolihhan/name-kaggle/actions/runs/42016381979/job/127207839023)
- Input SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`. The local pre-run contract SHA-256 was `6704b03b4212cbcfd3a07fc51d2c40148984ac0dac398a521037f02c0e5e65a6`. Runtime used NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0. [Input checksums](../../data/README.md)
