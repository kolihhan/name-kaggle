# CPR-v0 — genuine counterfactual patch experiment: STOP (2026-11-24)
Real completed authenticated [GitHub Actions #45362087702](https://github.com/kolihhan/name-kaggle/actions/runs/45362087702), SUCCESS. This includes actual labelled folds and C1/C2 inference; the result is **scientifically negative**. Code: `scripts/ntl_counterfactual_patch_retrieval_v0_20261124.py`. Frozen source and decision: `docs/research/ntl-cpr-patch-counterfactual-contract-20261124.md`.

## Provenance and costs
- Pinned official 2699 train, 150 prompt groups, Fold0 540 heldout (train 2159). Official full18 evaluated only on Fold0, **spent historical cohort**, not clean hidden validation.
- Publisher C1 360 rows/30 prompts/12 classes, C2 470 rows/50 prompts/12 classes. Publisher is **the same** in C1/C2, and served identity is self-reported without first-party proof. No training on C1 or C2; exact file hashes checked.
- 6498 train patches; 114287 TFIDF char features; mean counterfactual training patch weight 0.59248. Strong 4-view SVC control 266822 features. ~53.35 seconds model analysis wall time on CPU (additional fetch/setup excluded). No GPU, no code competition submission, no raw datasets committed.
- A unweighted patch maximum-similarity retrieval, B original within-prompt cross-generator overlap weighting, C within-prompt shuffled-weight placebo, D strong trained Sparse SVC. **All four predict among all 18 labels**, even when truth includes only 12 classes externally.

## Real same-row BA (not Public LB)

| Panel | A unweighted | B counterfactual | C shuffled placebo | D strong Sparse | B−A |
|---|---:|---:|---:|---:|---:|
| Official Fold0, 18 classes, N540 | 0.124074 | 0.114815 | 0.131481 | **0.383333** | **−0.009259** |
| C1 external, 12 classes, N360 | 0.102778 | 0.119444 | 0.097222 | **0.341667** | **+0.016667** |
| C2 external, 12 classes, N470 | 0.114281 | 0.103142 | 0.150204 | **0.270307** | **−0.011139** |

- On C1 B rescues 26 / regresses 20 vs A, but +0.01667 does not meet frozen +0.02 threshold.
- On C2 B rescues 17 / regresses 23; **B is below both A and placebo**.
- On official Fold0 B rescues 20 / regresses 25; **B is below both A and placebo**.
- C1+C2 pooled prompt-cluster bootstrap conditional 95% delta B−A `[-0.02533137,+0.01938004]` contains zero.
- Strong Sparse is better than B by 0.2685 BA on Fold0, 0.2222 on C1, 0.1672 on C2. The current patch-retrieval representation is not competitively viable.
- Original predeclared continuation gate `false`. Do not change parameters after inspecting the result; no score-driven salvage.

## Critical interpretation
- **FACT:** Prompt-specific novelty weighting does not transfer reliably across the 30/50 external prompts under this representation and strongly underperforms a cheap direct classifier.
- **INFERENCE:** The most salient lexical evidence may be tied to useful generator-specific word choices, so penalizing overlap can destroy signal. Or nearest-patch matching/char TF-IDF is too weak regardless of weighting. These alternatives cannot be isolated by this one experiment.
- **NOT PROVED:** That strong encoder patch late interaction as studied by HALvest is ineffective, or that semantic transfer/full18 provider-independent validation is established; this was only a cheap early falsifier.
- **Prior-art boundary:** The general patch late-interaction idea is related to [HALvest-Contrastive (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.2660/). The candidate's *within-prompt cross-generator weighting* was our tested intervention, not a new validated invention.

## Decision
**STOP CPR-v0 EXACT RECIPE.** No parameter sweep on C1/C2; no GPU scaling of this weak feature representation; no official submission; do not treat new script or passing CI as GOLD. The next real research lane must change representation and be measured against the strong same-row 18-label classifier, with a source-scoped independent test. Do not restart prompt-residual LDA, which already failed 11/13 external sources in a different archived experiment.
