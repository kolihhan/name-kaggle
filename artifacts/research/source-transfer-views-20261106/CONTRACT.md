# Source-transfer view ablation — pre-result contract

Date: 2026-11-06. Status: FROZEN BEFORE FIRST RUN. Single diagnostic, not a submission candidate.

## Why this, not another sibling/data admission audit

Observed own LoRA24 Gate A A1 > A0 (0.465713 vs 0.394243) on three source-held-out groups. Phase14 public-pool sparse baseline attained 0.417810 official grouped OOF. Public Lightning V47 reproduction scored 0.666, but changed multiple systems/data/heads; the gap is **not** attributable to a component. The recent sibling probe is RETIRED / NO FURTHER INVESTIGATION for this cycle. Earlier prompt-invariant-v2 small contrastive effect was not a convincing positive transfer proof.

**Question:** Are fixed word/content signals or style/surface signals more stable under a change of generation source? This is a mechanism diagnostic, not a claim of novel attribution architecture.

## Frozen one-shot design

Training: the exact Stage-1 A1 public-pool training partition reconstructed from the Phase14 43,781-row conservative subset: 30,133 external-only rows after removing three Gate A heldout sources (abdullah, zakir, semenoff) and their prompt-ID/prompt-key groups. Frozen raw SHA256: 37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec. All 18 labels; no official train or test files.

Two fixed source-shift diagnostic evaluations (previously inspected, **NOT newly independent**):
- alessandrolair/ntl-kaggle-benchmarks-generated-passages, ntl_kbench_generated.parquet, pinned SHA256 e98b1924269c30116f50623052f7fc4ea6148f7575d2fe483c9bd899b4ea2e21, 3,866 rows, 12 labels, author-created prompts.
- elliotthankinson/ntl-exact-gens-v3, pool_v8j_add.parquet, pinned SHA256 e8aeafcd75b9832c297324da360051a48fec3b58d1d8cfe251e03c2558f40185, 8,347 rows, 7 labels, source/protocol different.

Fit original Phase14's exact four feature views **once** on the same training texts, never on evaluation text:
char TF-IDF 3-5 (250k, wt 1), word TF-IDF 1-3 (180k, wt .7), shape-char TF-IDF 2-6 (60k, wt .5), small/function-word TF-IDF 1-3 (50k, wt .5), frozen min_df=2. Identical LinearSVC(C=.5,class_weight=balanced,random_state=42,max_iter=5000). Compare fixed arms by dropping feature views before fitting:
- FULL = char+word+shape+small
- LEXICAL = char+word
- SURFACE = shape+small

This is **view deletion**, not capacity-matched causal evidence. Shared vectorizer ensures source differences are not a feature-fitting artifact. Predict once on each source, with no selection/tuning. Measure source-specific eligible-label balanced accuracy (each label >=20 rows), per-family macro recall and confusion, and FULL-minus-SURFACE + FULL-minus-LEXICAL deltas. Report distinct 12-label vs 7-label metrics, never official competition 18-way BA. No result-based retuning, source filtering, reseeding, alternative family rules or new Kaggle submission.

Interpretation:
- SURFACE beats LEXICAL on both source shifts: evidence that lexical content is especially source-sensitive in these diagnostic sources, suggests next objective should encourage prompt/style invariance, but not sufficient to promote or claim hidden transfer.
- LEXICAL beats SURFACE consistently: falsifies the claim that style-only is sufficient in this setting. Consider content-stable contextual representation instead.
- Mixed: source/protocol interactions dominate; stop using one global claim.
- FULL alone wins: complementary views matter; test prompt-specific confounding before architectural promotion.

**Predeclared limitation:** source banks were already examined in reconnaissance, so they are **exploratory diagnosis only**; some prompt-bank independence/returned-model identity remains unresolved. Cross-source data origin and feature dimensionality confound causal inference. No Public LB is consulted.

Budget: GitHub Actions CPU only, up to 25 minutes / no GPU. One experiment run. Store only aggregate JSON; never commit or upload texts, prompts, row IDs or model outputs. If pinned files or exact training split fail, block honestly.

Post-result decision: regardless of numerical winner, retire this source pair for research selection, no extra sweeps. Choose a new substantive objective based on observed failure, or report that source transfer does not settle it.
