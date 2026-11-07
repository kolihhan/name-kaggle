# Conditional source-balanced risk — frozen CPU mechanism test

Date: 2026-11-08 MYT.
Status: FROZEN BEFORE ANY CANDIDATE PREDICTIONS.
Scope: one external-only CPU experiment. This is a diagnostic, not a submission candidate.

## Decision and prior evidence

Current competitive Best: Ozertuu reproduction Public BA 0.668 (58022340 and
58045266); V47 reproduction 0.666 (58008463). Our LoRA24 is 0.520.
Official history already directly checked in Actions 42385772029 additionally
shows V47 length40 58078151 COMPLETE 0.663 and Short-TRACE 58093864 COMPLETE 0.093.
The old length40 NOW entry saying pending is stale.

STABLEPAIR-18's matched-protocol gate failed and remains unchanged. Its training
is not reopened. Short-TRACE T-only standalone remains rejected; 14B upgrade,
post-result weight searches and another copied submission are not selected.

Global competing explanations: missing representation/capacity versus learning
the wrong mixture of collection sources within each model label. Strong public
systems already contain broad neural/lexical/likelihood views. Their data scale,
objective and inference differ simultaneously, so 0.668 alone cannot decide.

Alternatives considered:
1. More matched-protocol sibling data: scientifically relevant but checked data
   failed; do not repeat inventories or lower its gate.
2. Larger evaluator / new dense representation: no isolated evidence justifies
   a GPU pilot; existing rank summary result does not identify a capacity cause.
3. Conditional source-balanced empirical risk: CPU-executable, unchanged rich
   features and model class, with a source-shuffled control that isolates whether
   actual collection-source assignments help. SELECTED as a mechanism test.

Hypothesis: within-label source imbalance encourages the sparse classifier to
learn collection-specific cues. Equal training loss mass per observed source
within each label improves held-out-source attribution beyond a random
redistribution of the same per-label training weights.

This is established reweighting methodology, NOT a claim that source balancing
or a new attribution architecture was invented here. Group-shift research
motivates testing the objective, not an assumption it succeeds:
https://arxiv.org/abs/1911.08731 .
ATLAS supports evaluating modeling objectives under shifts but is already
represented in V47 and does not prove this weighting treatment:
https://arxiv.org/abs/2610.19583 .
Cross-resolution attribution also cautions against discarding lexical evidence:
https://arxiv.org/abs/2610.28434 .

## Frozen data, isolation and limitations

Use only public nishantdahal/ntl-handwriting-blend / pool_v4.parquet,
SHA256 37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec.
Reuse build_conservative_subset (43,781 external rows), then original
build_stage1_split: 30,133 preliminary train rows; held-out sources exactly
abdullah, semenoff, zakir. Remove train rows with NFKC/lowercase/whitespace-normalized
answers equal to any held-out answer, before fitting every feature or classifier.
Verify zero overlap of prompt_id, prompt_key and normalized answer after purging.
No official train/test, new API generation, paid service or GPU is read/run.

These sources and source-level outcomes were previously consumed in other
experiments. They are a disjoint TRAIN/TEST diagnostic, not fresh independent
model-selection or causal evidence. Source is an imperfect proxy for protocol;
different sets of available sources per label and exact endpoint provenance
remain limitations. Weighting cannot fill structural model/source zeros.
No causal protocol invariance or Kaggle score is inferred from this experiment.

Fit the exact Phase14 four TF-IDF views once on training rows only:
char 3-5 / 250000 / 1.0; word 1-3 / 180000 / 0.7;
shape-char 2-6 / 60000 / 0.5; small-word 1-3 / 50000 / 0.5.
Keep min_df=2, lowercase=False, sublinear_tf=True, float32, existing analyzers.
Classifier for every arm: LinearSVC(C=0.5, class_weight="balanced",
random_state=42, max_iter=5000), default squared-hinge and sklearn 1.6.1.
No feature cap, seed, source, C, clipping or threshold search.

## Arms and exact objective

B: existing class-balanced objective with sample_weight=1.

S: same rows/features/model; for row i of class y and source s,
w_i = n_y / (k_y * n_y_s), where k_y is the number of observed sources for y.
Thus sample weights sum to n_y within every class and every observed
class/source cell has equal mass. Combined with class_weight="balanced",
each class keeps equal total risk mass. Source is never an inference input.

P (negative control): permute S's weights among rows WITHIN EACH CLASS using
numpy.default_rng(20261108) and sorted class order. This preserves the exact
weight multiset, class mass, optimizer and regularization budget but destroys
their association with the true collection source. One permutation only.

Save same-row 18-class margins/predictions for all arms in scratch outside Git.
Publish aggregates only, no passages, prompt IDs, raw outputs or credentials.

## Evaluation and predeclared decision

For each of the three held-out sources, evaluate labels with >=20 held-out rows
and present in training. Require >=8 eligible labels and >=300 eligible rows
per included source; all three sources must qualify, otherwise INCONCLUSIVE
without relaxing the rule or selecting a favorable subset.

Report eligible-label balanced accuracy, exact label recalls, macro sibling and
provider errors, same-row rescues/regressions versus B, effective sample size
and training source/class support. Comparisons use identical row/label support
within each source. Never compare these source-dependent label sets to 18-way
Kaggle Public 0.668 or LoRA24 0.520.

Primary statistic: unweighted mean over the three source BAs.
Continue to a separately contracted next stage ONLY if all conditions hold:
- mean S minus B >=0.010;
- mean S minus P >=0.005;
- no source S minus B <-0.010;
- 95% percentile interval for mean S-minus-B excludes zero, with 2000
  prompt-cluster bootstrap replicates and fixed seed 20261108.
Cluster by prompt_key (fallback prompt_id), shared across sources, before
recomputing the per-source class recalls. Use identical paired draws for arms;
discard bootstrap draws missing a frozen eligible source/class cell.
Report retained replicates and declare interval inconclusive if <1900 remain.
This interval describes this reused diagnostic, not independent confirmation.

A convergence warning or a data/feature isolation failure means INVALID
engineering evidence; independently repair only that defect with predictions
discarded, and preserve intended scientific settings. Do not adapt the method.

If the gate fails: close this exact objective/representation treatment without
retuning, no official OOF and no competition submission. The result rejects this
operational reweighting hypothesis, not all domain-generalization objectives.

Even if it passes: no automatic Kaggle promotion. EVALUATION_POLICY.md still
requires frozen leakage-safe official grouped OOF, independent appropriate
transfer and same-row complementarity against a strong anchor. V47/Ozertuu
fold-safe fresh predictions/training provenance are not currently available;
own LoRA24 OOF cannot stand in for the 0.668 system.

## Restriction review

RESTRICTION: no standalone GPU pilot or new model checkpoint in this experiment.
WHY: objective can be isolated with existing CPU features.
EVIDENCE: raw public pool and exact feature code are available.
PIVOT: compelling CPU objective evidence plus a separately frozen
feature/checkpoint test that resolves a specific uncertainty.
STATUS: ACTIVE for this experiment only.

## Implementation plan and checks

1. Add scripts/run_conditional_source_balance.py reusing existing split/features.
   Test equal class/source risk mass, within-class shuffle preservation,
   cross-source prompt/text purge, BA decomposition and prompt-cluster pairing.
2. Run the tests, commit executable code, and execute the three frozen CPU arms
   once. Preserve aligned predictions outside Git and aggregate JSON in this
   artifact directory. No repository-tracked data file is changed.
3. Review actual results against every gate, update RESULT.md and the existing
   NOW.md. Retain all earlier contracts/negative evidence. Scientific continuation
   is decided from observed results, not from an aggregate leaderboard score.

Budget: one CPU run, at most 35 minutes; max 2 BLAS/OpenMP threads.
If computation cannot finish or valid data support fails, report the observed
blocker. Do not replace execution with synthetic attribution metrics.
