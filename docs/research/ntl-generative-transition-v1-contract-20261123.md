# Independent NTL Generative Token-Transition Fingerprint v1 — Frozen Experiment Contract
Date: 2026-11-23. Research branch: `research/ntl-generative-transition-fingerprint-v1-20261123`.

## Goal and scope
Ask whether model-specific **causal token-choice dependencies** supply transferable 18-class identity information beyond (a) token marginals and (b) a capacity-matched context-destroying negative control. This is **not** an attempt to modify, use, improve or ensemble the 0.715 external Champion. Unlike the already-failed MPCD linear-loss study, this compares fully specified class-conditional **generative probabilities**. Unlike the previously frozen unexecuted `conditional-denoising18-v1` (RoBERTa MLM residual adapters), this is a lightweight, **not pretrained** explicit Markov surrogate used to distinguish source-trace evidence from surface frequencies before spending GPUs. Neither method claims to approximate true proprietary generator probabilities.

**Important cancellation identity:** for fixed x, `argmax_m [log P_m(x) - log P_shared(x)] == argmax_m log P_m(x)`. Subtracting the same general LM score from all 18 candidates changes no prediction. This project specifically tests class-**dependent conditional probabilities** `P_m(t | t_previous)`, not a cosmetic common-residual layer.

## Inputs and fixed evaluation
Immutable original official training `data/train.csv` SHA `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`, `artifacts/phase1/folds.csv` SHA `94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d`. Only historical 2,699 train rows / 150 Prompt groups / all 18 class labels. 5 frozen prompt-disjoint folds; no Kaggle test, prefit ensemble, model weights, hidden labels, publisher metadata or heldout Prompt ID at inference. Every corpus count/vocabulary/class training statistic is fold-local. No official submission or new GPU run.

## Frozen mechanism: word-choice *abstract tokens*, not ordinary word memorization
Extract ordered tokens with regex `[A-Za-z]+(?:['’][A-Za-z]+)?|[0-9]+|[^\\w\\s]`; retain exactly the existing `scripts.run_style_baseline.FUNCTION_WORDS` lexicon as lowercase, preserve punctuation from fixed explicit ASCII/Unicode set, numbers as `<NUM>`, and replace all content words by `<W{length-bin}:{case}>` with fixed bins `1..8,9-12,13+` and `LOWER|TITLE|UPPER`. Map unknown symbols to `<SYM>`. Each response is transformed separately, no test-batch grouping. This anonymizes content vocabulary while retaining token choice, grammatical transitions and word shapes; it is a proxy style Markov model, not full token likelihood.
Use fixed `<BOS>` predecessor. Build shared training unigram/bigram counts. For each candidate class m build class-specific unigram/bigram counts. Fit from train only.

Global unigram distribution `p0(t)=(count_global(t)+0.5)/(N+0.5*V)`. Global conditional distribution `p0(t|h)=(count_global(h,t)+12*p0(t))/(count_global(h)+12)`.
Class unigram `p_m(t)=(count_m(t)+20*p0(t))/(count_m+20)`.
Class conditional `p_m(t|h)=(count_m(h,t)+16*p0(t|h))/(count_m(h)+16)`.
All candidates use equal class prior. Decision = highest **average per-token log probability**. No test source, trained classifier, pseudo-label or class-specific thresholds. Vocabulary is training-only; unseen tokens map to explicit `<UNK>`.

## Predeclared arms, one frozen run
- **A — Unigram generative**: `mean_t log p_m(t)`; marginal word/shape frequency only, no context.
- **B — Genuine causal conditional generative**: `mean_t log p_m(t | t_previous)`; same source data, counts and smoothing formulas.
- **C — Shuffled-context negative control**: **source-specific** bigram counts fitted from a deterministic, seeded *within-passage token permutation* preserving per-class unigram counts but destroying observed sequential transitions; global prior from real train remains unchanged; score on genuinely ordered heldout text. Identical class count/parameter structure and smoothing as B.
- **D — strong sparse direct classifier**: old `scripts.run_strong_sparse_anchor.build_fold_features/make_svc` fixed independent source-free supervised 4-view and style classifier, on the **exact same rows and folds**. Not an ensemble or mix.

## Main hypotheses
H1: source-specific ordering adds identity signal: B substantially beats both A and C in all-18 Prompt-grouped BA. H2: class-specific token marginals and dataset source artifacts already explain most of the effect: B ≈ C or B ≈ A. The D arm makes clear whether even a positive H1 is competition-relevant rather than a weak experiment.

## Precommitted analysis and decision
Report: fold BA, full 18 BA, each per-label recall, per-provider family macro, B/A/C/D paired rescues-regressions, 1,000 fixed seed Prompt-cluster bootstrap CIs for B−A, B−C and B−D, rows and preprocessing time. No threshold sweep.
- **Mechanism PASS** (only justifies independent source-transfer research): B−A >= +0.01, B−C >= +0.01, BOTH prompt-bootstrap CI lower bounds >0, ≥3/5 fold wins vs C, no class recall fall >0.15 vs A.
- **Competitive relevance signal:** B >= D on same OOF or independent demonstrated complementary rescues across validated sources. Else explicitly report B as weak even if H1 passes.
- If H1 fails, stop **this exact Markov abstraction and frozen smoothing**. Do not alter smoothing, token bins, mask sizes, seed, folds or candidate labels based on observed scores. Prior failures of prompt-contrastive and style-LDA don't eliminate all conditional generative models, and this result doesn't certify transfer to hidden Kaggle.
- **Independent transfer gate remains BLOCKED** until a truly distinct, full-18, source-disjoint, protocol-verified benchmark exists; historical official OOF must not be sold as fresh hidden BA.

## Operational bounds
CPU only, no pretrained model/adapter, no Kaggle hidden rerun, no official submission, 5-fold one run, fixed 2 CPU threads for numeric control, <=20 minutes GitHub-hosted Actions. Don't check official training text into new files or upload labels/predictions to public workflow output. Print aggregate metrics only. Preserve original Champion and science records.
