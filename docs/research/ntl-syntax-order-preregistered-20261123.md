# NTL syntax-order authorship test — frozen before execution (2026-11-23)

**Status:** [GitHub Actions run #45146166103](https://github.com/kolihhan/name-kaggle/actions/runs/45146166103) was **QUEUED** at latest inspection; **NO computed accuracy result exists yet**. It is misleading to count dispatch/queued as experiment complete. Code [`scripts/ntl_syntax_order_falsifier_20261123.py`](../../scripts/ntl_syntax_order_falsifier_20261123.py) and [workflow](../../.github/workflows/ntl-syntax-order-falsifier-20261123.yml) committed and source-frozen before any model result.

## Technical research question

Are **coarse, lexical-free grammatical ordering patterns** a distinct generator identity cue beyond coarse part-of-speech (POS) *frequencies*, and does any such difference survive a source change? This is not a claim of novelty or score improvement. In particular, [READER](https://arxiv.org/abs/2606.10794) already uses response-level temporal DCT embeddings; our own repo already includes temporal DCT v3. [Multi-Resolution Fingerprinting](https://arxiv.org/abs/2610.28434) already distinguishes syntactic/stylometric/semantic viewpoints. This experiment is strictly a bounded **negative-control test**, not a new competition model.

## Fixed, label-blind feature extraction

- Download **free NLTK averaged perceptron English POS tagger**, `nltk==3.9.2`; NO commercial model, GPU, QLoRA, online API at inference, target model weights or source-generated prompts.
- Strip word identities; retain coarse Penn-Treebank-derived tag categories only.
- C0 = POS unigram TF-IDF; C1 = the **same** POS unigram + 2-/3-gram grammatical sequences; C2 = same sequence model **after deterministic within-sentence tag-order permutation** that preserves unigram counts but destroys real local order.
- Fixed `LinearSVC(C=1, class_weight=balanced)`, same vectorizer settings except predeclared sequence ngram range; no hyperparameter search, no ensemble patch.

## Actual evaluation contracts

- **2699** official TRAIN examples, **150 prompt-disjoint** groups, frozen 5-fold assignment. Compare C1–C0 AND C1–C2 using same-row 18-class balanced accuracy, fixed grouped bootstrap.
- External: SHA-pinned Oleg 18-model **216** preselected 12/class, ALL `sensitive_topics`, previously studied. Fit on official TRAIN only; independent *source* evaluation but **not untouched prospective source evidence**.
- Additional old strong comparator: historical **0.691 Lightning V2** exact full final predictions on those same 216 (89/216 correct). Report C1 vs strong direct-replacement rescues *and* regressions, not only oracle complementarity.
- **Current official 0.715** final outputs are not accessible on the external 216. There is **no** legitimate 0.715 paired comparison in this test; no claim about beating champion.

## Stop rule

**Retire** if real POS order C1 fails to beat POS frequencies C0 or fails the shuffled-order control or loses transfer signal; *do not* tune on official OOF/Oleg216 to rescue. Even if both beat, treat as **hypothesis only** until independently sourced, source/prompt-disjoint, strong-baseline paired truth exists. Do not submit a standalone weak POS classifier or imply 0.76. The user explicitly rejects replacing strong 0.715 with a weak proxy and paid target-model API calls.

**Current result:** only frozen executed workflow dispatch; **NO accuracy, no new training of QLoRA, no paid API, no new Kaggle submission**.
