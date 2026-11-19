# NTL — fixed prompt-disjoint hierarchical hard-negative representation diagnostic (2026-11-19)

Status: **preregistered before executing either arm**. Authorized by owner's "go" immediately after proposing the fixed hierarchy+hard-negative experiment. One CPU run, no external sources, no Kaggle GPU, no official submission.

## Research question
Would an explicit model-family decomposition and representation-level, **same-prompt sibling hard-negative contrastive** learning improve genuine prompt-grouped 18-label classification compared with:
- **C0**: flat 18-class CE on the same trainable representation;
- **CH**: soft hierarchical family-head + within-family sibling head, **no contrastive loss** (capacity/formulation control);
- **C1**: same architecture and exact training recipe as CH, plus weighted supervised contrastive loss with siblings under matched prompt as hard negatives.

Only C1−CH tests the contrastive treatment; C1−C0 measures the full proposed architecture. A separate older simple flat-vs-hierarchy LinearSVC test had a +0.0093 source cohort delta, failed its own gate; this experiment is NOT a repeat of that method or the rejected prompt-column CE.

## Data and evaluation contract
- Exactly official TRAIN `data/train.csv`, 2699 responses, 150 prompt groups, 18 labels. Pin SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`; immutable `artifacts/phase1/folds.csv` SHA256 `94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d`.
- Fixed 5 folds, 30 entire held-out prompts per fold, no cross-fold prompt, text/label-only inference. No external sources, no test labels, no contaminated incumbent pseudo-OOF.
- Each fold fits char TF-IDF ngrams 2–5 (case-sensitive; min_df 2; max_features 24k; sublinear_tf) and SVD 256 (seed 20261118, 5 iterations), **train prompts only**. L2-normalize shared features. This deliberately shares the previously executed prompt-CE experiment's input representation for comparability. Encoder above the frozen features: `Linear(256,128) → GELU → LayerNorm`, trainable with each objective.
- Three arms with the same initial encoder and fixed data order for a fold; model family `claude,gemini,glm,gpt,kimi,qwen`. Full flat 18-way CE vs soft family `P(family|text)*P(model|family,text)` vs that hierarchy + contrastive.
- Fixed seed 20261119 + fold; 120 optimizer steps; prompt-batches of 12 distinct randomly selected train prompts at each step, AdamW lr=0.003, weight decay .01, no early stopping; contrastive weight .15, temperature .2. Positive pair = same true model, different prompt; negative pair = another model, doubled for same family and 3× for same-family same prompt. **No label metadata at heldout inference.**
- C1 classifier loss remains exactly CH. Only train-time contrastive changes; no inference-time prompt assignment, grouping, class prior or threshold tuning.
- Five-fold OOF pooled macro 18-class recall, family and within-family error rates, rescues / regressions vs C0 and CH, BA by true class, per-fold paired delta. 2,000 stratified-at-fold prompt-cluster bootstrap replicates for differences. Aggregate JSON only in GitHub artifact, no row IDs/text/OOF predictions published.
- **Primary**: C1−CH `ΔBA`; **secondary** C1−C0. Predeclared gate: primary ≥ +0.03 BA, prompt-cluster 95% CI lower > 0, no increase in cross-family errors, and >0 net rescues for within-family mistakes. If not, **REJECT** the fixed architecture, no hyperparameter rescue. If C1≥CH but CH<C0, do not promote.
- Existing historical ruler: paired prompt-cluster 80%-power MDE ≈.026 for a *different* historical comparison, effective independence could make MDE much larger (150 clusters vs 2699 rows). Prospective expected effect is `≥.03` under a hypothesis that representation-level sibling metric learning fixes at least 5% of the current family-confused errors; **not an empirical forecast**. The actual achievable effect on these reused prompts is unknown.
- **Crucial**: This is a development mechanistic diagnostic on repeatedly-used train folds, NOT an unbiased estimate of the current official **0.702** system. The scored upstream strong ensemble has contaminated/prefit lineage, so there is no valid same-fold `0.702` OOF control. Even if gate passes, NO Kaggle score-facing promotion without a separate non-spent validation source and compelling fully scoped deployment path. Official best remains 0.702.
- Competing hypotheses: H1 explicit family and hard sibling negatives create transferable source-invariant identity signal; H0 weak char representation and highly coupled optimization only repackage source/style biases and may harm family generalization. A shuffled negative control was already negative for prompt-column CE, so contrastive is a different representation-level intervention, not evidence of a new general model family.

## Pivot / stop conditions
If contrastive adds <.03, CI overlaps zero, or cross-family errors increase: STOP this exact objective. Do not run weight, temperature, batch-size or seed sweeps on these same prompts. Consider representation backbone/feature ceiling or selective re-inference next, not more external-source searches. No GPU or official submission this run.
