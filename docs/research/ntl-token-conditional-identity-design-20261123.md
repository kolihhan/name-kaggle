# NTL — design-first research reset: token-conditional identity, no compulsory fresh-18 prerequisite

**Date:** 2026-11-23 MYT. **Status:** DESIGN FROZEN FOR FEASIBILITY; **NO MEASURED NEW BA YET**.
**Research branch:** `research/ntl-onepass-generative-identity-design-20261123`. **Keep original scored 0.715 Public BA (#58680531) unchanged.**
This is architecture research, NOT a 0.715 postprocessor, threshold patch, ensemble, or permission to tune on Public LB.

## 1. Correct the research prerequisite

An independently sourced, version-certified full 18-label benchmark is **not needed before** proposing, prototyping, training, or testing an architecture on permitted official 18-label TRAIN data. It is required **only for the stronger empirical claim** that a candidate genuinely improves generalization across those 18 exact sources/channels, unless another genuinely independent full-label assessment is available. No fresh data does not mean no research.

Allowed now: frozen prompt-disjoint official TRAIN for exploratory 18-way mechanism development; provenance-eligible overlapping-label multi-source subsets for *subset-scoped* transfer/falsification; all rights respected.
NOT allowed: call repeated original OOF unseen evidence; mix 13-label channel A and five missing labels from channel B and claim source-independent macro BA; generalize a two-class gain to 18-way; call publisher asserted served-model names authenticated; tune official Public score as training feedback.

## 2. Mathematical retraction and related work

The previously suggested simple baseline correction `S_m(x) = L_m(x) - L_shared(x)` is **identical** to `L_m(x)` under class argmax whenever the same `L_shared(x)` is subtracted from every class. It cannot by itself improve BA. Pre-existing source-conditioned LM models and clipped token likelihood (ATLAS, Ko et al., 2026, https://arxiv.org/abs/2610.19583) are *prior art*, not our invention. ATLAS is mainly tested at family level, not evidence of exact NTL 18-version transfer.

Existing NTL evidence that this design must not ignore:
- Historical official TRAIN 2699/150 prompt/18 label: fixed strong sparse four-view OOF 0.3670892.
- DistilGPT2 context surprise 55 features OOF 0.1952026; context present, standalone poor.
- DistilGPT2 projected-64D, 128-token-category conditional heads OOF 0.1826547, below even source unigram 0.2022620; weak genuine contextual generative heads **already tested**.
- TFIDF-SVD256 small 128D hierarchy+hard-negative CH 0.275963 versus C1 0.273741; contrastive did not help.
- Old GLM 0.683 real cross-source reroute 88/128 to 77/128; NOT a test of current 0.715.
- Source-shift Oleg 216 on former 0.691 scorer 89/216, sensitive-topic only; cannot estimate new 0.715 transfer.
- Attempted weak SVC global Hungarian override dropped 0.715 to official 0.594; never use weak evidence to overwrite champion.
These are non-comparable model families/OOF/Public; do not compute a BA delta between unrelated sets.

## 3. Competing hypotheses — first NEW mechanistic test

**H1 (token-choice interaction):** Generator identity is partly encoded in how an actual chosen token interacts with its *preceding causal context*. A **frozen strong backbone**, full actual token identities, and class-specific conditional interaction can recover a measurable increment over an equally trained direct discriminative head **without** 18 source-conditioned full-vocabulary LM passes. This differs from the retired DistilGPT2 128-token-category *P(token bucket | context, source)* head: here the observable is token-level *P(source | context, chosen token)*, optimized for final text-level attribution, preserving the full observed vocabulary.

**H0 (lexical/source artifact):** Gains, if any, are explained by unigram token identity, response length, topic or official-train-specific style; stronger backbone alone gives equal or better performance, and source-changed data breaks the interaction benefit.

**Not a claim of original theoretical invention**; the novelty target is a specific efficient full-vocabulary conditional *discriminative* energy representation and carefully controlled counterfactual, not an ATLAS reproduction or a generic new ensemble.

### Architecture: TCAE-v0, one backbone forward

Input text x tokenizes to y_1 ... y_T under a pinned public pretrained causal LM: **PRIMARY Qwen/Qwen3-14B-Base** (HF reports 14.7683B parameters, Apache-2.0). **SECONDARY hardware fallback Qwen/Qwen3-8B-Base** (HF reports 8.1907B parameters) *only if a pre-score, pre-training 14B inference/memory/throughput feasibility check demonstrably fails*. Neither DistilGPT2 nor 1.7B is eligible as the study backbone. Both candidates' exact revision SHAs, quantization config and hidden-size details must be frozen before the real experiment. If neither fits available GPU resources, mark blocked rather than automatically reverting to small models.
At position t, use h_(t-1) from causal last-hidden state (BOS for first token), and the pretrained frozen LM **output projection weight row** e(y_t) of the **actual next token** (compatible with LM heads implemented as nn.Linear, not just nn.Embedding). No future token is allowed into h_(t-1).

```
a_t       = A h_(t-1)                                      [rank r]
u_(m,t)   = (B_m a_t) dot e(y_t) + b_m                     [18 source energies]
S_m(x)    = mean_t u_(m,t)                                 [single-text logit]
P(m|x)    = softmax_m S_m(x)                                [18 labels]
train loss = CE_18(S(x), true_model)                       [per-response]
```

- **One shared causal LM forward**. No 18x backbone runs; computes all 18 energies from the same token/context embeddings. It is a *discriminative energy model*, NOT a normalized generative likelihood.
- Rank `r=16` frozen for first trial; shared A and 18 B_m trainable, along with b_m; backbone/output embeddings frozen, no target-generator weights or external publisher outputs. Track the exact trainable parameter counts.
- Input cap `256` tokens if genuinely feasible under memory/runtime; record truncation. This improves coverage vs the failed first-64-token bucket study but capacity is a **confound**: ablate token cap only in a separately pre-registered study, not this one.
- Score is mean over non-pad tokens; early/late diagnostic strata permitted for analysis but *not* automatically added as fitted weights.
- No response-text embedding can read the true label, prompt_id, row ID, model name or source metadata at inference.

### Fair controls (two arms + a targeted lexical control)

**C0 direct:** same frozen backbone / tokenizer / token cap / split / number of optimization updates and examples. Pool the causal **current-token** hidden states h_t (for all observed response tokens, using hidden[:,1:]) with **masked mean**, then a *capacity-near-matched* MLP 18-way CE. C1 uses h_(t-1) AND e(y_t); C0 must see all tokens, not a disadvantaged prefix-only encoding. If near-matching trainable parameters cannot be achieved without changing the representation, report ratio; do not hide it.

**C1 TCAE:** exactly the formula above; same data, hidden cache, optimizer update budget, seeds, early-stop policy (none), train-fold-only preprocessing, macro BA.

**Clex:** same class-specific source score computed from token embeddings e(y_t) without h_(t-1), trained by 18-way per-response CE. This is a token-identity-only falsifier. It is not an equal-capacity source-normalized LM and cannot replace C0. Require C1 to beat **both** C0 and Clex before interpreting context interaction as useful.

NO source-specific thresholds, changed scorer routing, prompt matching at test time, label-decoder constraints, ensemble mixing or Public feedback.

## 4. Exact experimental setup to freeze BEFORE scoring

1. Official competition TRAIN: precisely 2699 texts, 150 prompt_id clusters, 18 labels; existing canonical immutable grouped five-fold split, verify SHA (official 40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707; folds 94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d). Existing test/hidden competition labels **not** used.
2. Training and all feature normalization are fold-local; hold out entire prompt groups. Pretrained model inference with identical cached hidden states for all arms may be done label-free across rows, provided no training/fitted state sees validation text/labels. No frozen public champion is claimed to have clean OOF on these rows.
3. Seed: 20261123; 5 folds. First prefit **computational preflight** on synthetic text checks pinned backbone can load, real causal h offset is correct, 256-token truncation, QLoRA unnecessary (backbone frozen), RAM/Vram cost, and scores do not change when metadata are removed. The preflight emits **no BA**.
4. Final single experiment: optimizer AdamW lr 0.001, wd 0.01, max 500 updates/fold (not 500 epochs); select batch size once by measured GPU memory **before seeing labels/metrics**, no early-stop tuning. Use per-label balance only in *training* via fixed weighted sampler (class weights on fit fold); scoring always macro BA. No continuous sweeps.
5. Measure C0/C1/Clex paired on **identical 2699 held-out rows**: 18-way macro BA, per-class recall, errors within/cross family, 5-fold deltas, number of rescues/regressions (C1 vs C0 and Clex), 2000 prompt-cluster bootstrap 95% intervals, response-length / prompt strata, truncation, wall clock, peak GPU memory. Log aggregates only, no raw responses, per-row predictions, hidden text, model weights, competition artifacts or source tokens to public Git.
6. **Mechanism PASS only if** C1−C0 >=+0.025 BA AND 95% prompt-bootstrap lower >0, C1−Clex >=+0.025 BA AND its lower >0, >=3/5 folds positive over both, no true-class recall drops >0.15 vs C0. Else RETIRE the fixed TCAE-v0 and revisit backbone/data/objective class, not tweak weights/length/seed on spent OOF.
7. Even PASS = **historical DEV mechanism only**, not estimated Public BA, transfer, novel model-family identity proof, or official submission permission.
8. Before evaluating on a separate source: check source terms, exact overlapping labels and pretraining overlap, and compare matched C0/C1 on common-label subset. A 2-, 5- or 13-label transfer test can falsify scope-specific mechanism; missing labels are NOT filled by another source to fake 18-class BA.
9. Champion reference: official 0.715 remains untouched. No claim to have better competitive score without an actual official scored result or strong independent same-row 18-class evidence. No Kaggle Public tuning loop.

## 5. Cost/runtime and preflight stop

- Historical weak DistilGPT2 CPU attempts did not test 8B/14B causal representation capacity. Primary 14B at FP16 requires roughly 29.5 GB just for model weights before activations; 4-bit weights are roughly 7.4 GB theoretical packed minimum **plus quantization metadata, hidden states, output embeddings and runtime overhead**. Actual feasible peak memory must be measured on the available GPU, not inferred from theoretical 4-bit size. Inference should use frozen 4-bit LM with trainable heads if quantization is needed; do not assume 14B fits 1x T4, or that adding 2x T4 makes CUDA device mapping automatic.
- Budget preflight with **no actual model training**: primary 14B exact revision and real CUDA load/forward on controlled synthetic inputs, record architecture/cpu offload/quantization and tensor-device alignment; ONLY IF infeasible attempt predeclared 8B fallback; hidden cache footprint (2699 x capped tokens x hidden-dim x dtype), 5-fold 3-head training estimated walltime, and hidden 4000-text inference extrapolated from actual token throughput. If memory > available, or projected inference >9 hours with margin, mark INFEASIBLE and *do not silently change the backbone, method or token cap after seeing BA*.
- Do not launch paid inference, external generation, or a competition submission. Do not claim Kaggle GPU quota from an older snapshot. Do not expose official TRAIN text, embeddings or label arrays in a public repository or public logs.
- Current action authorization: **design-first architecture approval; preflight may be prepared.** Expensive Kaggle GPU training is a separate research run *only after the design and model revision/cost contract are pinned*.

## 5a. 2026-11-23 owner correction: stop small-model-first research

The owner's instruction is explicit: repeated DistilGPT2 / 1.7B toy representation experiments are no longer an adequate central research direction. **Replace previous Qwen3-1.7B backbone plan with Qwen3-14B-Base PRIMARY and Qwen3-8B-Base explicit feasibility-only fallback.** We are not claiming model size alone increases 18-class Balanced Accuracy. Keep the controlled C0/C1/Clex design unchanged so backbone power is not conflated with a new architecture or unfair control.

A synthetic-only class-head implementation already passed GitHub Actions but **has not loaded either larger backbone or measured a single real attribution accuracy value**. Any actual GPU call must disclose measured peak VRAM, wall-clock, free quota and source SHA before beginning full 5-fold training.

## 6. Scientific decisions and pivot triggers

```
CURRENT BEST: authenticated Public BA 0.715 (#58680531).
BEST FACT: strong external-code reproduction, not a scored new trainable architecture.
FAILURE HYPOTHESES: (i) weak token representation/context; (ii) domain/source+protocol shift; (iii) strong-score correlation and uncalibrated patching. None is uniquely proven for the 0.715 candidate.
PREVIOUS UNJUSTIFIED RESTRICTION: fresh independently version-certified FULL18 panel before ANY architecture research.
NEW RULE: not required for mechanism exploration; needed before declaring full 18-class source transfer.
SMALLEST DISCRIMINATOR: frozen C0 vs C1 vs Clex prompt-grouped same-row with pinned strong causal hidden states.
PIVOT TRIGGER: fails either controlled delta, family regression, or hidden-scale runtime check.
POST-PASS ACTION: independent subset-scope transfer falsifier before considering 18-class champion deployment.
```

## Prior-art and reproducibility pointers

- ATLAS: https://arxiv.org/abs/2610.19583 ; https://github.com/komseok0109/ATLAS
- Existing negative GCTH source and contract: `scripts/ntl_generator_conditional_token_head_v1_20261123.py`; `docs/research/ntl-gcth-v1-result-20261123.md`
- Existing weak hierarchy result: `artifacts/research/hier-hardneg-20261119/RESULT.md`
- Champion architecture review: `docs/research/ntl-evgendvorkin-eda-vs-0715-0594-20261123.md`
- Repo scientific rules: `RESEARCH_DOCTRINE.md`, `EVALUATION_POLICY.md`, `PUBLIC_VISIBILITY.md`
