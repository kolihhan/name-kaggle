# PPO vs task-aligned learning — research decision (2026-11-12 MYT)

**Scope:** Scientific assessment only. No model fit, dataset ingestion, GPU run, Kaggle submission, changes to an existing candidate, or Public-LB tuning.

## Decision

**Do not launch PPO / GRPO / RLHF as the next Name That LLM classifier experiment. Reopen discriminative representation learning as an *option for a separately authorized, provenance-gated* future experiment. Stop unvalidated small changes around the incumbent.**

The Kaggle task is **single-pass, closed-set 18-class supervised authorship attribution**. Labels for 2,699 official training passages are known; the metric is 18-class macro recall/balanced accuracy. PPO optimizes an expected reward from sampled actions and requires a suitable policy/reward loop. For this known-label one-action classification problem, class-weighted cross entropy and contrastive/ranking objectives offer denser, generally more sample-efficient supervised signal. A terminal 0/1 attribution reward is sparse, macro recall can be addressed by label weighting and validation/calibration, and there is no genuine multi-step environment needing policy-gradient credit assignment. **No claim that PPO could never help is made**: reconsider only if a verifiable multi-step inference or budgeted evidence-acquisition policy is constructed with an independent evaluable reward that cannot be optimized more simply.

## Current facts (not comparisons across incompatible evidence)

- Latest authenticated official **Public BA 0.679, submission `58210835`, COMPLETE**, attributed reproduction of publicly available Ozertuu pipeline; no Private BA known.
- WeMM `58244483`, MGT `58233227` and earlier Lightning `58167751` had **PENDING** ledger status at latest pinned checks; do **not** infer new score from successful private preview.
- Independent own LoRA24 Public BA **0.520**; Short-TRACE standalone Public **0.093**. These are failures of specific executions, not permanent evidence against all trainable encoders or all learned sequence representations.
- Historical grouped OOF: Phase7 **0.370052** => Public **0.369** versus Phase15 **0.405322** => Public **0.357**. Repeated fold improvements did **not** transfer.
- Frozen champion scored on 347 external texts: **260/347 correct**, **87 wrong**, **71/87 within-provider sibling errors**, on **12 observed target labels** while retaining 18 predicted labels; combined *conditional* 12-label BA **0.749795**, not comparable to 18-label official Public 0.679. Most salient Gemini 3.8→3.7/3.6 confusion, plus GPT Sol/Terra/Luna. Source/protocol/version identification unresolved.
- Nine-source intake on Nov 11 found **0 independently qualified unused, protocol-comparable full-18 holdouts**; this does not imply that every partial source is unusable for *exploratory mechanism testing*. In particular an 18-class merged pool is not independent merely because of complete class coverage.
- Official Kaggle: unseen test prompts; public/private test distribution mixes differ; Public LB uses roughly 40% and final Private 60%; ≤9 h CPU/GPU, offline. Source: https://www.kaggle.com/competitions/name-that-llm/overview/abstract and /leaderboard.

## Global diagnosis: model identity vs acquisition protocol

**Observed:** sibling identity separation is a major error mode in the already-inspected 347 rows, and historical validation-to-LB prediction is weak.

**Leading hypotheses (not proven):**

H1: Current representation lacks weak *conditional sibling-specific* cues beyond provider/lab style. Adding generic embeddings or more global balancing may add correlated evidence only.

H2: Training pools conflate exact model identity with source, endpoint version, prompt templates, reasoning mode, temperature, system wrapper or target length. Learned labels transfer poorly because acquisition channel changes.

H3: Existing scoring/validation and high complexity inference (global marginal correction) are too tailored to dev/source mix. Confidence need not mean ground-truth correctness, so unsupervised correction could improve or hurt macro recall.

**Do not collapse H1 and H2.** A new large encoder trained on the same confounded pool is not a cure for H2. Another source/protocol heuristic is not a cure for H1.

## Model-class alternatives and priority

| Priority | Lane | What would discriminate it | Risk / stop rule |
| --- | --- | --- | --- |
| 1 | **Source/protocol identifiability audit** | Model/source/protocol/served-version crossed coverage; source-only baseline vs model-label signal; exact label+prompt dependence | If source and label are inseparable, mark NON-IDENTIFIABLE, do not claim model-specific learning |
| 2 | **Prompt-conditioned sibling contrastive/ranking supervised learning (proposal)** | Same backbone and compute: class-weighted CE control versus CE + within-provider hard negative from same-prompt responses, positives from other prompt groups. Compare new-source *same-row* sibling rescues/regressions, plus total recall and provider drops | Requires independently allowed provenance-cleared training and at least a genuine reserved source/prompt split. An improved familiar GroupKFold OOF alone fails |
| 3 | **Frozen independent representations (WeMM candidate already submitted)** | Authenticated official score only when terminal; ideally a frozen same-row independent test with per-class changes | Different source pipeline and removal of sibling-style training sites confound attribution to WeMM itself; no Public-LB retuning |
| 4 | **Conditional likelihood from actual generators or rigorously calibrated surrogates** | Matched protocol likelihood separation of close siblings on unseen prompts | Most closed generator logits unavailable; surrogate likelihood is not target likelihood; compute constraint |
| Park | **PPO/GRPO over 18 labels** | None justified today: sparse 0/1 reward for known labels versus direct CE | Higher variance/complexity without demonstrated mechanism, especially small fixed label set |
| Park | **Global transport threshold/weights, length, Short-TRACE and old source balancing** | Need fresh mechanism and independent validation to reopen | Existing negative experiments and Public-LB adaptive overfit risk |

### Smallest discriminatory plan — no run authorized by this document

1. **Close the score ledger**: query official Kaggle statuses for `58244483`/`58233227`/`58167751` read-only. A terminal score establishes competition outcome only, not the cause of a gain.
2. **Record source×label×protocol coverage**, and which data could plausibly have fed existing incumbent components. A clean full-18 benchmark may be infeasible; partial classes may support clearly scoped diagnostics, but never claim 18-label BA from a 12/13-label cohort.
3. **Before any expensive fitting**, preregister one paired *architecture-class* test on a reserved, not-yet-inspected cohort, with trainable weights/data/execution explicitly authorized. Control A: same trained encoder/compute with class-balanced CE. Treatment B: CE + same-prompt within-provider sibling hard negatives / cross-prompt positives. Control C: shuffled sibling labels or permuted source assignment where valid. Record per-label recall, source-conditioned confusion, same-row rescues/regressions, and group-bootstrap uncertainty. Protect provider-level recall; test no leakage from prompt bank or source aliases.
4. **Only if B gives independent, repeatable rescued-minus-regressed improvements and no catastrophic class collapse**, decide whether a fresh Kaggle one-shot implementation is warranted. Do not label successful preflight as score. If not, abandon the mechanism rather than layering a post-hoc gate.
5. If a source-qualified reserved benchmark is impossible, state BLOCKED, permit only explicitly exploratory capped experiments, and **do not use Public-LB feedback as a hyperparameter search oracle**.

## Restriction-pivot contract

- **No new training** — reason: no clean provenance-gated full-18 validation and limited user compute. Evidence: CDA-L24 source invalidation; repeatedly reused official folds. Pivot trigger: legitimately licensed source/protocol-separated training and protected evaluation + bounded accessible accelerator. **Status: REOPENED AS SCIENTIFIC OPTION, EXECUTION STILL BLOCKED.**
- **Do not use PPO for classifier** — reason: no sequential action/reward objective and CE gives denser true-label learning signal. Pivot trigger: independently specified sequential evidence-selection decision problem with meaningful reward and budget, and competitive baseline against simpler supervised or bandit methods. **Status: ACTIVE.**
- **No old-fold/leaderboard weight tuning** — reason: proven OOF/Public divergence and hidden public/private mix difference. Pivot trigger: new source/prompt-disjoint evidence, not just another Public number. **Status: ACTIVE.**

## Exit criteria

This decision does **not** report a breakthrough or improvement. Evidence of progress requires a new representation to beat a matched control *on reserved source/prompt-disjoint rows* by predeclared metrics, and eventually an official terminal Kaggle score. The near-term scientific question is: **does genuine sibling identity information survive generation-channel changes, or are we learning the channel?**
