# Independent Research PIC Review — generator identity vs. generation-channel confounding

**Date:** 2026-11-06 MYT  
**Scope:** read-only research and forensic analysis of **already-existing** evidence. **No training, new model, Kaggle submission, notebook execution, new scientific experiment, deployment, or GitHub Actions workflow created.**  
**Verdict:** Previous global hypothesis "mostly sibling confusion" is **incomplete**. The error mechanism differs strongly by provider family. Pursue *conditional evidence reliability*, not a global style-only classifier, unconditional hierarchical routing, or leaderboard ensembles.

## 1. Research question / hypothesis audit

**Question:** What kind of information is missing from our independently developed 0.520 system, and which scientific explanation is supported instead of copying higher-scoring public ensembles?

Candidate explanations:

- H1: The learned representation already identifies most **providers** but cannot distinguish close model **siblings**.
- H2: Some provider families are themselves poorly identifiable under our source/prompt training regime.
- H3: Apparent signatures reflect *generation source/protocol* (prompts, decoding defaults, formatting) rather than model identity.
- H4: Published pipelines profit from their data/model diversity or test-batch assumptions, not any one newly added specialist.

**Decision:** H1 is true for certain families, false as a universal description. H2 is strongly supported in archived GLM/Kimi predictions. H3 remains unresolved. H4 remains unidentifiable without controlled component comparisons.

## 2. Primary evidence and exact re-audit

Original scientific artifact: [GitHub Actions 41398543856](https://github.com/kolihhan/name-kaggle/actions/runs/41398543856), downloadable artifact `13125872633`, archive SHA256 **fa675fcf8ac3860984b449748cf067c3da29389a921417e5e1047bb22a7be8ce**. This turn downloaded and independently verified that SHA. Read `gate_b_output/gate_b_candidate_oof.csv` in memory.

Integrity checks: **2,699 rows, 2,699 distinct IDs, 150 prompt groups, 18 true labels**. Columns `model`, `plain_prediction`, `centered_prediction`, `fold_id`. No official text copied to repository. No hidden test labels, no new predictions.

**Definitions for every true class k:**
- correct = P(predicted exact model = k | true model = k).
- sibling error = P(predicted another model in the **same provider family** | true model = k).
- cross-family error = P(predicted a **different provider family** | true model = k).
- by class: correct + sibling error + cross-family error = 1.
- Each aggregate below is averaged over all **18 classes**, so it is aligned to the Kaggle *model-level* balanced-accuracy weighting; this is NOT 6-provider-class balanced accuracy. The provider family is inferred from the model-name prefix, consistent with the 18 competition labels.

### Full 18-label macro decomposition (new analysis of archived rows)

| Measure | Plain head | Centered head | Centered minus Plain |
|---|---:|---:|---:|
| Model exact balanced accuracy | **0.546761** | **0.547850** | **+0.001089** |
| Error: wrong sibling in same provider family | **0.259771** | **0.241583** | **-0.018188** |
| Error: wrong provider family | **0.193468** | **0.210567** | **+0.017099** |
| Provider-family correct, macro over 18 model labels | **0.806532** | **0.789433** | **-0.017099** |

**Important finding:** The centered head's ~**1.82 percentage point** reduction in within-family errors coincides with ~**1.71 percentage point** *increase* in cross-family errors. Net model-level BA increases just ~**0.11 percentage points**. Thus "centered helps siblings" is incomplete; it trades one error type for another. It is an **observed association between different heads**, not a causal isolation of centering alone.

Micro cross-check for plain: **1,476 exact correct**, **701 sibling-wrong**, **522 cross-family-wrong** = 2,699 rows; sibling errors are **57.32% of wrong rows**. Micro and macro agree closely because almost every class has 150 examples, but for competition conclusions use the 18-class macro figures above.

### Different families have fundamentally different bottlenecks

| True provider | Plain exact model recall (macro over that provider's labels) | Plain wrong sibling | Plain wrong provider | Plain provider-family correct |
|---|---:|---:|---:|---:|
| Gemini | 0.6033 | **0.3717** | 0.0250 | **0.9750** |
| GPT | 0.5467 | **0.3333** | 0.1200 | **0.8800** |
| Claude | 0.6097 | 0.2312 | 0.1591 | 0.8409 |
| GLM | 0.2367 | 0.1833 | **0.5800** | **0.4200** |
| Kimi | 0.5133 | 0 (only one Kimi label) | **0.4867** | 0.5133 |
| Qwen | 0.6600 | 0 (only one Qwen label) | 0.3400 | 0.6600 |

For Gemini, identifying the family is almost solved *in this archived grouped OOF*; exact siblings are not. For GLM, an approach that first hard-assigns family would enshrine the wrong provider **58%** of the time given the present plain-head predictions, before any GLM sibling specialist can help. For single-label Kimi and Qwen, there is no *within-family* sibling problem under this competition's label inventory, and all errors are necessarily cross-provider.

Examples from class-level counts (prior probabilities not retuned): Gemini 3.7 Flash correct **0.3667**, sibling wrong **0.6133**, cross-family wrong **0.0200**; GPT 5.6 Sol correct **0.3333**, sibling wrong **0.5600**, cross-family wrong **0.1067**.

Fold-level plain error type (descriptive, repeated-consumed folds): 
- fold 0: 141 sibling / 108 cross;
- fold 1: 145 / 106;
- fold 2: 145 / 109;
- fold 3: 126 / 100;
- fold 4: 144 / 99.

**Risk:** These are old, repeatedly observed official OOF rows and a trained/fitted system. The 5 folds are not new independent distributions and the holdout bank was used in earlier development. Do not treat these slices as fresh validation, use family rates to fit new thresholds, or extrapolate rates to the hidden test.

## 3. Corroboration and contrary evidence

### Existing in-repo negative results

1. `artifacts/candidates/prompt-residual-style-v1/RESULT.md` (earlier branch): prompt-centered LDA reduced mean external-source accuracy ~**0.03579**, worse on **11/13** sources. Do not rebrand this failed method as a new candidate.
2. `artifacts/candidates/prompt-invariant-contrastive-v2/RESULT.md` (earlier branch): paired source-transfer delta ~**+0.002685** but CI crossed zero; failed preregistered gate. Do not retune its loss.
3. `artifacts/research/source-transfer-views-20261106/RESULT.md`: SURFACE-only weaker than LEXICAL on both examined external banks; FULL-minus-LEXICAL +0.00734 on one, -0.00160 on the other. **Not** evidence content is universally robust because source labels differ and lexical char n-grams encode style too.
4. `artifacts/candidates/dct-temporal-fingerprint-v3/RESULT.md`: DCT token trajectory transfer **UNRUN**, not a candidate with proven generalization. No GPU work authorized.

### Higher-scoring public systems are not causal proof

- Independently built LoRA24 **0.520** on official Public leaderboard; author's Lightning V47 reproduced **0.666**; Ozertuu V3 reproduced **0.668**. The latter two remain **attributed reproductions of other people's work**, not novel project research.
- Exact prior source audit `artifacts/research/ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md`: V47 simultaneously changes model count, data sources, fitted sparse readouts, family-conditional specialists, inferred test groups, calibration, and auxiliary likelihood scores. Cannot assign 0.146 of public gain to one mechanism.
- V47 already implements conditional family specialists and ATLAS-like uncertainty evidence. Another ordinary family-specific expert is **not** novel by itself; our primary evidence also argues against applying the same family routing to GLM, Gemini, GPT, Kimi.

### External primary/field research (claims scoped)

- [Official Kaggle description](https://www.kaggle.com/competitions/name-that-llm/overview/abstract) confirms 18 model labels, **balanced accuracy per model**, and public/private may have different model mix. [Official data description](https://www.kaggle.com/competitions/name-that-llm/data) confirms test prompts unseen and test rows have no prompt IDs.
- [Kaggle community thread 751802](https://www.kaggle.com/competitions/name-that-llm/discussion/751802) *self-reports* that "one model per inferred prompt" Hungarian assignment improved grouped OOF **0.374 -> 0.481** but changed Public **0.353 -> 0.350**. It does **not** prove the actual hidden test has repeated model/prompt rows, but is a strong counterexample to the assumption that grouped-train benefits must transfer.
- [ATLAS, Who's Behind the Text?](https://komseok0109.github.io/ATLAS/) explicitly models source-conditioned token likelihoods and evaluates domain/model-version shift. **Its published benchmark task is primarily model-family attribution, not these 18 exact competition labels**. Its paper metrics cannot be substituted for Kaggle BA.
- [Profiler, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1265/) analyzes contextual inference pattern via surrogate logits and context losses. Its reported AUC gains are on other models/datasets and do **not** validate any Name That LLM modification.
- [Where Does Authorship Signal Emerge in Encoder-Based Language Models?, BlackboxNLP 2026](https://aclanthology.org/2026.blackboxnlp-1.47/) finds scoring mechanism can change where learned *human authorship* information appears. Potentially useful representation hypothesis, but **not direct evidence** that DCT, late interaction, or another scoring head will distinguish model-generated siblings here.
- [IJCAI 2026 Implicit Identity Technologies survey](https://ijcai.org/proceedings/2026/880) distinguishes output-intrinsic fingerprinting from explicitly embedded watermarking and emphasizes identifiability, robustness, and deployability. Only output text is observable here; don't propose privileged watermark keys or access to generator internals.

## 4. Better global hypothesis

**Supported by this audit:** There are at least **two attribution regimes**, not a single universal "style fingerprint" failure.
- Regime A (Gemini/GPT): strong provider-level identity, inadequate within-provider *sibling* discrimination.
- Regime B (GLM and some single-label families): provider-level confusions dominate; sibling-only correction cannot help them.
- Claude mixes both.

**Unresolved causal explanation:** These patterns may arise from real underlying generator signatures, class/protocol distributions, learned representation capacity, or data-provenance imbalance. The present model-only OOF cannot identify cause.

**Higher-value research question:** Is there *source/protocol-stable token-level evidence* that separates model identity from generation-channel artifacts, and can it be used with **family-conditional error diagnostics** while protecting weak family attribution? This reframes evaluation rather than announcing an invented architecture.

This is a **research hypothesis, not a proven idea or submission candidate**. Its key novelty bar is explaining and transferring conditional evidence across sources, *beyond* already published family gates, ATLAS likelihood, and Profiler context loss. Simply combining those methods is not a new contribution.

## 5. Adversarial predictions and stop rules (proposal only; NOT executed)

| Claim | What independent evidence would support it | What would falsify it |
|---|---|---|
| Gemini/GPT sibling errors are stable under generation-channel shift | Same-label and prompt-disjoint multi-protocol heldouts retain high provider accuracy but weak sibling recall | Error decomposition reverses or disappears outside official OOF |
| GLM's problem is genuinely provider-level rather than source-bank mismatch | Cross-source GLM examples generated with documented model/version/protocol retain cross-provider confusions | True source/version/protocol controls restore provider recall without architecture changes |
| Sequence/context evidence adds identity information beyond pooled embeddings | Matched, heldout, truly independent token-level comparison; equivalent inference/computation budget | Performance gain disappears when source/protocol/prompt differences are controlled |
| Family-conditional decisions can improve exact attribution | New candidate passes source-transfer checks with no collapse in cross-family macro recall and especially GLM/Kimi recall | Conditional improvement is achieved only by cross-family regressions, or disappears off-source |
| Test-time inferred one-model-per-prompt constraint is reliable | Explicit source-construction documentation and genuinely independent hidden-like evidence | Repeats/partial groups or externally observed failures of hard assignment |

**We do not have the required clean matched-protocol/source holdouts, aligned frozen-model predictions and manifests now.** Prior banks were examined, and their label mixes differ. Therefore there is no grounds to proclaim validation or to create an experiment/submission today.

## 6. Research doctrine checkpoint / decision

- **CURRENT BEST INDEPENDENT:** our LoRA24 Public **0.520**; source heldout A1 > A0; official grouped OOF **0.54676** (spent).
- **BEST REPRODUCED EXTERNAL:** Ozertuu V3 Public **0.668**, from another author. Apex V1 `58045266` was previously accepted but its terminal score is **not checked in this research audit**.
- **MAIN FAILURE:** heterogeneity of error type: 18-label macro wrong-sibling 0.25977 vs wrong-provider 0.19347; **Gemini ~97.5% correct provider vs GLM ~42.0%**.
- **GLOBAL BOTTLENECK HYPOTHESIS:** provider evidence and within-provider evidence have different robustness and confounding; one shared correction likely sacrifices one for the other.
- **LOCAL NEXT RESEARCH TASK (read-only):** trace published mechanisms and exact source/decoding provenance **as evidence**, not run another simulation, model fit, or new competition submission. Determine whether a genuinely independent matched-label, multi-protocol dataset exists before promising empirical discrimination. Examine code for the one-per-prompt assumption and family mass preservation before claiming they are safe.
- **RESTRICTION:** No new user-run ML experiments / model fits / Kaggle submissions / deployments or Actions launches. **Reason:** explicit current user instruction and prior unnecessary leaderboard chasing. **Status:** ACTIVE; reopening requires a new explicit request from the user, not a "go" attached to research.
- **RETIRED as active next direction:** generic SVC/view-weight sweeps, duplicated author Notebook reproduction, class-bias tuning from 2,699 spent OOF, hard prompt-group Hungarian matching, prompt-residual LDA and re-tuned prototype contrastive.
- **NOT RESOLVED:** source-conditional reliability, full upstream training manifests for third-party LoRA, independent sibling-transfer rows, whether any new method improves hidden 18-label BA.
- **PIVOT TRIGGER:** New *non-reused*, verified-provenance evidence shows the predicted within- vs cross-family behavior fails or identifies a different bottleneck; then revise diagnosis, rather than fine-tune old heldouts.

**Disposition:** Completed a new, auditable analytic decomposition of archived evidence; no model or deployment work. Research PIC remains responsible for the scientific problem and its falsifiable explanation, not running published submissions.
