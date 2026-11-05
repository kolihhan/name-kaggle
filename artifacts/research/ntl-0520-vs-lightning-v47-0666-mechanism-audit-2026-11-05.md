# Name That LLM — Mechanism audit: own 0.520 vs reproduced public Lightning V47 0.666

**Date:** 2026-11-05  
**Status:** Exact-source static code audit COMPLETE; causal score attribution NOT established; no new training or submission.  
**Question:** What *actually* differs, which explanations are justified, and what smallest independent test would separate the alternatives?

## Decision in one paragraph

**FACT:** Our Kaggle code-competition submission `58008463` completed with **Public 0.666** (read-only authenticated Kaggle detail in [Actions 42014195583](https://github.com/kolihhan/name-kaggle/actions/runs/42014195583)); previous self-developed single-model Stage-2 LoRA24 submission scored **0.520**. The +0.146 absolute/+28.08% relative improvement is a **between-systems public-leaderboard difference**, *not* a controlled architectural ablation. We reproduced another author's published, already-fit V47 solution, **not** an independent breakthrough by this project. Exact pinned original `lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble/47`, notebook byte SHA-256 `be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d`, 37 executable code cells; source pull/check [Actions 42016381979](https://github.com/kolihhan/name-kaggle/actions/runs/42016381979) and preceding [41953332095](https://github.com/kolihhan/name-kaggle/actions/runs/41953332095). **Most likely broad explanation:** much larger and more diverse already-fitted signals, source-augmented specialist heads, inferred group/within-family reasoning, and calibrated mixing. **UNKNOWN:** the individual contribution of any component to the observed +0.146, because we have no identical-test per-component ablation.

## Evidence anchors

1. **Own 0.520 implementation:** `scripts/two_stage_lora24_submission.py::run_final_fit`, `_infer_plain_test`; `scripts/build_two_stage_lora24_kaggle_kernel.py`; exact predecessor metrics and confusions in `artifacts/research/ntl-sibling-confusion-pivot-2026-11-04.md`.
2. **Published V47 mechanism:** exact pulled Apache-2 Notebook (SHA above), which creates `core_math.py`, `core_readout.py`, `core_worker.py`, `ntl660_readout.py`, `legacy/comp_cached.py`, `gsq/infer.py`, and `ensemble.py` using `%%writefile` cells. Source not copied into the repository; temporary source audit artifact, [Actions 42016381979](https://github.com/kolihhan/name-kaggle/actions/runs/42016381979), retained briefly.
3. **Competition benchmark:** [official evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/abstract) is balanced accuracy over 18 labels; public leaderboard represents approximately **40%** of hidden test, private about **60%** ([Leaderboard page](https://www.kaggle.com/competitions/name-that-llm/leaderboard)). This is *not* full private transfer evidence.
4. **Precursor evidence:** Stage-2 own grouped OOF plain BA `0.5467611235` over 2,699 official rows; 1,223 wrong rows, including 701 within-provider-family errors (57.32% of errors). This was repeatedly consumed OOF; no transfer claims based on that percentage alone.

## Actual implementation comparison

| Axis | Our scored 0.520 `two-stage-lora24-v1` | Lightning V47 scored 0.666 (original author, directly reproduced by us) | What can be concluded |
|---|---|---|---|
| Base learned representation | One frozen Qwen3-8B-Base trunk with Stage-1 LoRA24 checkpoint; Stage-2 official-data fine-fit; plain 18-way head | **Six core model members:** `q14v5` Qwen3-14B, `q8v5` Qwen3-8B, `ll8v5` Llama-style prefix, `n14b` Qwen3-14B, `q25_14` Qwen2.5-14B, `deb` DeBERTa; several LoRA members average **six adapter checkpoint heads** (five folds + full fit). Plus supplementary specialists | Representation diversity/capacity differs materially; effect size UNKNOWN |
| Learned non-neural stylistic evidence | The earlier project tested strong sparse features as separate rejected phases; **0.520 scored artifact uses the single neural plain head**, not an ensemble of them | Core loads **three fitted sparse readouts**: `svc4` weight **12**, `tfidf` **0.25**, `distort` **0.10**, as weighted log-prob votes; complement adds stylegrams, encoders, LoRA heads, token attention, family gate heads | Earlier evidence that sparse approaches alone reached only ~0.37 does not rule out complementary marginal information alongside stronger neural models |
| Group-conditioned information | Final `_infer_plain_test` does not group hidden test rows and does not use the centered head | `topic_groups` uses test-text TF-IDF similarity, hierarchical grouping capped at 18; where groups permit, averaged centered adapter heads and pretrained Gaussian shared-prompt group heads contribute; complement has independent stylegram groups | **Hypothesis:** inferring prompt/topic peers offers cancellation of prompt-specific content; **risk:** wrong groupings or distribution/size shift; no ablation proven |
| Model sibling discrimination | Original consumed grouped OOF has 701 same-family errors (not a fresh holdout) | Complement includes separate **Claude style/LoRA, GPT, Gemini, GLM family gates** that preserve family total mass and adjust sibling probabilities, confidence-aware; Atlas likelihood specialist runs on uncertain rows | Mechanistically aligned to the observed error class; **cannot infer** how many errors it fixes or whether it explains 0.146 |
| Test-time inference composition | Qwen3-8B LoRA24 plain-head logits, argmax | Sequential GPU/CPU stages: core logits, complementary logits, optional Atlas, full GSQ Qwen3.8-27B inference; probability mixture; official-train-only typography evidence and class-marginal balancing; argmax | V47's scoring is the whole pipeline. Replacing only backbone with a larger model does **not** reproduce it |
| Prior training data | Frozen Stage-1 external source(s), then **2,699** official rows at Stage-2 | V47's embedded `trained_artifact_contract` declares `rows_seen.external=119899`, `rows_seen.official=5398` for a specific trained artifact; `ntl666-prefit` provenance declares a **190,588-row pool**; 16 public Kaggle dataset inputs, eight linked Notebook inputs, five model refs | Material **data scale and provenance** differences. These are different audit counts and **must not be added** or interpreted as unique independent labeled examples; overlap, creation dates, legality, and hidden similarity are unverified |
| Submission mechanics | Our Notebook version with own final fit and inference | Direct official competition submission of existing Apache-2 public V47 Notebook, `-k ... -v 47`; official server scores full hidden rerun | No training by this team; personal GPU quota is irrelevant to direct existing-version submission |

### Important: real V47 probability weights (not just the title)

The published Notebook's **initial** CONFIG literal contains `gsq_share = 0.07`, but **cell 45 overwrites it to `0.05` before the actual `ensemble.py` runs**, and `ensemble.py` asserts exactly `0.05` (source: cell 45 and `ensemble.py::main`). Thus the actual scored probability mixture is:

```text
P0 = 0.85 * softmax(core) + 0.15 * softmax(complementary)
P1 = P0                               (clear rows)
P1 = 0.90 * P0 + 0.10 * Atlas         (uncertain rows; margin < 0.30)
P2 = 0.95 * P1 + 0.05 * GSQ_27B
final = balance_logits(P2) + gated_typography_evidence
label = argmax(final)
```

Equivalent **pre-postprocessing** mixture coefficients:

| Rows | core | complementary | Atlas | GSQ 27B |
|---|---:|---:|---:|---:|
| No Atlas gate | **80.75%** | 14.25% | 0% | **5.00%** |
| Atlas gate active | **72.675%** | 12.825% | 9.50% | **5.00%** |

These percentages concern the **top-level probability blend only**, not causal influence, model importance, CPU/GPU runtime share, or contributions inside the core/complement streams. **Reject the simplistic claim that 'using Qwen3.8-27B alone caused the 0.146 gain'.** Its probability mix share is 5%; a 5% expert can still flip small-margin predictions, but without paired predictions we cannot count those flips.

### Deep mechanisms visible in source

- **Core learned representations:** `core_worker.py` runs staged LoRA checkpoints, `core_readout.py::member_scores` averages log-softmax outputs across six adapters for `q14v5`, `q8v5`, and `ll8v5`; uses plain, group-centered sibling heads, and fixed Gaussian scoring when available. `core_votes` adds **fitted** SVC4/TF-IDF/distorted-text heads; this means the core is **not simply a LoRA majority vote**.
- **Family conditional gate:** `legacy/comp_cached.py::_apply_conditional_family_gate` normalizes a family's sibling logits and interpolates fitted family scores with confidence-dependent strength; it restores the same total family log-mass using logsumexp. This specifically avoids a specialist overriding cross-family decisions merely to change sibling identity. Testable hypothesis, not proven improvement.
- **Group inference:** `core_math.py::topic_groups` fits TF-IDF on **unlabeled test passages** to estimate prompt peers, then `core_readout.py` subtracts inferred group averages; `gauss.py` scores grouped feature matrices. **This is transductive unsupervised fitting** during inference, even though no supervised model weight training occurs in V47's submitted run. It uses no hidden labels in the visible code.
- **Calibration:** `core_math.py::balance_logits` uses each class's *official training* frequency and the predictions' test marginal, strength .30, target_mix .35, 100 iterative balancing steps, blended on log probabilities. `typography_evidence` learns Bernoulli-like punctuation frequencies from official train and gates them by predicted uncertainty, passage length and observed marker coverage. These can improve or hurt transfer depending on hidden class proportions and topic/format distributions.
- **Uncertainty routing:** `ensemble.py` calls the Atlas head only when the first two mixture scores differ by less than .30 (for >=50 rows); its top-level influence on selected rows is at most 9.5% of probability mass before postprocessing. Complexity is targeted toward ambiguous examples, not uniform compute.
- **Preview shortcut:** `emit_known_preview` fingerprints **exactly three public interactive example rows**, emits three format-correct predictions without model inference, and explicitly records `model_weights_loaded=False`. It is **NOT** a valid GPU/hidden-sized preflight. For any other inputs, V47 goes through the full multistage ensemble. Our authenticated official `COMPLETE 0.666` confirms the competition scorer accepted and scored the official submission, **not** that the preview's 20-second run exercised the hidden full pipeline.

## What the +0.146 DOES NOT prove

1. **No controlled ablation.** 0.520 and 0.666 change model count, model family, representation, trained artifacts, source provenance, calibration, inference grouping and conditional specialists simultaneously. There is no paired held-out per-row matrix or component-off score; assigning percentages of the improvement to particular components would be fabricated.
2. **Public-only measurement.** Kaggle's ~40% public portion can differ from private 60%. The private score for `58008463` is still blank; neither public gain nor author version history proves private transfer.
3. **No full provenance clearance.** The 16 public dataset refs are metadata-readable, but the training/source snapshots, collection dates, potential overlap with official or hidden prompt banks, external data licenses and independently sourced holdouts have **not** been individually audited. Public attachment != leak-free transfer evidence; equally, there is no basis to accuse the author of leakage.
4. **Hyperparameter selection unknown.** We do not know which parts of V47's frozen probability weights, class-balancing choices, or auxiliary experts were chosen using the same public LB. An upstream version score by itself is not independent validation.
5. **Old OOF error fraction is descriptive.** The 57.32% same-family error fraction counts only our *already-consumed* 0.520-development OOF and is not proof that V47's family specialists rescue those rows on unseen data.
6. **Artifact success is not research novelty.** We reproduced the original author's work via existing Kaggle code, with appropriate attribution. Engineering/score reproducibility is valuable, but it doesn't demonstrate a new learning or attribution method.

## Updated bottleneck diagnosis, restriction and decision

- **CURRENT BEST:** reproduced public V47 score 0.666 (third-party code), our independently developed candidate 0.520.
- **REAL EVIDENCE OF STRENGTH:** authenticated Kaggle scorer returned 0.666; our own LoRA24 Stage-2 plain OOF 0.54676 and external Stage-1 transfer A1 0.46571 vs A0 0.39424 (spent evidence, not fresh).
- **MAIN FAILURE MODE:** the old learned representation has family-sibling confusion in already-observed OOF; public-to-grouped-OOF drop indicates possible source/domain change (not proven).
- **GLOBAL HYPOTHESIS:** single-representation/single-head and insufficient independent source/stylometric/sibling cues constrain the current independent method; extra data + complementary representations may matter more than raw backbone size.
- **RESTRICTION:** No user-side retraining or repeated Kaggle submissions for this audit.
- **RATIONALE:** User preference and GPU quota; prevent LB feedback loops.
- **REOPEN TRIGGER:** a separately approved, pre-registered disjoint-source/prompt experiment establishes an independently useful mechanism and resources are available.
- **STATUS:** ACTIVE for this audit; does not permanently ban training or new model classes.

## Follow-up executed: disjoint paired-predictions availability gate — 2026-11-05

**VERDICT: NOT TESTABLE FROM RETAINED ARTIFACTS; NOT an empirical V47 ablation.** We completed a targeted inspection and did not use GPU, fit models, read hidden test data, or make a competition submission.

| Checked evidence | Exactly available | Can it support unbiased 0.520-vs-0.666 mechanism comparison? |
|---|---|---|
| Own original Gate-B Actions run `41398543856`, artifact `13125872633` (downloaded and ZIP contents verified this turn) | `gate_b_output/gate_b_candidate_oof.csv` and five fold-level prediction CSVs on all **2,699 official training rows** | **No.** This is repeatedly reused official OOF for our model, while the V47 and its upstream final checkpoints already use official train. V47 predictions on these rows would be in-sample/contaminated, not a fair paired holdout. |
| Own original Gate-A Actions run `41333631967`, artifact `13094213203` (downloaded and ZIP contents verified) | **One aggregate JSON** `two_stage_lora24_gate_a.json` containing source/condition metrics, not individual predictions/logits/embeddings | **No.** No row-aligned labels and predictions for an actual paired test; those source groups also informed Stage-1 model selection. |
| Published Lightning V47 source audit `42016381979`, artifact `13410085387` | Exact SHA-pinned public Notebook and metadata only; the Notebook source builds `core.npz`, `complementary.npy`, `pre-balance.npz`, `final-logits.npz` **at inference time** | **No.** Source code and output *recipes* are not independently labeled row-aligned evaluations. Public three-row fast preview uses a hardcoded fixture. There is no cached disjoint labeled component prediction matrix in this inspected source artifact. |
| Project's frozen DCT-v3 transfer contract `artifacts/candidates/dct-temporal-fingerprint-v3/RESULT.md` | 14,506 external heldout rows by `alperen/hailey/willrice`, locked **for v3**; old source fingerprints and labels exist in its own specific contract | **Not presently.** The split's independence from **Lightning's** external training pool and trained heads is not proven, and no V47 predictions on it exist. Reusing the v3 split without checking source/prompt provenance would mistake one model's holdout for another's. |
| Kaggle official submission `58008463` | Status COMPLETE and **Public 0.666**; previous independent submission Public 0.520 | **No.** Two aggregate metrics on a hidden Public-LB subset, with no per-example labels or counterfactual component scores, cannot separate more data from model diversity or family gates. |

**Boundary:** This was a focused inventory of the listed repo and Actions artifacts and the exact published source, not an assertion that no other independent data or private checkpoints could exist anywhere. The necessary combination **(same independently labeled source rows, provable exclusion from every compared training/fitted artifact, and per-row component predictions for both models)** has **not been found**.

### Exact-source CPU mechanism test (completed; NOT a scoring experiment)

- Added reproducible `scripts/audit_v47_family_gate_cpu.py` and Actions workflow `.github/workflows/ntl-v47-family-gate-source-pinned-cpu-audit.yml`.
- [GitHub Actions **42021098034**](https://github.com/kolihhan/name-kaggle/actions/runs/42021098034) **SUCCESS**. Source was pulled from the published exact `/47` Notebook and byte-checked against SHA-256 `be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d`. The two original functions `_np_logsoftmax` and `_apply_conditional_family_gate` were **compiled and executed verbatim from AST** on generated logits. No published model/head was loaded and no competition text was used. Aggregate evidence artifact `13411512310` (`ntl-v47-synthetic-family-gate-proof`).
- For each of five sequential gates (`claude`, `claude_lora`, `gpt`, `gemini`, `glm`), 384 synthetic rows passed the invariants: max change in total family log mass **4.44e-16**, max outside-family logit change **0.0**, all **64** deliberately high-confidence family rows stayed unchanged, and low-confidence rows demonstrably changed.
- **Counterexample to an overly strong interpretation:** A crafted synthetic sample's top individual label changed from `gpt-5.5` to `claude-fable-5` after the Claude gate, **despite unchanged total Claude family mass**. A family-mass-preserving specialist can still change which *family* the top-1 label belongs to. Therefore do **not** claim this gate can never alter the cross-family top-1 decision.
- These tests establish exactly how the **published implementation behaves** for synthetic inputs, **not** that this gate improves balanced accuracy, predicts real errors, or causes part of the +0.146 score difference. No empirical component effect size is identified.

### Scientific next decision

Stop the proposed paired ablation **at the availability/provenance gate** instead of using the spent official folds or making repeated leaderboard submissions. To reopen, first secure a genuinely disjoint labeled external prompt bank **and** training-source manifests for all compared fitted V47 artifacts, plus precomputed aligned logits or one separately authorized inference-only run for both models. Freeze the contrast before looking at outcomes; if data independence cannot be proven, report **NOT TESTABLE**. No retraining or official submit is needed for the current completed audit.

\n### Highest-value next check (inventory executed; empirical paired ablation NOT TESTABLE)

**No training, no Kaggle submit:** Search for *existing* per-row component prediction artifacts and source metadata for the frozen V47 package and our own model. Prefer **external prompts/source banks demonstrably excluded from both systems' model/head training**, not the repeatedly examined official 2,699 rows or the Kaggle public leaderboard. If both per-component predictions and disjoint labeled rows exist, lock a paired comparison (same rows, 18-class balanced accuracy, family-vs-sibling confusion, per-source change) with single-core, plus SVC4, plus fitted family gates, plus test-group logic, plus optional GSQ as counterfactuals. No weight/threshold search. If the artifacts or provenance are unavailable, report **NOT TESTABLE** rather than claiming ablation.

**First priority** is the distinction **source diversity vs representation diversity** (and how much within-family gating really helps); raw `27B vs 8B` model size is too confounded to isolate from the public score alone.

**Explicit stop:** No new source-specific tuning, self-training, hidden-score optimization, Notebook edits, or competition submissions as part of this report.
