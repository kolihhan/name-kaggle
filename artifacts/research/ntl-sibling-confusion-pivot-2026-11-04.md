# Name That LLM — postmortem and next research pivot (2026-11-04)

**STATUS: ACTIVE RESEARCH, NO NEW TRAINING OR SUBMISSION.** This supersedes the mistaken broad project pause inserted after the Cosmorin hidden-run failures. The user asked to stop the repeated failed approach and examine a genuinely different direction, **not** to abandon Name That LLM or switch to Gemma 4.

## Frozen evidence checked

- Strongest valid **own** Kaggle Public balanced accuracy: **0.520**, `two-stage-lora24-v1` submission `57908683`. Stage-1 source transfer A1 **0.4657129266** vs frozen-control A0 **0.3942430082**; Stage-2 official prompt-group OOF plain **0.5467611235** over 2,699 rows. This is positive real transfer evidence, unlike an upstream author's score.
- The copied external Cosmorin published-0.622 Notebook did **not** yield any own valid 0.622 score. Submission `57983359` had detailed user-visible `Notebook Threw Exception`; `57985641` later returned CLI `COMPLETE` with both score fields blank. V3 direct submit returned HTTP 400. Four private synthetic/CPU component probes passing did not establish a hidden-runtime fix. **Stop retries of that mechanism.**
- Official Kaggle rules: 18 classes, balanced accuracy, unseen prompt groups in hidden test, about 4,000 passages, no hidden `prompt_id`, offline notebook inference with `submission.parquet`. These are material evaluation constraints. Sources: https://www.kaggle.com/competitions/name-that-llm/overview/description and https://www.kaggle.com/competitions/name-that-llm/data
- **Important negative external example:** a Kaggle discussion reports that Hungarian one-per-model-per-prompt matching improved one participant's OOF from 0.374 to 0.481 but degraded Public 0.353 to 0.350. This is an **unverified community report, not official test construction evidence**, and warns against exploiting a train-only matching assumption. Source: https://www.kaggle.com/competitions/name-that-llm/discussion/751802

## Read-only original OOF evidence inspection

Downloaded the *existing* original GitHub Actions Gate-B artifact **13125872633** from successful run [41398543856](https://github.com/kolihhan/name-kaggle/actions/runs/41398543856), ZIP digest `sha256:fa675fcf8ac3860984b449748cf067c3da29389a921417e5e1047bb22a7be8ce`. Inspected `gate_b_output/gate_b_candidate_oof.csv` in memory; **never committed official text or per-row predictions to GitHub**. The file contains 2,699 unique IDs, 150 grouped prompts, 18 true labels, and plain/centered predictions. Used provider-prefix family mapping on the already-frozen labels and counted raw errors.

| Quantity | Observed |
|---|---:|
| Plain OOF balanced accuracy | 0.546761123540 |
| Centered OOF balanced accuracy | 0.547849863286 |
| Plain wrong rows | 1,223 |
| Wrong **within same provider/model family** | **701 / 1,223 = 57.32%** |
| Wrong across provider/model family | 522 / 1,223 = 42.68% |
| Plain vs centered prediction disagreements | 1,134 / 2,699 |
| Centered rescues plain errors | 353 |
| Centered breaks plain correct predictions | 350 |

Examples of the **descriptive**, already-consumed OOF confusion distribution: Gemini 3.7 Flash -> Gemini 3.8 Flash: 50; GPT 5.6 Sol -> GPT 5.6 Terra: 40; Gemini 3.8 Flash -> Gemini 3.7 Flash: 37; GLM 5.3 -> GLM 5.3 Flash: 35. Within-family errors out of total wrong rows by true family: Google 223/238; OpenAI 250/340; Anthropic 173/292; ZAI 55/229. These are *not* preselected fresh evaluation slices and must **not** become threshold/label-specific tuning targets.

These are observations from repeatedly seen official OOF rows, **not independent validation or evidence a new method scores higher**. They indicate an already learned family signal with a substantial remaining **sibling-model discrimination** problem; they do not prove that the same error mixture holds on hidden test.

## Decision: pursue sibling-model discrimination *as a research question*, not a new leaderboard submission

### Global hypothesis

The current 8B adapted representation already captures a useful family-level stylistic signal. Much residual error is confusion between close model siblings, often under new prompts. An improved attribution representation should discriminate **within-family model identity** while preserving source/prompt transfer. This would be a true capability change rather than a class-prior correction or a copied ensemble.

### What is rejected now

1. Re-running published Cosmorin V1/V3 or blindly repairing 14B hidden exceptions.
2. Using prompt clustering plus Hungarian one-to-one assignment as if hidden-test prompt structure were guaranteed.
3. Toggling plain/centered head, adjusting per-label biases, or tuning blend weights from these already-consumed 2,699 OOF rows. Centered's 353 rescues are almost cancelled by 350 regressions; it also depends on privileged same-prompt centering unavailable as hidden-test metadata.
4. Reopening `prompt-residual-style-v1` LDA, which failed cross-source transfer on 11/13 sources (mean delta -0.03579), or claiming frozen DCT v3 has any score (its scientific paired transfer gate is still unrun).
5. Treating another grouped split, the current Public LB or a synthetic CPU smoke as fresh independent scientific evidence.

### Smallest useful **no-training / no-submission** next check

Use the already-existing, provenance-audited held-out **external source** artifacts (if predictions/embeddings are available without running GPU models), and test whether within-family residual error and plain-vs-centered disagreement replicate **by source and prompt bank**, with labels/family mapping fixed in advance. Define the data-availability/integrity gate first; do not download/model-infer on new GPU. If the required frozen held-out predictions are not already available, mark this analysis **UNAVAILABLE**, do not fabricate it.

Only if the mechanism survives an actual disjoint transfer check should a separate, preregistered model-family hypothesis and test plan be considered. A fundamentally stronger family-aware representation would generally require fitting/adapting a model; the user's current preference is **no self-training**, so that implementation is **not authorized** by this note.

### Reopening/stop triggers

- **Stop** the no-training audit if archived held-out predictions are missing or provide no transfer-relevant contrast. Do not start model computation simply to fill an evidence gap.
- **Revisit** the representation class only with a new independent source-transfer prediction and frozen admission rule.
- **No new official Kaggle submission** without a new viable inference candidate, full hidden-sized smoke under the same constrained runtime, and explicit go-ahead.

## Completed no-training source-evidence availability check

**UNAVAILABLE — stop at this boundary, no additional compute.** Checked the original Gate-A recovery workflow `41333631967`. Its retained external-transfer evidence artifact **13094213203** (downloaded as a read-only ZIP) contains exactly one JSON report, `two_stage_lora24_gate_a.json`, with A0/A1 scores, per-source aggregate metrics and the predefined PASS verdict. It does **not** contain individual held-out passages with their predicted and true labels, logit vectors, or embeddings. The separate artifact **13094069155** is a 116 MB Stage-1 A1 checkpoint, not a frozen held-out prediction table.

Therefore existing source-level aggregate BA proves **A1 > A0** on the held-out sources, but **cannot** test whether 57.32% within-family confusion repeats there. No data should be guessed, and a new GPU forward pass or model training is not authorized by this read-only check.

The useful result is therefore a *diagnosis of an observed OOF bottleneck* with a clearly identified independent-evidence gap, not a new leaderboard candidate. The next scientific action, if separately authorized, would be a tightly specified **inference-only** source-heldout prediction export (not retraining) and an exact same-family-error-rate comparison with the existing 701/1,223 observation; freeze the source subset and failure thresholds first. Until such evidence exists, no ensemble tuning or Kaggle submission is justified.

## Evidence distinction

- **FACT:** All counts above were computed from the archived original OOF CSV; scores and submission references are recorded in canonical repo artifacts.
- **INFERENCE:** Sibling discrimination is likely the main useful research bottleneck, though its hidden prevalence is unknown.
- **UNKNOWN:** Whether a no-training inference transformation can improve those mistakes, whether external source predictions for this exact final candidate are archived, and whether any next candidate would beat 0.520 on hidden test.

Original raw evidence remains in GitHub Actions. This report contains only aggregates and plan, **no model fitting, Kaggle submissions, new GPU runs, raw texts or per-row labels**.
