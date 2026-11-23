# NTL Independent Lane — Matched-Prompt Conditional Discrimination (MPCD-v1)

Date: 2026-11-23. **Frozen before any current-lane score.**
Branch: `research/ntl-matched-conditional-loss-v1-20261123`.
Type: CPU mechanism study; **not** a submission or a new hidden-accuracy estimate.

## Goal and honest prior art

Final objective: 18-class `Name That LLM` official hidden Balanced Accuracy, highest authenticated reference 0.715. **Candidate does not import or use 0.715's code, weights, labels, output or Kaggle logits.**

Historical counterevidence:
- Previously rejected **prompt-residual-style-v1** (frozen ModernBERT+LDA of prompt residuals) lost 0.03579 average source BA across 13 external sources, in 11/13 source slices.
- Previously rejected **prompt-invariant-contrastive-v2** (Qwen LoRA + cross-prompt prototype CE) won only +0.002685 BA with 95% CI crossing zero on 11-label source transfer.
- Previous LoRA v1 already applied plain+centered CE, so **supervised prompt-conditioned training is prior art in the repo**. This experiment is a *different scoring loss and fully independent single-response readout*, not a claim of a novel algorithm / SOTA.
- Blind batch Hungarian treatment officially scored **0.594**, down 0.121 from reproduced external ensemble 0.715, and is not involved here.
- The official 2699 rows and permanent 5-fold `prompt_id` manifest are repeatedly used historical DEV, NOT independent confirmation. No uncontaminated 18-class external benchmark has been certified.

## Two competing mechanistic hypotheses

**H1 (matched comparative identification):** Learning a score function whose correct model must identify its text *among other responses to exactly the same prompt* improves single-response generalization, because shared topic distractors cancel inside a conditional loss during fitting. It may work even without inferring test-time groups.

**H2 (generic regularization / source fingerprint):** Additional conditional-softmax loss simply regularizes an already underfit low-dimensional readout, or reinforces dataset/creator artifacts, rather than learning truly paired content controls. Prediction: matching *fake* prompt groups made by independently shuffling prompt IDs within each model should yield the same benefit as genuine paired prompts.

### Distinguishing observation

C1 should beat BOTH C0 and C2, with real prompt training and **without using prompt IDs at inference**. If C1≈C2, H1 is not independently supported. If all methods stay under strong sparse 0.3671, this linear feature class is insufficient as a competitive candidate even if the conditional objective helps.

## Frozen experiment

Data: `data/train.csv` 2699 rows, 18 true models, 150 true prompt groups, and the existing immutable five-fold manifest `artifacts/phase1/folds.csv`; existing `load_snapshot` verifies hashes and train/validation prompt separation. No external/fresh data, no test.csv, no hidden labels.

Feature extractor, fitted ONLY on each fold's TRAIN:
- word-agnostic character TF-IDF ngrams 2–5, max 45,000, min_df=2, raw uppercase preserved;
- fixed 128-dimensional TruncatedSVD on sparse features, train-only, L2 normalized;
- established 155-style-feature `extract_style_features`, train-only StandardScaler, clip [-4,4], row-L2 normalized and multiplied by 0.30;
- concatenate these features, no label-dependent feature selection or inference-time Prompt group assignment.

Same multinomial linear softmax readout W/b, same starting zeros, same C=1.5 quadratic regularization, L-BFGS-B max 65 iterations, max 2 CPU threads and same fixed optimizer tolerances for all arms.

Arms:
- **C0** row-wise 18-class cross-entropy.
- **C1** C0 + `0.5 * conditional_column_cross_entropy`, where each training true-source label is compared against responses to its **genuine same-prompt panel**. Each held-out prediction is `argmax(X_test W + b)`, with NO test prompt grouping.
- **C2** same C1 objective and all hyperparameters, but shuffle training **prompt IDs independently within each true class** with frozen fold-specific seed. This preserves group sizes and exactly one observation per model per prompt (barring one missing row), but destroys matched topics. C2 tests whether pairing matters, not whether an extra CE term helps.

No hyperparameter or model sweep. No 0.715 evidence used during fitting. A single historical 5-fold GroupKFold OOF diagnostic, class-wise recalls, paired rescue/regressions, variance and wallclock. Neither C1 nor C2 is evaluated on train/fold contamination.

Primary metric: all 2699 **true** labelled DEV rows, balanced accuracy across all 18 classes, paired C1−C0 and C1−C2. Also per-fold BA, rescue/regression, 1000-draw by-prompt bootstrap 95% CI, time and projection sanity. The script includes analytic gradient numerical finite-difference tests and fake-panel integrity checks.

## Stop / go

To justify deeper external-source study (NOT direct submission), demand ALL:
1. C1 − C0 pooled 18-way BA >= 0.02;
2. C1 − C2 pooled BA >= 0.01, ruling out generic regularization;
3. both prompt-bootstrap CI lower bounds > 0;
4. C1 wins at least 4/5 fold-specific BA vs C0;
5. no class recall drops >0.15 compared with C0;
6. no evidence leakage / runtime failure.

Even if all pass: **only historical DEV mechanism signal**, cannot claim a hidden-competition uplift or skip the fresh source-disjoint transfer gate. If not, do not tune weight, SVD dimensions, ngram, regularization, seeds or folds using this outcome; retire the specific objective/representation setting and reconsider source/publisher nuisance with independently certified full-18 matched data.

## Cost and restrictions

GitHub Actions CPU 2 cores, timeout 24 minutes, zero GPU, no official Kaggle training/job, competition submit, hidden data, file mutation outside branch, original Champion change or external model weights. It fits a lightweight CPU-only linear classifier, not a neural LLM. Trial remains research—not a claim of generative model fingerprint SOTA.
