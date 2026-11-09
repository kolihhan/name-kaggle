# Public Lightning 0.681 — frozen inference reproduction, no neural or supervised fitting
Date: 2026-11-09, before any new own Kaggle submission/score.

## Goal and truth status
Reproduce, ONCE, the already-public Kaggle competition Notebook
`lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b`, published under the title
`0.681 | Qwen3-Embedding LoRA + Qwen3.8-27B`, in the user's own Kaggle
account. This is **reproduction of external work**, not original research, nor
an independently demonstrated improvement; public score 0.681 is the
external author's result and our own confirmed best remains 0.668
(submission 58045266/58022340) until a new official result is verified.
No LB-conditioned tuning, ensembles, threshold changes or weight search.

## Frozen source and dependencies
Use Kaggle CLI to pull exactly the above publicly listed original Notebook
and preserve its cells, all 21 Kaggle Dataset attachments and six Kaggle Model
attachments. Freeze and log SHA256 of the upstream Notebook BEFORE uploading
one own version; fail if the source identity or attached inputs change.
Retain all author/source attribution and license notices; add a single noncode
credit cell. All pretrained data are referenced by their existing public
Kaggle slugs and are never embedded into Git.

The audited source has `CONFIG['RELEASE_READY']==True`, 6 models, and
`trained_artifact_contract` confirming prefit SVC, Q14/Q8 Gaussian parameter
files and pretrained neural adapter/model weights
(`lightningv08/ntl-olmo2-native24-final-inference-assets/16`).
The main `ensemble.py` calls `core_worker.py` WITHOUT `--train-final`.
Static audit Actions: 42609149437, 42609474556, 42609692279.
These are source-code checks, not an official score.

## Hard no-training gate
User said **NO MORE TRAINING TO SUBMIT**. No new GPU training, LoRA fit,
optimizer backward/step, source training, model calibration from labeled
official rows, or Gaussian supervised `gauss.fit`.

The public code defines unused training paths. When creating our
reproduction, inject fail-closed runtime guards into the `gauss.fit`
source-consumer module and the `if training:` neural model branch.
If either becomes active in preview/hidden rerun, the Notebook MUST fail
rather than train. An existing output must never be relabeled as new evidence.
No change to active inference outputs is intended.

**Scope distinction**: the original inference pipeline uses unsupervised
TF-IDF `fit_transform` on the *unlabeled query texts* for dynamic
prompt/topic clustering. This transforms query context without fitting to
official labels or updating neural weights. It is permitted here solely as
inference-time feature preprocessing, not supervised model training. No CPU
or GPU model-fitting experiments are authorized.

## Kaggle one-shot
- Use exact original public 0.681 Notebook Python source and metadata input
  slugs; change only own Kaggle kernel ID/title, private visibility, attribution
  and the two no-supervised-training fail-closed guards.
- Internet disabled, original competition source, T4 machine shape, original
  public prefit model/dataset sources unchanged.
- Require original public kernel status COMPLETE and original user Kaggle
  competition ledger to have no same-description submission.
- Static compile/document/guard checks first.
- Push only one new private kernel version and wait for preview status COMPLETE
  plus `submission.parquet` output via paginated files check.
- Submit exactly this kernel version once (Code Competition) with description
  `Frozen public Lightning 0.681 reproduction no new supervised training 20261109`.
- Verify accepted Submission ID and official ledger/score. No resubmit/retune
  on public score. Kaggle hidden rerun is an inference job, not an authorized
  new model training job.
- If runtime path actually needs supervised fit, errors, license/asset access,
  GPU quota or 9h limit blocks success, **STOP**. Do not relax guards or train
  to rescue this reproduction; record verified failure.

## Evaluation
Compared only against historical own official Public Best 0.668, and
publicly claimed upstream 0.681. No causal or generalization conclusion from
these different Public-LB entries. Old independent CDA transfer failed.
Preserve all historical negative outcomes and the existing EVALUATION_POLICY.
