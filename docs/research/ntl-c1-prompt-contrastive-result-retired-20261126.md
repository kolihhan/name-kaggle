# NTL C1 prompt-matched contrastive — executed result and decision (2026-11-26)

**Status: RETIRED for this exact frozen E5 representation and loss protocol.** Not a Kaggle competition submission or new Public score. Champion stays **0.718**, official submission **#58755304**.

## Exact executed evidence

Authenticated [GitHub Actions run #45759015821](https://github.com/kolihhan/name-kaggle/actions/runs/45759015821): **SUCCESS**, all four new behavior tests passed, frozen 335M E5 embedding generated for the full official training data (**2,699 texts, 1,024-D**), and 3-fold prompt-group out-of-fold model fitting completed. Kaggle official TRAIN: **18 classes / 150 prompt groups**, no external provider-generated data, no GPU or official competition submission.

Preregistered protocol: [C1 contract](ntl-c1-prompt-contrastive-prereg-20261126.md). Same projector, labels, train-prompt groups, initialization, optimizer, batch order and 8 epochs. One changed mechanism: ordinary supervised CE vs CE plus 0.2 prompt-matched InfoNCE contrastive term using same-generator different-prompt positives and same-prompt different-generator hard negatives.

| Metric | CE-only | C1 contrastive |
|---|---:|---:|
| Official-train 18-label grouped OOF BA (DEV) | **0.19151379567486948** | **0.18299776286353467** |
| Fold 0 BA (900 held-out rows) | 0.1822222 | 0.1855556 |
| Fold 1 BA (900 held-out rows) | 0.1922222 | 0.1877778 |
| Fold 2 BA (899 held-out rows) | 0.2001814 | 0.1757143 |

- Paired OOF delta **−0.008516032811334812** absolute BA.
- Same-row C1 rescues **192**, regressions **215**; net **−23** correct.
- Prompt-cluster bootstrap 95% confidence interval for BA delta: **[−0.023340852597564, +0.0055558662689535535]**, crosses zero.
- Predeclared advance gate **FALSE**: contrastive BA <0.500, gain <+0.025, bootstrap lower bound <0, rescues ≤ regressions.
- Total encoder + OOF computation **1858.4 seconds (~31 minutes)**, CPU only.
- Encoder identity and embedding checksum: `intfloat/e5-large-v2`, revision `d94960066016d2064f2a3a2dbc65107a2b48230d`; embedding SHA256 `a5ec2ee659f553d19e1effecae0325e3c89bbd79bfaf81bb0a9d5edbc70d9b9d`.

Some C1 class recalls exceeded CE while others dropped: e.g. Gemini 3.8 +0.053, GPT Luna +0.087, GPT 6 Astra +0.093; Gemini 3.6 −0.153, GLM Flash −0.113, GPT 5.5 −0.087. No global improvement. Don't select specialist gates based on these inspected DEV slices.

## Decision and interpretation

**RETIRE this exact frozen-embedding supervised contrastive C1 objective**, no alpha/temperature/epoch/seed/model-scale sweep on reused official TRAIN, and do not promote to 4B LoRA or submit a Kaggle notebook. The evidence refutes the narrow idea that the fixed E5 representation benefits from this particular loss, not every contrastive learning method or every larger trainable model.

The absolute 0.183 OOF score is an OFFICIAL-TRAIN grouped-DEV estimate for a very different frozen E5 linear readout and **must not be compared directly with the incumbent Kaggle Public 0.718** as if both were same-distribution scorers. The 0.718 ensemble likely uses different pretrained source data, models and task-specific evidence. No fresh 18-label same-row incumbent comparison was generated. Submission formatting preflight passing is engineering success, not modeling success.

## Next discriminating research lane (not yet executed)

Find **independent generator-specific token likelihood** or pretrained model-identity features *not already covered* by current 0.718 strong ensemble. Read the precise v36 `infer.py` and stage provenance first; compare against source-locked independent examples where all 18 true labels and serving versions are credibly documented. Explicitly distinguish likelihood evidence from request-protocol/format artifacts, record raw scorer cost, and predeclare an unchanged incumbent same-row control. If no trustworthy independent panel/checkpoint/runtime route exists, report the blocker; do not submit blindly. A new model class requires a new preregistered experiment, not a relabeled C1 phase.

**No Kaggle GPU, no Kaggle Notebook push, no competition submission performed for C1.**
