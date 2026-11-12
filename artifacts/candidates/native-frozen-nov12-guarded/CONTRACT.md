# Native pretrained 18-class NTL — frozen, guarded one-shot reproduction

Date: 2026-11-12. Explicit user directive: produce a new official submission for **both** NTL and Gemma 4. This is a score-seeking external-source reproduction, not our original model method and **not** a claimed improvement until our Kaggle ledger scores it.

## Candidate / provenance

- Source: [public native-ensemble Notebook](https://www.kaggle.com/code/shakshamkarkiii/ntl-native-ensemble).
- Exact original `.ipynb` SHA-256: `2b04b4d7ccdeeaef6ed62a52e24ee6f6f1702a50143c740a6d0a3bbce90e6920`.
- Exact original normalized code SHA-256: `5c8c2b63b3ae61b00bc50e74761723b81f7c23b0fc2808ada3310408f3a59a16`.
- Original Notebook: COMPLETE, publicly viewable as of this date, with 21 Dataset, 8 Kernel and 6 pretrained Model sources. No source code or competition data copied into the GitHub repo; downloaded source is ephemeral inside GitHub Action.
- Existing own verified champion: Public BA **0.679**, Kaggle submission `58210835`. Earlier inference-only Lightning entry `58167751` remains PENDING; this is a different public Notebook lineage, not a retry of that submission.
- Source score: **unverified**. The experiment uses no assumed upstream Kaggle score; a title, popularity or source author is not proof of a particular public BA.

## Hard constraints and frozen intervention

This is **inference only**, preserving published frozen pretrained inputs; no new neural, LoRA, Gaussian or supervised readout training. Original source has exactly 4 fit call sites: three unlabeled TF-IDF transforms and one supervised `gauss.fit()` fallback. The fork guards the latter to raise on execution. It also blocks `AtlasModel(training=True)` and `core_worker --train-final`. Unlabeled test text grouping and TF-IDF vectorization are allowed. If any guard is triggered, the Notebook must fail rather than retrain. Do not weaken guards or fall back to lower-quality unverified training.

Exact single change to public Notebook: add attributable credit markdown and the three fail-closed no-training guards. No transductive weight tuning, label adjustment, manual test labeling, new train/test split, source-level leakage or Kaggle Public LB adaptation.

- Own Kaggle Notebook slug: `kolihhan/ntl-native-frozen-nov12-one-shot` (private).
- Own official submission message: `NTL frozen native pretrained ensemble guarded no supervised fitting 20261112`.
- One version-1 Notebook push only; wait for real private preview COMPLETE and `submission.parquet` before submitting it once.
- Must verify current official NTL daily slot and that exact message and own slug do not exist already. If source SHA, input topology, image/runtime, guard count, preview, quota or receipt is ambiguous, stop. Never blindly retry a submission call.
- No new GPU training. Kaggle will execute inference for preview/scoring with its own allowed T4 runtime. One contest submission, not multiple.

## Evaluation and decision

Primary endpoint: actual Kaggle terminal Public balanced accuracy, compared against 0.679. Pending, error or unscored submissions **do not** count as success. The algorithm's scientific originality and independent 18-label transfer remain **unproven**: no new clean same-row independent holdout was available as of 2026-11-12. Even a Public gain would not establish improvement on the Private leaderboard.

This is a user-authorized score attempt despite missing independent transfer validation, not retrospective evidence. No extra source/data leak in GitHub logs. Original third-party license not assumed beyond Kaggle's public Notebook fork capability; retain original credits and do not distribute raw source/weights in repository.
