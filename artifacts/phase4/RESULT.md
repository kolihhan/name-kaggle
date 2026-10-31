# Phase 4 — Official-only trainable ModernBERT result

Date: 2026-10-31

## Result

The official-only end-to-end ModernBERT baseline completed all five permanent prompt-grouped folds on Kaggle GPU.

| Model / representation | OOF balanced accuracy |
|---|---:|
| Phase 1 char TF-IDF + LinearSVC | **0.31930** |
| Phase 3 frozen ModernBERT + LinearSVC | **0.30224** |
| Phase 4 trainable ModernBERT | **0.25781** |

Phase 4 fold scores:

- fold 0: 0.22778
- fold 1: 0.24444
- fold 2: 0.29259
- fold 3: 0.26296
- fold 4: 0.26124

## Verification

The run passed the experiment contract before its result was accepted:

- official `train.csv` SHA-256 matched `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`;
- fold fingerprint matched `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`;
- 2,699 official rows were scored;
- 150 prompts and 18 labels were present;
- external rows were exactly 0;
- all five folds completed;
- no NaN, OOM, runtime failure, or single-class prediction collapse was observed.

The canonical GitHub Actions artifact for the run has digest:
`sha256:915f4239e782166e43a3a775ee9d29959ef19da0f7d6e5e9e693098ebd8ac9ab`.
It contains the full OOF CSV and Kaggle run log.

## Interpretation

**Reject this full-fine-tuning configuration as the next modeling path.**

The result is not evidence that the validation pipeline is broken. The same immutable official data hash and fold fingerprint were enforced, and the run completed cleanly. Instead, end-to-end fine-tuning with this tiny 2.7k-row dataset and the frozen Phase 4 configuration appears to destroy useful pretrained representation for several classes.

Large recall drops relative to frozen ModernBERT include:

- `gemini-3.1-pro-preview`: about -0.253
- `gpt-6-astra`: about -0.233
- `claude-opus-5`: about -0.227
- `gpt-5.6-luna`: about -0.220

Some classes improve, so this is not a trivial classifier collapse. It is more consistent with optimization / catastrophic-forgetting mismatch on a very small supervised set.

## Decision

Do **not** respond by blindly increasing epochs or starting a broad ModernBERT hyperparameter sweep.

The cheapest visible gap is now the classical baseline itself: our Phase 1 baseline uses mainly character n-grams, while stronger public honest prompt-grouped baselines combine character n-grams, word n-grams, and numeric style features and report CV around 0.37.

Next experiment: build an **official-only honest style baseline** under the exact same frozen folds. Only after closing that cheap baseline gap should we spend more GPU budget on LoRA / larger-model attribution approaches.
