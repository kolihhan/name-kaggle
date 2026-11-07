# Short-TRACE 8B direct Kaggle code-competition submission — frozen v0

Date: 2026-11-07. **Status: frozen before any official submission score.**

## User scope change
User explicitly requested **skip separate GPU pilot, instead submit a Kaggle Notebook for official hidden rerun**. There is no local/Colab/GitHub Actions GPU model execution in this path. Kaggle official hidden rerun still needs a Kaggle accelerator for 8B inference; the user's request does not change the mathematical compute requirement. Only one official code-competition submission is allowed for this candidate. No Public-LB tuning.

This v0 **does not overwrite the scientific v1 pilot contract**. The original external-source H/S/T pilot stopped at a missing 4-bit dependency and has no scores. The direct path is an exploratory competition-facing implementation, **not a claimed scientifically promoted H/S/T mechanism** under the established EVALUATION_POLICY.md. Thus even a good leaderboard score cannot by itself prove true transitions beat shuffled controls.

## Fixed model, data and arms
- Evaluator **Qwen/Qwen3-8B-Base**, pinned Kaggle Model `qwen-lm/qwen-3/transformers/8b-base/1`. Frozen neural weights. **NF4 4-bit**, fp16 compute, no inference API, no model fine-tuning.
- Dependency: public pinned bitsandbytes 0.50.2 wheel from PyPI SHA256 `55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283`, Kaggle Dataset `kolihhan/ntl-short-trace-offline-bnb-0502`; no network in Notebook.
- Official `name-that-llm` 2,699 labeled rows, 150 prompt groups, 18 classes. **No external training data** in this v0. Official hidden test only used at inference. Nothing published from official train/test into GitHub.
- Rank features from `scripts/short_trace_core.py`: max 384 evaluator tokens, nine fixed rank bins, same frozen deterministic shuffled-null seed 20261107.
- H = 9-dim marginal histogram; S = H + shuffled-transition 81-dim; T = H + real-transition 81-dim. H/S are **diagnostic controls only**. Single submitted prediction rule: `T` + `StandardScaler` + `LogisticRegression(C=1,class_weight=balanced,max_iter=2000,random_state=20261107)`. This final readout is trained on the official 2,699 examples, with no official validation rows included in any fold's train set for OOF diagnostics.
- Print 5-fold `GroupKFold` official-train OOF BA on H, S, T to logs; no tuning or gating from these historical official folds. These are not the previously frozen permanent fold map and are not valid new fresh independent transfer evidence. Data are grouped by `prompt_id`, every group stays inside one fold. All metrics are macro per class.
- Full hidden test: expect ~4,000 rows, output strictly two columns `id,model`, unique non-null IDs and permitted labels; save `/kaggle/working/submission.parquet`.
- Only preview (1–3 Kaggle fixture rows, **not official competition rerun**) emits a schema-only dummy valid-label file, explicitly flagged, to avoid burning a standalone GPU training/inference session on the fixture. Hidden rerun must execute the full Short-TRACE evaluator, no dummy fallback.
- Existing 9h official runtime limit applies. **UNKNOWN** until hidden rerun: 8B runtime/memory suitability and degree of source shift.

## Preflight and stop
1. Vendored source compiles; static tests of rank bins, shuffling and builder pass.
2. Byte-pinned public wheel exists, runner pins wheel/source; Kaggle private internet-disabled competition Notebook metadata includes `competition_sources: ['name-that-llm']`, 8B model input and offline runtime input.
3. Only after exact Notebook version push with no immediate infrastructure error: attempt **one** `kaggle competitions submit name-that-llm -k <actual slug> -v <version> -f submission.parquet -m 'Short-TRACE 8B T no local GPU pilot'`. Never submit original research-only v1 kernel: it cannot output submission.parquet.
4. Log Kaggle submission ref, then inspect actual scoring status when available. If the kernel fails from GPU quota, OOM, PyPI wheel incompatibility, timeout, or runtime API, record as infrastructure/runtime failure, **not invalidation of the TRACE representation**.
5. Do not tweak thresholds, remove the control, tune model/weights based on the score, or silently substitute a stronger prefit baseline if the result is poor.
