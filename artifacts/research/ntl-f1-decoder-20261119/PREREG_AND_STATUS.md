# NTL F1 — frozen decoder-layer attribution experiment contract
2026-11-19, prospective to any **actual 1.7B pretrained decoder feature extraction** or C1/C2 experimental score.

## Scientific hypothesis

- H1: intermediate hidden states of a **frozen autoregressive 1.7B decoder** encode more transferable generator-identity signal than sparse character patterns, on novel prompts.
- H2: the apparent decoder representation is dominated by prompt/topic or source artifacts; removing prompt-nuisance directions from train-only prompt means will erase or destabilize apparent gains.

This test is NOT a comparison against the third-party 0.702 champion's unavailable honest OOF. It is allowed to start without a new independent 18-class data source. Official 2,699 rows / 150 prompts / 18 models remain **adaptively spent DEV-only** evidence.

## Fixed data and model

- Official TRAIN SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`; official fold manifest SHA256 `94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d`.
- Frozen five 30-prompt-group folds from `artifacts/phase1/folds.csv`. No post-result seed sweep, shuffled folds, test-prompt access, or fictitious new holdout.
- Frozen model **Qwen/Qwen3-1.7B-Base**, Apache-2.0 per Hugging Face model-card metadata, pinned commit `658b6ee46470b73f3e0e04e57fc9fb7031fd7daa`. This model is **a feature encoder only**, NOT a replica of competition generator `qwen3.8-max`.
- Colab T4, fp16, no decoder fine-tuning; max 512 tokens, attention-masked mean pool at ¼, ½, ¾ and final hidden layer (four concatenated embeddings); preserve row IDs in a **private ephemeral Colab NPZ**, never publish features, test data or raw competition passages.
- C0: fold-training-only char_wb 3–5 TF-IDF (min_df2, max_features100k), class-balanced multinomial sklearn LR C=1.
- C1: fold-training-only StandardScaler on frozen decoder vectors, same class-balanced LR C=1, **no prompt projection**.
- C2: same C1, but per-training-fold PCA k=8 on the mean feature vectors of training prompt groups, project out exactly those eight directions. Heldout prompt IDs/text never inform fitted mean/PCA/feature scaler.
- Fixed initial compare C1−C0, then C2−C1. NO k=32, no second backbone, no strength/weight/threshold tuning on spent folds.

## Mandatory measurements

18-class macro BA, all 18 per-label recalls, cross-family vs sibling wrong counts, rescues and regressions, 5-fold totals, paired prompt-cluster bootstrap 5,000 resamples with CI 95%. A fold-local stability test prevents data leakage. No manufactured incumbent OOF.

- **Strong signal to continue to a separate source-transfer check**: C1−C0 macro BA ≥+0.08, paired cluster CI lower ≥+0.04, and at least 12 of 18 labels suffer no >.02 recall decrease. This is **not a Kaggle submission promotion by itself**, even if met; historical LoRA24 OOF 0.546761 is a different/partially source-exposed control.
- **Kill v0**: C1−C0 Δ≤+.03 OR paired CI upper ≤0. No GPU reruns or seed sweeps to rescue.
- Between these outcomes: mark weak/inconclusive. Compare projection effect C2−C1 separately; do not selectively promote the better arm post hoc.
- **Model release gate**: Only if the mechanism is strong, separately verify source-disjoint transfer and 9h offline scoring feasibility before any one-shot official submission; verify Kaggle account quota; keep 0.702 (#58514818) immutable.

## Already executed before F1 decoder extraction

- [Real C0 and CPU contract #44541223988](https://github.com/kolihhan/name-kaggle/actions/runs/44541223988): **SUCCESS**, 2,699 rows, 150 prompts, 5 folds 540+540+540+540+539, `C0_char_lr = 0.291509` macro BA (DEV only). No decoder weights/embedding data/inference in this CPU run.
- [Real Qwen3Model tiny random-weight API + repeated actual C0 #44541786043](https://github.com/kolihhan/name-kaggle/actions/runs/44541786043): **SUCCESS**, 4-layer output shape [2,512] for tiny 8-layer model. **This tests the architecture API with random weights, not pretrained generator attribution quality.** Masked-pool and fold-local PCA tests PASS. Reproduced C0=0.291509.
- GPU extractor: `scripts/ntl_decoder_probe_extract_colab_20261119.py`.
- Paired evaluation: `scripts/ntl_decoder_probe_eval_cpu_20261119.py`.
- One-click Colab: [`notebooks/ntl_f1_frozen_qwen3_decoder_probe_colab_20261119.ipynb`](../../notebooks/ntl_f1_frozen_qwen3_decoder_probe_colab_20261119.ipynb).

## Actual pending scientific result

**C1 and C2 are NOT RUN, have NO BA, and are not promoted.** The current ChatGPT tooling has no remotely connected Colab T4 execution capability, and the owner's approved compute scope does not authorize paid Hugging Face GPUs or random Kaggle development GPUs. The owner can open the Colab notebook, choose T4, Run all; it downloads an **aggregate-only JSON** for review and does not submit. Do not call this a confirmed model-quality gain.
