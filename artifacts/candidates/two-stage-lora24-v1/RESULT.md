# two-stage-lora24-v1 — Gate A Result

Date: 2026-11-02 (MYT)

## Verdict

**PASS**

The frozen external-only Gate A passed all predeclared decision rules. LoRA adaptation (A1) beat the frozen-backbone head-only control (A0) on the pooled held-out metric and on all three held-out public sources.

This result authorizes progression to a separately planned Gate B. It does **not** itself execute Gate B, official OOF, Stage 2, or any Kaggle competition submission.

## Scientific execution lineage

- Candidate: `two-stage-lora24-v1`
- Frozen scientific trigger commit: `6a7168e166a719889a7f6ad77ed818a5dfee75d7`
- Original GitHub Actions run: `41309458267`
- Kaggle kernel: `kolihhan/name-that-llm-two-stage-lora24-gate-a`
- Scientific Kaggle execution count: **1**

The original GitHub Actions poller exhausted its `540 × 30 s` polling window while Kaggle still reported `KernelWorkerStatus.RUNNING`. That CI failure was an orchestration timeout, not a scientific Gate A verdict. The failed workflow never reached output download or metric verification.

No failed-job rerun was used because that path would execute `kaggle kernels push` again and create a second scientific run.

An attach-only recovery path was added under TDD:

- RED commit: `19f5cf85a598a1d7ccf5dd6ff39064f19e4c434a`
  - expected result: `1 failed, 19 passed`
  - sole failure: recovery path absent
- GREEN/recovery commit: `62b015a90c6b44bf880e6802d43b162fdca77ce7`
- Recovery GitHub Actions run: `41333631967`

The recovery job contained no `kaggle kernels push`. It attached to the existing kernel, observed `KernelWorkerStatus.COMPLETE`, downloaded that existing output, verified it, and preserved the exact A1 checkpoint from the original scientific execution.

## Frozen data and eligibility

- Public source: `nishantdahal/ntl-handwriting-blend`, `pool_v4.parquet`
- Raw SHA256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`
- Raw rows: `75,783`
- Exact Phase14 subset: `43,781` rows / `3,736` prompt IDs / `18` labels
- Training rows after held-out leakage removal: `30,133`
- Held-out sources: `abdullah`, `zakir`, `semenoff`
- Eligible held-out labels: `13`
- Eligible held-out rows: `3,324`
- `prompt_id` overlap: `0`
- `prompt_key` overlap: `0`

Eligibility therefore exceeded the frozen minimum of 8 labels and 300 held-out rows.

## Gate A metrics

| Metric | A0 | A1 | A1 − A0 |
|---|---:|---:|---:|
| Pooled balanced accuracy | `0.3942430082` | `0.4657129266` | `+0.0714699184` |

Per-source balanced-accuracy deltas:

| Source | A0 | A1 | Delta |
|---|---:|---:|---:|
| `abdullah` | `0.5080051721` | `0.5886589423` | `+0.0806537702` |
| `semenoff` | `0.3453408161` | `0.4117698040` | `+0.0664289879` |
| `zakir` | `0.4677680284` | `0.5758311807` | `+0.1080631523` |

- Mean source delta: `+0.0850486368`
- Negative-source count: `0 / 3`
- Frozen decision: `pass`
- Decision reason: `all_frozen_gate_a_rules_passed`
- `official_oof_authorized`: `true`

## Runtime and implementation invariants

The recovered JSON independently records the frozen runtime behavior for both A0 and A1:

- `exact_one_epoch = true`
- `triplet_padding = false`
- `stock_qwen_final_norm_retained = true`
- A0 optimizer steps: `1,857`
- A1 optimizer steps: `1,857`
- A0 triplet microbatches: `11,142`
- A1 triplet microbatches: `11,142`
- skipped scaler steps: `0` for both conditions

Pinned model/runtime:

- `Qwen/Qwen3-8B-Base`
- revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- `transformers==4.57.1`
- `peft==0.17.1`

## Leakage / competition boundaries

The verified result records:

- `official_rows_scored = 0`
- `official_oof_runs = 0`
- `competition_test_rows_used = 0`
- `public_lb_tuning = false`
- `kaggle_submission_authorized = false`

No Phase19 shadow fold was reused and no alternate source holdout was minted.

## Preserved evidence

Recovery run `41333631967` produced:

1. Gate A evidence artifact
   - artifact ID: `13094213203`
   - name: `two-stage-lora24-gate-a-evidence`
   - ZIP digest: `sha256:9374e77a0eb59956484dc9bb052fcc55ccea4d95e14e62976ba23c4ba97fcf8e`

2. Exact A1 checkpoint artifact
   - artifact ID: `13094069155`
   - name: `two-stage-lora24-gate-a-a1-checkpoint`
   - size: `116273873` bytes
   - ZIP digest: `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`

The checkpoint contains the LoRA adapter, `heads.safetensors`, `meta.json`, and tokenizer state. Gate B, if implemented, must initialize from this exact artifact lineage rather than a retrained equivalent.

## Interpretation

Gate A provides positive evidence for the narrow hypothesis it was designed to test: on unseen whole public sources, the frozen LoRA adaptation improves attribution-style transfer over the same backbone with only trainable normalization/readout components.

It does **not** establish hidden-Kaggle transfer, does not supersede the Phase 7 competition champion, and does not validate official-data Stage 2. Those claims remain untested.

## Next authorized work

Create a separate Gate-B implementation plan under the frozen evaluation policy. Gate B may use only the exact preserved A1 checkpoint above as its Stage-1 initialization. No automatic Gate-B execution or Kaggle competition submission is authorized by this result.
