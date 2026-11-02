# Gate B Checkpoint Lock — two-stage-lora24-v1

Date: 2026-11-02 (MYT)

Status: **LOCKED FOR GATE B IMPLEMENTATION**

This file freezes the exact Stage-1 A1 checkpoint transport that any Gate B implementation must consume. It records transport lineage only; it does **not** authorize or report Gate B Stage-2 training or official OOF.

## Source checkpoint

- Candidate: `two-stage-lora24-v1`
- Gate-A source Actions run: `41333631967`
- Source artifact ID: `13094069155`
- Source artifact name: `two-stage-lora24-gate-a-a1-checkpoint`
- Source artifact digest: `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`
- Exact model: `Qwen/Qwen3-8B-Base`
- Exact model revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`

The source checkpoint is the exact A1 checkpoint produced by the single valid Gate-A scientific execution. A retrained equivalent is not an acceptable substitute.

## Private Kaggle transport

- Dataset ref: `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`
- Privacy: private, as reported by the successful Kaggle create response
- Creation Actions run: `41373243039`
- Dataset status: `ready`
- Current version: `1`
- Dataset version operations performed: **0**
- Further dataset create/version operations authorized: **false**

Kaggle expanded the deterministic uploaded `a1-checkpoint.tar.gz` into the `a1-checkpoint/` directory when creating dataset version 1. The transport tar filename is therefore not expected to remain as a dataset file.

## Byte-level checkpoint identity

Original source checkpoint file manifest:

```text
sha256:c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0
```

Recovered Kaggle dataset checkpoint manifest, recomputed after downloading the ready version-1 dataset:

```text
sha256:c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0
```

The manifests are identical across exactly `12` checkpoint files.

Deterministic pre-upload transport archive SHA256:

```text
8f3c6dabe6a3d910b80b6acafce6155ae6350dafb28b1b987f7cdb7eff7bf32a
```

Required version-1 dataset contents verified by recovery include:

- `a1-checkpoint/adapter/adapter_model.safetensors`
- `a1-checkpoint/adapter/adapter_config.json`
- `a1-checkpoint/adapter/README.md`
- `a1-checkpoint/heads.safetensors`
- `a1-checkpoint/meta.json`
- tokenizer state under `a1-checkpoint/tokenizer/`
- `checkpoint_lineage.json`

## Attach-only recovery evidence

- Recovery Actions run: `41374175606`
- Recovery artifact ID: `13110507634`
- Recovery artifact name: `two-stage-lora24-gate-b-checkpoint-recovery`
- Recovery artifact digest: `sha256:ed0666b1d214299d1194782d021fb476a6535ebd4ed0bb0cecfda30b774e62d0`

The recovered JSON records:

- `attach_only_recovery = true`
- `dataset_create_called = false`
- `dataset_version_called = false`
- `gate_b_training_called = false`
- `dataset_version = 1`
- `status.status = ready`
- `checkpoint_file_count = 12`
- `observed_manifest_sha256 == file_manifest_sha256 == c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`

The materialization job was skipped in the successful recovery run, so the ready dataset was only attached to and verified; it was not recreated or versioned.

## Gate B consumption rule

Every Gate-B official fold must start fresh from **this exact version-1 dataset checkpoint**. Implementations must fail closed if any of the following differ:

- dataset ref or version;
- source artifact ID/digest;
- checkpoint file manifest SHA256;
- model ID/revision;
- checkpoint metadata or label order;
- required adapter/head/tokenizer files.

No fold may inherit Stage-2 weights or optimizer state from another fold.

## Still not authorized by this lock

This lock does not authorize:

- a Gate-A rerun;
- a new checkpoint dataset/version;
- Gate-B Stage-2 execution before its runner/orchestration/evaluation implementation is complete and verified;
- official OOF consumption before the formal Gate-B run;
- competition-test inference;
- Public-LB tuning;
- a Kaggle competition submission.
