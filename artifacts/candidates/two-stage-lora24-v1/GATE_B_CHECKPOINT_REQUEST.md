# Gate B checkpoint materialization request

Date: 2026-11-02 (MYT)

This request authorizes exactly one completed non-scientific transport operation for `two-stage-lora24-v1`:

- source GitHub Actions run: `41333631967`
- source artifact: `13094069155`
- source artifact digest: `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`
- destination: private Kaggle dataset `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`

Infrastructure-only retry history:

1. Actions run `41340518541` failed at `kaggle datasets create` because Kaggle CLI 2.2.5 imported the undeclared runtime dependency `six`. Artifact download, digest check, checkpoint-meta validation, and deterministic manifest creation had passed. No destination dataset was created.
2. Actions run `41372745749` passed the `six` dependency fix and all lineage checks, then failed locally before upload because the dataset title was 53 characters while Kaggle requires 6–50. The exact file manifest remained `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0` and the deterministic transport SHA256 remained `8f3c6dabe6a3d910b80b6acafce6155ae6350dafb28b1b987f7cdb7eff7bf32a`. No destination dataset was created.

This authorizes another infrastructure-only retry of the exact same source artifact and destination after shortening only the human-readable dataset title. No checkpoint bytes, archive bytes, source artifact, destination slug, scientific configuration, or lineage may change.

This request does **not** authorize Gate A retraining, Gate B Stage-2 training, official OOF execution, competition-test inference, Public-LB tuning, or a Kaggle competition submission. The workflow must fail closed rather than overwrite or version an existing dataset with the same slug.
