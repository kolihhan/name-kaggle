# Gate B checkpoint materialization request

Date: 2026-11-02 (MYT)

This request authorizes exactly one non-scientific transport operation for `two-stage-lora24-v1`:

- source GitHub Actions run: `41333631967`
- source artifact: `13094069155`
- source artifact digest: `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`
- destination: private Kaggle dataset `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`

This request does **not** authorize Gate A retraining, Gate B Stage-2 training, official OOF execution, competition-test inference, Public-LB tuning, or a Kaggle competition submission. The workflow must fail closed rather than overwrite or version an existing dataset with the same slug.
