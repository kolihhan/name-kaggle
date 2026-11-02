# Gate B checkpoint materialization request

Date: 2026-11-02 (MYT)

This request authorizes exactly one completed non-scientific transport operation for `two-stage-lora24-v1`:

- source GitHub Actions run: `41333631967`
- source artifact: `13094069155`
- source artifact digest: `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`
- destination: private Kaggle dataset `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`

Infrastructure-only history:

1. Actions run `41340518541` failed at `kaggle datasets create` because Kaggle CLI 2.2.5 imported the undeclared runtime dependency `six`. Artifact download, digest check, checkpoint-meta validation, and deterministic manifest creation had passed. No destination dataset was created.
2. Actions run `41372745749` passed the `six` dependency fix and all lineage checks, then failed locally before upload because the dataset title was 53 characters while Kaggle requires 6–50. The exact file manifest remained `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0` and the deterministic transport SHA256 remained `8f3c6dabe6a3d910b80b6acafce6155ae6350dafb28b1b987f7cdb7eff7bf32a`. No destination dataset was created.
3. Actions run `41373243039` revalidated the same source artifact, reproduced the same file-manifest and transport digests, uploaded `checkpoint_lineage.json` and the exact 111 MB `a1-checkpoint.tar.gz`, and Kaggle returned `Your private Dataset is being created` for the exact destination slug. The workflow then failed only because it immediately called `kaggle datasets files` while the newly created dataset was not yet readable and received HTTP 403. The create operation had already succeeded, so **no further dataset create or version operation is authorized**.
4. Attach-only recovery run `41373774444` observed the existing dataset as `ready`, `current_version_number=1`, and listed the complete expanded checkpoint tree plus `checkpoint_lineage.json`. Its evidence step then failed because it expected the uploaded tarball filename, while Kaggle had expanded the tarball to `a1-checkpoint/...`. No create, version, upload, training, OOF, test inference, or submission operation occurred in that recovery.

This commit authorizes one more attach-only verification of the already-ready version-1 dataset. It may download the existing dataset, recompute the 12-file checkpoint manifest under `a1-checkpoint/`, and compare it to the original manifest `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`. It must not upload data, create a dataset, create a dataset version, retrain Gate A, execute Gate B Stage 2, score official OOF, use competition test rows, tune on Public LB, or submit to the competition.
