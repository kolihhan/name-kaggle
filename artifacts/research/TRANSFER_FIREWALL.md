# Final Transfer Firewall — Verified Lock

Date: 2026-11-03 MYT

## Status

**LOCKED.** This is a validation-budget policy, not a model result and not a Kaggle-submission authorization.

Public GitHub Actions run `41590073872` completed successfully. It downloaded immutable Phase14 artifact `13067907516`, ran the focused firewall tests, rebuilt accounting from `conservative_public_subset.parquet`, verified zero prompt-id/key overlap after removal, and uploaded the lock artifact.

Evidence artifact:

- ID: `13209726705`
- name: `transfer-firewall-lock`
- digest: `sha256:7de14265ac8490511b134a11832ab427d1e15a75077376b3d4e305a58ec2f190`

Canonical machine-readable lock: `artifacts/research/TRANSFER_FIREWALL_LOCK.json`.

## Source budget

Phase14 conservative subset contains 17 source identities.

Already consumed as fresh model-selection / transfer evidence:

- `abdullah`
- `alperen`
- `hailey`
- `jadkutbi`
- `lucasboesen`
- `maniceet`
- `semenoff`
- `willrice`
- `zakir`

These nine sources are now development/regression evidence only.

Final firewall sources — every remaining admissible source, with no ranking or model-score selection:

- `abhinav`
- `dophuchung`
- `kaplinskiy`
- `keshav`
- `prvsiyan`
- `prvsiyan_v1`
- `seyit`
- `zbigniew`

`model_scores_used_for_selection=false`.

## Leakage-safe accounting

- firewall rows: `15,206`
- consumed-side rows before prompt leakage removal: `28,575`
- training rows removed because their `prompt_id` or `prompt_key` appears in the firewall: `18,216`
- development/training rows after leakage removal: `10,359`
- prompt-id overlap after removal: `0`
- prompt-key overlap after removal: `0`

The large removal count is expected because several public sources reuse common audited prompt banks. It is precisely why raw source identity alone is not enough for an honest transfer split.

## Coverage limitation

The frozen firewall is not a perfect all-18-label oracle.

- eligible firewall rows: `14,013`
- eligible labels: `17 / 18`
- missing eligible label: `claude-fable-5`
- the leakage-clean development side also retains the same 17 represented labels

This limitation was discovered **after** the all-unconsumed firewall rule was frozen. Do not now shrink, swap, or cherry-pick firewall sources to recover the eighteenth label. That would turn validation design into post-hoc optimization.

The correct interpretation is: this is a strong 17-label source-shift firewall, not a hidden-test replacement and not an all-label promotion oracle.

## Policy after v3

1. Do not mint a new fresh source holdout for each future candidate.
2. Use the already-spent source splits for mechanism development and regression diagnostics.
3. A materially new candidate must be frozen before any firewall prediction is produced.
4. The firewall may be consumed once for that frozen candidate. Afterward it is spent.
5. Do not tune weights, thresholds, label subsets, routers, pooling variants, or source membership from firewall results.
6. Official grouped OOF remains adaptive comparison evidence only.
7. Kaggle Public LB remains one-shot external evidence, never a tuning loop.

## v3 boundary

This firewall does not modify `dct-temporal-fingerprint-v3` and must not be retrofitted into its already-frozen transfer gate.

The v3 scientific run remains `41580819667`; its existing source split is `alperen / hailey / willrice` and remains the only valid v3 transfer gate.
