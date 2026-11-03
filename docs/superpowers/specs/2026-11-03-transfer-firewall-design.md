# Transfer Firewall Design

Date: 2026-11-03 MYT

## Goal

Prevent future Name That LLM candidates from meta-overfitting the finite Phase14 public-source pool by stopping the pattern of minting a new whole-source holdout for every idea.

## Current evidence budget

Already consumed as model-selection / mechanism-transfer evidence:

- v1 Gate A: `abdullah`, `semenoff`, `zakir`
- rejected v2 paired transfer: `jadkutbi`, `lucasboesen`, `maniceet`
- frozen v3 paired transfer: `alperen`, `hailey`, `willrice`

These sources remain valid development/regression evidence but are permanently spent as fresh evidence.

## Firewall rule

Use the exact Phase14 conservative subset from `pool_v4.parquet`. Reserve **every admissible source not listed above** as the final source-shift firewall.

No ranking, score-based source choice, source-size optimization, label cherry-picking, or retry split is allowed. The selection uses source identity only and therefore does not inspect model scores.

For any future firewall evaluation:

1. training may use only the consumed-source side;
2. remove every training row whose `prompt_id` or `prompt_key` occurs in any firewall row;
3. the firewall remains untouched during mechanism development, hyperparameter selection, thresholding, routing, representation selection, and candidate choice;
4. a candidate must be frozen before firewall predictions are generated;
5. firewall results may be consumed only once for promotion of that frozen candidate;
6. after the firewall is consumed, it becomes spent evidence and cannot be rebranded as fresh for a later candidate.

## Development policy after v3

- Reuse already-spent source splits for development diagnostics.
- Do not create v4/v5/v6 by allocating another fresh source subset each time.
- Official permanent folds and Phase19 shadows remain adaptive/spent evidence.
- Kaggle Public LB remains external evidence only and cannot select candidate details.

## Eligibility accounting

The lock generator must report, without fitting a model:

- complete source inventory;
- consumed source list;
- firewall source list;
- heldout rows;
- training rows before/after prompt leakage removal;
- eligible firewall labels with at least 20 heldout rows and representation in training;
- eligible firewall rows;
- prompt-id and prompt-key overlap after removal, both required to be zero;
- explicit `model_scores_used_for_selection=false`.

If the reserved firewall leaves weak coverage or insufficient training support, record that as a limitation. Do **not** shrink or replace the firewall after inspecting the accounting merely to make the gate easier.

## Relationship to v3

This policy does not modify `dct-temporal-fingerprint-v3`, its already-frozen transfer split, or queued scientific run `41580819667`.

If v3 fails, it stops under its existing contract. The next candidate must first prove its mechanism on spent development evidence, then freeze before touching this firewall.

If v3 passes, continue its predeclared official OOF and deployment gates unchanged. The firewall remains reserved for a later materially new candidate rather than being retrofitted into v3.