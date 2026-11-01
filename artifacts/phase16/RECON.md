# Phase 16 — public-only neural asset reconnaissance

Date: 2026-11-01

## Decision

**CLOSE the free stage-one LoRA shortcut. Do not run a permanent-fold experiment from any currently discovered checkpoint.**

Phase 16 was opened after the Phase 15 external-sparse system failed to transfer from local grouped OOF (`0.4053218990802883`) to Public LB (`0.357`), while Phase 7 remained the best verified Kaggle system (`0.369`). The goal was to find a genuinely different neural representation without using a checkpoint that had already seen the official 2,699 competition training passages.

## What was checked

Read-only audits inspected the public 0.622 / Nishant two-stage-LoRA ecosystem, including:

- `nishantdahal/ntl-two-stage-lora`;
- `nishantdahal/ntl-handwriting-blend`;
- the public `two-stage-lora-on-the-public-pool` and `two-stage-lora-part-two` notebooks where accessible;
- mounted asset metadata and file manifests;
- a final public Kaggle search over Nishant-owned assets and `stage1`, `stage one`, `public-only`, `public pool`, `ntl lora`, and `two stage lora` terms.

Final asset-search workflow: `41280911745`.
Artifact: `phase16-stage1-asset-search`, ID `13073157826`, digest `sha256:5bcbc0c6999e347ae71b95c84db6f11ea5e19aa3f63e0f028a408a295f3c773d`.

The search scanned 39 candidate datasets and 42 relevant kernel hits.

## Findings

### 1. Nishant published LoRA checkpoints are final two-stage checkpoints

The public asset cards state that the LoRA checkpoints were trained:

1. first on public passages; then
2. on the competition training passages.

`nishantdahal/ntl-handwriting-blend` contains:

- `lora_qwen3_14b/adapter/adapter_model.safetensors`;
- `lora_qwen3_14b/heads.safetensors`;
- `lora_qwen25_14b/adapter/adapter_model.safetensors`;
- `lora_qwen25_14b/heads.safetensors`;
- `deberta_v3_large/...`;
- `pool_v4.parquet`.

There is no separately named stage-one/public-only checkpoint in the exposed file manifest.

Because these final checkpoints saw the official competition train in stage 2, using them to score those same official rows as OOF would contaminate validation.

### 2. `swaroopmeher/ntl-nishant-p2-qwen3-14b` is not a stage-one checkpoint

Its card explicitly describes it as an unchanged public copy of Nishant's `lora_qwen3_14b` folder, with byte-identical files.

It therefore inherits the same two-stage training provenance and is not admissible for honest OOF.

### 3. `mrralphie/ntl-pool-b2m` is stage-one *training data*, not weights

This dataset is a 56,853-row public passage pool assembled for the first stage of a two-stage LoRA classifier. It contains data/provenance/build files, not a trained stage-one adapter or neural encoder checkpoint.

It could support a new training run, but it is not a free pretrained representation.

### 4. `swaroopmsrset/agrisam-stage1-lora-seed42` is unrelated

The literal `stage1` marker is real, but the asset belongs to a different task/project and is not an NTL model-identity representation. It is a discovery false positive.

### 5. No clean stage-one-only NTL checkpoint was found

The exhaustive literal-marker search produced four candidates; after provenance review, zero are currently admissible as a public-only NTL neural checkpoint.

The earlier attempt to pull likely Nishant training-notebook refs also returned access-denied / no usable source, so there is no reproducible public stage-one snapshot recovered from that route either.

## Consequence

Do **not**:

- use Nishant's final LoRA/DeBERTa checkpoints for official OOF;
- treat copied final checkpoints as stage-one assets;
- spend a permanent-fold shot on a contaminated representation;
- submit another Kaggle entry from this line;
- retrain a 14B LoRA merely to imitate a public notebook without a separately justified experiment contract.

## Next research gate

A neural Phase 16/17 experiment is allowed only if one of these becomes true **before** looking at another permanent-fold result:

1. a verifiable public-only / stage-one NTL checkpoint is found; or
2. a new, resource-bounded neural representation is trained from scratch from an independently frozen contract, with official validation folds kept untouched.

Otherwise the correct action is to move to a genuinely different mechanism rather than continuing asset archaeology.
