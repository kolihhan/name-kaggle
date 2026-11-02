# Gate B runtime lock — `two-stage-lora24-v1`

Date: 2026-11-02 (MYT)

This file locks the immutable official-OOF runtime transport used by Gate B. It is transport/orchestration evidence only; it does not change the frozen scientific contract.

## Dataset identity

- private Kaggle dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-gate-b-runtime`
- status: `ready`
- version: `1`
- source commit embedded in lineage: `1d466f4c756394aea4b57f056238ff457f9f0852`
- official train SHA256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
- permanent fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`
- official rows: `2699`

## Exact runtime file set

Exactly eight frozen files are carried by the runtime dataset:

1. `data/train.csv`
2. `artifacts/phase1/folds.csv`
3. `scripts/__init__.py`
4. `scripts/run_two_stage_lora24_gate_b_gpu.py`
5. `scripts/run_two_stage_lora24_gate_b.py`
6. `scripts/run_two_stage_lora24_gate_a.py`
7. `scripts/run_style_baseline.py`
8. `scripts/run_strong_sparse_anchor.py`

The lineage declares:

- 8-file manifest SHA256: `8d6c05d5ec29841d4a25cbf7c903c5d0414d3bf754b4bdd80b1cfdba64c1a933`
- deterministic pre-upload transport SHA256: `f818750300838089f3610347074da52a8340d74a57f54b698ce6c65ea68c3785`
- competition-test rows included: `0`
- public Stage-1 rows included: `0`
- scientific configuration changed: `false`

Kaggle expands the uploaded transport archive when exposing the dataset. Therefore the immutable scientific identity is verified from the expanded eight files and their declared per-file SHA256/size values; the original transport SHA remains provenance evidence, not a requirement that the tar container itself survive mounting.

## Attach-only byte verification

Expanded-layout attach-only verification succeeded in Actions run `41397900907` at commit `a11056d829fbe7b829b886f522b75b678543806c`.

Recovery evidence:

- artifact ID: `13119669832`
- artifact digest: `sha256:a1df7ac90acf20010970b03b504b8de66324c1488676425559b716fc27160001`
- artifact size: `2177` bytes
- dataset status: `ready`
- dataset version: `1`
- recovered expanded-file manifest SHA256: `8d6c05d5ec29841d4a25cbf7c903c5d0414d3bf754b4bdd80b1cfdba64c1a933`
- expanded-file manifest verified: `true`
- attach-only recovery: `true`
- dataset create called: `false`
- dataset version called: `false`
- Gate-B training called: `false`

The materialization/create job was explicitly skipped in this recovery run.

## Runtime consumption rule

Any Gate-B scientific execution must:

1. attach only this ready version-1 runtime dataset;
2. locate exactly one `runtime_lineage.json`;
3. require the exact source commit, official train SHA, fold fingerprint, row count, manifest SHA, and transport-provenance SHA above;
4. locate exactly one expanded root containing all eight declared runtime files;
5. verify every file's SHA256 and size against the lineage;
6. rebuild the eight-file manifest and require SHA256 `8d6c05d5ec29841d4a25cbf7c903c5d0414d3bf754b4bdd80b1cfdba64c1a933` before any model training starts;
7. copy only those verified files into the writable Kaggle working directory;
8. never recreate or version the runtime dataset inside the Gate-B execution path.

A different runtime dataset version, changed file manifest, changed official train bytes/folds, or a recreated runtime is not an equivalent substitute for this lock.
