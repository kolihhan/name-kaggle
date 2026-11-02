# Submission Promotion — `two-stage-lora24-v1`

Date: 2026-11-02 (MYT)

## Status

**COMPLETE — SUBMITTED CHAMPION**

- Kaggle submission: `57908683`
- Description: `two-stage-lora24-v1 plain final fit`
- Public LB: **`0.520`**
- Exact submission record: `artifacts/submission/THIRD_SUBMISSION.md`

The promotion gate is closed. Do not rerun v1 because of score and do not use Public `0.520` for post-hoc threshold/blend tuning.

## Frozen candidate

- candidate: `two-stage-lora24-v1`
- base identity: `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- Kaggle model source used for the final offline kernel: `qwen-lm/qwen-3/transformers/8b-base/1`
- Stage-1 checkpoint dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`, version `1`
- checkpoint manifest: `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`
- official train SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
- official train rows: `2699`
- final Stage-2 fit: all official rows, exactly `2` epochs
- competition prediction: **plain head only**
- centered head used for submission: `false`
- competition-test rows used for training: `0`
- competition-test rows used for model selection: `0`
- Public-LB tuning: `false`

## Gate evidence

### Gate A

- A0 held-out BA: `0.3942430082`
- A1 LoRA held-out BA: `0.4657129266`
- delta: `+0.0714699184`
- all three held-out-source deltas positive

### Gate B

- plain-head OOF BA: `0.5467611235396471`
- Phase-7 OOF reference: `0.3700521998508576`
- delta: `+0.17670892368878954`
- 5 / 5 folds won
- paired prompt-bootstrap 95% CI: `[+0.15743723589361175, +0.19594425801640553]`
- no class-recall safety veto

### Kaggle transfer

- Public LB: `0.520`
- Public minus frozen OOF: `-0.0267611235396471`
- previous Phase-7 Public champion: `0.369`
- Public improvement vs Phase 7: `+0.151`

The modest OOF-to-Public drop is materially healthier than the Phase-15 failure and validates two-stage LoRA as the submitted anchor.

## Final execution

Kaggle kernel:

- ref: `kolihhan/name-that-llm-lora24-v1-final`
- successful version: `2`
- accelerator: T4
- internet: disabled

Offline runtime:

- dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2`
- manifest: `2ede08586490ab65e52d57bc95ea95580a6c3ea0c7b269d57bef6fe117b39275`
- `transformers==4.57.1`
- `peft==0.17.1`
- `tokenizers==0.22.1`
- `huggingface-hub==0.36.0`

Training evidence:

- epoch 1 mean loss: `2.9026665187047587`
- epoch 2 mean loss: `1.630542965978384`
- optimizer steps: `300`
- skipped scaler steps: `0`
- training seconds: `3226.4846546649933`

Submission artifact:

- rows: `3`
- columns: `id, model`
- SHA-256: `6e8b2a61d98c415c39e09815fe83482407ce7ff27bd64f1c617003d9e51b3ab4`

## Operational correction

The first post-training workflow incorrectly asserted `len(submission) > 3`. The official test has exactly three rows. This was a verifier/orchestration defect, not a scientific failure. The successful kernel-v2 output was re-verified by its frozen SHA and lineage and then submitted without retraining.

Later same-description submissions created by orchestration retries are operational duplicates, not independent experiments. Do not use them as additional scientific evidence and do not create further identical retries.

## Next promotion boundary

v1 is closed. A future v2 must be a genuinely distinct, pre-frozen candidate. The highest-value next direction is complementarity using existing plain/centered/sparse disagreement evidence, with a rule fixed before another leaderboard result and a non-Public transfer check where possible.
