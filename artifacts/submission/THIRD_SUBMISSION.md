# Third Submission — `two-stage-lora24-v1`

Date: 2026-11-02

## Result

**NEW CHAMPION**

- Kaggle submission ref: `57908683`
- Description: `two-stage-lora24-v1 plain final fit`
- Status: `COMPLETE`
- Public LB: **`0.520`**
- Private LB: unavailable during competition

Previous submitted champion:

- Phase 7: `0.369`, submission `57838845`

Absolute Public-LB improvement over Phase 7:

- **`+0.151`**

Frozen Gate-B OOF for the same v1 candidate:

- OOF balanced accuracy: `0.5467611235396471`
- Public minus OOF: **`-0.0267611235396471`**

This is a materially better transfer result than Phase 15 (`0.4053218990802883` OOF -> `0.357` Public). The two-stage LoRA direction is therefore promoted to the submitted competition anchor.

## Frozen scientific identity

The submitted v1 preserves the predeclared identity:

- candidate: `two-stage-lora24-v1`
- model: `Qwen/Qwen3-8B-Base`
- model revision recorded in lineage: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- Kaggle model source: `qwen-lm/qwen-3/transformers/8b-base/1`
- exact Stage-1 checkpoint dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`, version `1`
- Stage-1 checkpoint manifest: `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`
- official training rows: `2699`
- official train SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
- Stage-2 epochs: exactly `2`
- final primary head: `plain`
- centered head used for submission: `false`
- competition-test rows used for training: `0`
- competition-test rows used for model selection: `0`
- Public-LB tuning: `false`

## Final-fit execution evidence

Kaggle kernel:

- kernel ref: `kolihhan/name-that-llm-lora24-v1-final`
- successful kernel version: `2`
- accelerator: T4
- internet: disabled

Training log:

- epoch 1 mean loss: `2.9026665187047587`
- epoch 2 mean loss: `1.630542965978384`
- optimizer steps: `300`
- skipped scaler steps: `0`
- training seconds: `3226.4846546649933`

Offline runtime:

- runtime dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2`
- runtime manifest: `2ede08586490ab65e52d57bc95ea95580a6c3ea0c7b269d57bef6fe117b39275`
- `transformers==4.57.1`
- `peft==0.17.1`
- `tokenizers==0.22.1`
- `huggingface-hub==0.36.0`

Generated hidden-test artifact:

- rows: `3`
- columns: `id, model`
- SHA-256: `6e8b2a61d98c415c39e09815fe83482407ce7ff27bd64f1c617003d9e51b3ab4`

The first post-training verifier incorrectly asserted `len(submission) > 3`. The official test file contains exactly three rows, so that verifier rejected a valid artifact. The kernel itself had already completed successfully. The existing kernel-v2 output was subsequently re-verified by SHA and lineage before submission; the model was **not retrained** for this correction.

## Interpretation

### FACT

- v1 Public `0.520` is far above Phase 7 Public `0.369`.
- v1 Public is `0.02676` below its frozen Gate-B OOF `0.54676`.
- The hidden-distribution drop is modest compared with Phase 15 and supports the two-stage LoRA mechanism as the new competition anchor.

### Do not infer

- `0.520` is not evidence that Public LB should now be used for threshold/blend tuning.
- The post-Gate-B plain/centered/sparse majority diagnostic remains adaptive OOF evidence and is **not** silently promoted into this v1 result.
- No additional submission should be created merely to probe Public LB.

## Duplicate-submission note

Orchestration retries also created later submissions with the same description that were still reported `PENDING` when the completed score was checked. They do not represent new scientific candidates and should not be interpreted as independent experiments.

## Next candidate

Keep `two-stage-lora24-v1` / Public `0.520` as the submitted anchor. The next candidate should target **complementarity**, not another identical 8B retrain: freeze a new v2 rule using existing plain/centered/sparse error diversity, obtain a non-Public transfer check where possible, and only then spend another submission.
