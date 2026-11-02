# NOW

Updated: 2026-11-02

## Competition state

- **Current submitted champion: `two-stage-lora24-v1` — Public LB `0.520`, submission `57908683`.**
- Previous champion: Phase 7 gated specialist — Public LB `0.369`, submission `57838845`.
- Phase 15 remains rejected — OOF `0.4053218990802883`, Public LB `0.357`, submission `57861337`.
- Absolute Public-LB gain of v1 over Phase 7: **`+0.151`**.
- Exact submission record: `artifacts/submission/THIRD_SUBMISSION.md`.

## `two-stage-lora24-v1` evidence

### Gate A

- A0 held-out public-source BA: `0.3942430082`
- A1 LoRA BA: `0.4657129266`
- delta: `+0.0714699184`
- all three held-out-source deltas positive

### Gate B

- primary plain-head OOF BA: **`0.5467611235396471`**
- centered diagnostic OOF: `0.5478498632861049`
- Phase-7 reference OOF: `0.3700521998508576`
- delta vs Phase 7: `+0.17670892368878954`
- fold wins: **5 / 5**
- frozen 5,000-resample prompt-bootstrap 95% CI for candidate-minus-Phase7: `[+0.15743723589361175, +0.19594425801640553]`
- class-recall safety veto: not triggered
- exact Gate-B result: `artifacts/candidates/two-stage-lora24-v1/GATE_B_RESULT.md`

### Kaggle transfer

- Public LB: **`0.520`**
- Gate-B OOF: `0.5467611235396471`
- Public minus OOF: **`-0.0267611235396471`**

This is a materially better transfer pattern than Phase 15 and promotes two-stage LoRA to the competition anchor. Do **not** reinterpret Public `0.520` as permission to tune thresholds/blends directly against Public LB.

## Frozen submission lineage

- candidate: `two-stage-lora24-v1`
- base: `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- Kaggle model source: `qwen-lm/qwen-3/transformers/8b-base/1`
- Stage-1 checkpoint dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`, version `1`
- checkpoint manifest: `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`
- official train rows: `2699`
- official train SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`
- Stage-2 final fit: exactly `2` epochs
- primary submission head: `plain`
- centered head used for submission: `false`
- test rows used for training: `0`
- test rows used for model selection: `0`
- Public-LB tuning: `false`

## Final Kaggle execution

- kernel: `kolihhan/name-that-llm-lora24-v1-final`
- successful kernel version: `2`
- T4, internet disabled
- runtime dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2`
- runtime manifest: `2ede08586490ab65e52d57bc95ea95580a6c3ea0c7b269d57bef6fe117b39275`
- epoch-1 mean loss: `2.9026665187047587`
- epoch-2 mean loss: `1.630542965978384`
- optimizer steps: `300`
- skipped scaler steps: `0`
- generated submission rows: `3`
- submission artifact SHA-256: `6e8b2a61d98c415c39e09815fe83482407ce7ff27bd64f1c617003d9e51b3ab4`

The first post-training verifier incorrectly asserted `len(submission) > 3`; the official competition test contains exactly three rows. The successful kernel output was subsequently re-verified by SHA and lineage and submitted **without retraining**.

## Duplicate-submission note

Orchestration retries produced later submissions with the same `two-stage-lora24-v1 plain final fit` description that were still `PENDING` when the completed result was checked. These are duplicate operational retries, **not new scientific experiments**. Do not create more submissions to probe Public LB.

## Post-Gate-B optimization evidence

Existing read-only error-overlap diagnostics show substantial complementary signal:

- Phase-14 four-view SVC OOF BA: `0.4178100919711658`
- plain LoRA vs Phase14 hard-prediction disagreement: `0.5602074842534271`
- LoRA-only correct rows: `641`
- Phase14-only correct rows: `293`
- either-component oracle BA: `0.6552970420084514`
- plain vs centered disagreement: `0.4201556131900704`
- two-head oracle BA: `0.6775466070096942`
- adaptive diagnostic three-way hard majority (`plain`, `centered`, Phase14 SVC; all-distinct tie -> plain): `0.5674869500372856`, +`0.020725826497638433` over plain, 5/5 folds improved, same-OOF prompt-bootstrap CI `[+0.01111111, +0.03037037]`

**This majority is adaptive diagnostic evidence only.** It was examined after v1 OOF was visible and must not be silently promoted or submitted as if independently validated.

Public-method research already in the repo supports the broader complementarity direction: strong public systems combine two-stage LoRA representations with additional classifiers/backbones rather than relying on one 8B plain head alone. See `artifacts/research/public-two-stage-lora-audit-2026-11-01.md`.

## Current execution policy

- `two-stage-lora24-v1` is the submitted anchor. **Do not rerun/retrain v1 merely because the score is known.**
- Do not spend another Kaggle submission on an identical v1 retry.
- Do not tune directly against Public `0.520`.
- GitHub Actions should return to a minimal/read-only role; the one-shot training/submission/status workflows were operational scaffolding, not the scientific workflow.
- Repository was temporarily made public by explicit user authorization for execution. Training/submission are now complete, so it should be returned to **private**. Current connector permissions do not expose repository-visibility mutation; do not falsely claim it was changed.

## Next single action

**Freeze a new v2 complementarity candidate rather than retraining v1.** Start from the already-observed plain/centered/sparse diversity, define one simple combination rule before another Kaggle result, seek a non-Public transfer check where possible, and spend at most one next submission on the frozen v2 candidate.
