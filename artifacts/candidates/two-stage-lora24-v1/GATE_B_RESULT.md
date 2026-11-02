# Gate B Result — `two-stage-lora24-v1`

Date: 2026-11-02 (MYT)

## Verdict

**PASS**

The first valid frozen Gate-B official OOF execution completed successfully. The primary **plain head** beats the frozen Phase-7 reference under all four predeclared Gate-B conditions.

This closes the one-shot Gate-B result boundary for `two-stage-lora24-v1`. Do not rerun Gate B because of score. The centered head remains diagnostic only under v1.

## Scientific execution / evidence lineage

- Candidate: `two-stage-lora24-v1`
- Existing Gate-B Actions run: `41398543856`
- Run conclusion: `success`
- Evidence artifact: `13125872633`
- Artifact name: `two-stage-lora24-gate-b-evidence`
- Artifact digest: `sha256:fa675fcf8ac3860984b449748cf067c3da29389a921417e5e1047bb22a7be8ce`
- Candidate OOF rows: `2699`
- Candidate prompts: `150`
- Candidate labels: `18`
- Exact Stage-1 checkpoint dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`, version `1`
- Checkpoint manifest: `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`
- Runtime dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-gate-b-runtime`, version `1`
- Runtime manifest: `8d6c05d5ec29841d4a25cbf7c903c5d0414d3bf754b4bdd80b1cfdba64c1a933`

No new GitHub Actions execution was launched to consume this result; the already-existing completed artifact was read and evaluated.

## Frozen comparison

| Metric | Phase 7 | Gate B plain | Delta |
|---|---:|---:|---:|
| OOF balanced accuracy | `0.3700521998508576` | **`0.5467611235396471`** | **`+0.17670892368878954`** |

Candidate fold BAs:

`0.5388888889 / 0.5351851852 / 0.5296296296 / 0.5814814815 / 0.5487867178`

Phase-7 fold BAs:

`0.3833333333 / 0.3388888889 / 0.3722222222 / 0.3870370370 / 0.3688378033`

Gate-B plain beats Phase 7 on **5 / 5** folds.

Centered-head diagnostic BA: **`0.5478498632861049`**. It is not promoted or blended inside v1 because the frozen contract declares the plain head primary and centered head diagnostic only.

## Paired prompt bootstrap

Frozen settings:

- sampling unit: `prompt_id`
- prompt groups: `150`
- resamples: `5000`
- seed: `20261101`

Candidate-minus-Phase7 BA:

- point delta: `+0.17670892368878954`
- 95% CI lower: `+0.15743723589361175`
- 95% CI upper: `+0.19594425801640553`

The 5,000-resample result was computed with a vectorized implementation that preserves the frozen evaluator's sorted-prompt sampling and `numpy.random.RandomState(20261101)` sequence. It was cross-checked against the repository's original bootstrap implementation on identical initial resamples before the full run.

## Four-condition decision

1. Candidate overall BA > Phase 7: **PASS**
2. Bootstrap 95% CI lower bound > 0: **PASS**
3. Candidate wins at least 3 / 5 folds: **PASS — 5 / 5**
4. Class-recall safety veto: **PASS — no veto labels**

Frozen decision: **PASS**.

## Per-class recall

| Label | Phase 7 | Gate B plain | Delta |
|---|---:|---:|---:|
| claude-fable-5 | `0.14094` | `0.25503` | `+0.11409` |
| claude-opus-4-6 | `0.51333` | `0.74667` | `+0.23333` |
| claude-opus-4-8 | `0.20667` | `0.61333` | `+0.40667` |
| claude-opus-5 | `0.61333` | `0.78000` | `+0.16667` |
| claude-sonnet-5 | `0.38000` | `0.65333` | `+0.27333` |
| gemini-3.1-pro-preview | `0.64667` | `0.82000` | `+0.17333` |
| gemini-3.6-flash | `0.41333` | `0.67333` | `+0.26000` |
| gemini-3.7-flash | `0.39333` | `0.36667` | `-0.02667` |
| gemini-3.8-flash | `0.48000` | `0.55333` | `+0.07333` |
| glm-5.3 | `0.18000` | `0.19333` | `+0.01333` |
| glm-5.3-flash | `0.20000` | `0.28000` | `+0.08000` |
| gpt-5.5 | `0.38000` | `0.54000` | `+0.16000` |
| gpt-5.6-luna | `0.36000` | `0.51333` | `+0.15333` |
| gpt-5.6-sol | `0.18667` | `0.33333` | `+0.14667` |
| gpt-5.6-terra | `0.28000` | `0.54667` | `+0.26667` |
| gpt-6-astra | `0.57333` | `0.80000` | `+0.22667` |
| kimi-k3 | `0.24000` | `0.51333` | `+0.27333` |
| qwen3.8-max | `0.47333` | `0.66000` | `+0.18667` |

The only negative recall delta is `gemini-3.7-flash = -0.02667`, far below the frozen `0.15` safety-veto threshold.

## Leakage / execution invariants

Verified from the completed Gate-B artifact:

- every fold loads a fresh copy of the exact Stage-1 checkpoint;
- fit rows are only `fold_id != f`;
- validation rows are only `fold_id == f`;
- zero fit/validation prompt overlap;
- class weights are fit-partition-only;
- exactly two Stage-2 epochs;
- no validation-driven early stopping/model selection;
- public Stage-1 rows mixed into Stage 2: `0`;
- competition-test rows used: `0`;
- Public-LB tuning: `false`;
- no post-Stage2 checkpoint was saved by the OOF workers;
- Kaggle competition submission authorized by Gate B itself: `false`.

## Evidence strength

**Strong-but-adaptive / supporting, not independent hidden-distribution evidence.**

The permanent official OOF rows have been repeatedly consumed across the project, so this OOF result must not be treated as a hidden-test oracle. Gate A's source-level holdout supplies separate mechanism/transfer evidence, but the next competition-facing check remains a separately authorized frozen submission.

## Promotion consequence

`two-stage-lora24-v1` is now eligible for a **separate submission promotion/preflight**. Gate B does not auto-submit.

For v1 submission, preserve the frozen scientific identity:

- initialize from the exact Stage-1 A1 checkpoint;
- final-fit Stage 2 on all `2699` official rows using the same two-epoch Stage-2 configuration;
- use **plain head only** for v1 predictions;
- do not introduce centered blending, sparse arbitration, new backbones, new data, thresholding, Sinkhorn/Hungarian, or Public-LB-driven changes inside v1;
- verify final-fit lineage, offline notebook reproducibility, submission schema, and no competition-test use during training before explicit submission authorization.
