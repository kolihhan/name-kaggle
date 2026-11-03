# Privileged Centered-to-Plain Distillation — Research Dossier

Date: 2026-11-03 MYT
Status: RESEARCH ONLY — not an activated candidate, not authorized to consume the final transfer firewall, official OOF, or Kaggle Public LB.

## Problem this mechanism targets

The submitted v1 trains two views of the same response representation:

- plain student view: `z`
- privileged centered view: `z_i - mean(z_same_prompt_microbatch)`

The centered view can cancel prompt-shared nuisance only because same-prompt peers are available during training/evaluation. Hidden competition test rows do not provide reliable prompt groups, so centered inference is not deployable.

The verified v1 mechanism audit showed that centered is complementary rather than globally superior:

- centered rescues 353 plain errors;
- centered breaks 350 plain-correct rows;
- net row gain is only +3;
- 207 / 353 centered rescues are same-family plain errors.

Therefore the evidence does **not** justify unconditional plain-to-centered KL matching.

## Research hypothesis

Treat the centered view as **privileged information** available only during training. Transfer its nuisance-cancelled decision signal into the deployable plain head only on examples where the centered teacher is demonstrably more reliable with respect to the training label.

This is a Learning Using Privileged Information / generalized-distillation framing:

- teacher observes privileged same-prompt context through centering;
- student observes only the single-response representation available at inference;
- teacher signal is detached;
- ordinary supervised CE on the student remains the primary objective.

Relevant literature:

- Vapnik & Izmailov, *Learning Using Privileged Information: Similarity Control and Knowledge Transfer*, JMLR 2015: https://www.jmlr.org/papers/v16/vapnik15b.html
- Lopez-Paz et al., *Unifying distillation and privileged information*, ICLR 2016: https://leon.bottou.org/papers/lopez-paz-2016
- Martinez-Garcia et al., *Teacher privileged distillation: How to deal with imperfect teachers?*, Knowledge-Based Systems 2025: https://www.sciencedirect.com/science/article/pii/S0950705125003855
- Penaloza et al., *Privileged Information Distillation for Language Models*, ICML 2026: https://proceedings.mlr.press/v306/penaloza26a.html

The literature motivates the form of the experiment; it does not establish that centered-to-plain transfer will work for this competition.

## Safest objective shape

For each same-prompt training microbatch, compute the already-existing v1 logits:

- `p = plain(z)`
- `c = centered(center_triplet(z))`

Keep the ordinary plain supervised loss:

`L_plain = CE(p, y)`

Define teacher eligibility from **training labels only**, per row:

`eligible_i = CE(c_i, y_i) < CE(p_i, y_i)`

Only eligible rows receive detached teacher soft targets:

`L_KD = mean_i[ eligible_i * T^2 * KL(softmax(c_i/T).detach() || softmax(p_i/T)) ]`

Candidate objective, if this direction is activated later:

`L = L_plain + L_centered + lambda * L_KD`

The exact `lambda` and `T` must be frozen from prior art / predeclared engineering rationale before any new transfer-firewall result. They must not be swept against OOF, firewall, or Public LB.

## Why selection is necessary

The centered head is an imperfect teacher. The verified v1 OOF behavior is almost perfectly balanced between rescues and breaks, so unconditional imitation would force the student toward many known teacher mistakes as well as teacher successes.

The eligibility rule is not a router and is never used at inference. It is a training-only teacher-quality guard using labels already legitimately available for supervised training.

The rule is also deliberately generic: it is derived from teacher-vs-student supervised loss, not from post-hoc inspection of which model families or length buckets happened to be rescued in spent OOF evidence.

## What this is materially different from

Rejected v2 used cross-prompt exact-model prototypes to shape representation geometry. It failed its frozen fresh-transfer gate.

This hypothesis instead transfers **same-prompt nuisance cancellation** from an observed privileged centered teacher into a response-only deployable student. It does not introduce cross-prompt prototype attraction, a family router, prompt reconstruction, or inference-time grouping.

## Failure modes to watch

1. **Teacher/student collapse** — because both views share the backbone, KD can become redundant.
2. **Moving-teacher instability** — a jointly trained centered head is not a fixed oracle. Teacher logits must be detached; if activated, the contract must decide whether to use simultaneous or staged distillation before seeing results.
3. **Selection starvation** — too few eligible rows could make KD irrelevant.
4. **Easy-example bias** — teacher eligibility may select only already-easy rows rather than genuine nuisance-cancellation cases.
5. **Overconfidence transfer** — temperature and weight can amplify centered mistakes; no hyperparameter sweep is allowed from firewall/Public LB evidence.
6. **Shared-backbone leakage of objective** — even with detached teacher logits, KD gradients alter the shared backbone through the plain path. This is intended but must be explicit in the contract.

## Evidence policy

- Current v1 OOF rescue/break analysis is spent diagnostic evidence only.
- Do not choose model-family masks, length thresholds, confidence thresholds, or KD hyperparameters from those OOF rows.
- Do not activate this candidate while v3 is unresolved.
- If v3 fails, freeze a complete v4 contract first.
- Development may use already-spent source evidence.
- The final 8-source firewall may be consumed only once by a frozen materially-new candidate, according to `artifacts/research/TRANSFER_FIREWALL.md`.
- A firewall PASS still does not authorize leaderboard tuning; official grouped OOF and one-shot Public LB remain separate evidence layers.

## Current implementation scope

Only a synthetic loss primitive and unit tests are authorized now. They must not load any competition/public-pool rows or produce any model score.
