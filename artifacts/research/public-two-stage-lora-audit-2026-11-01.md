# Public two-stage LoRA audit — 2026-11-01

Status: **research evidence only — no checkpoint promotion and no Kaggle submission authorization**

## Question

Does the current public Kaggle work reveal a materially stronger modeling mechanism that can be reproduced honestly under this repository's evaluation policy, rather than copied as a prefit checkpoint that has already seen all official competition train rows?

## Verdict

**YES — the LoRA24 training mechanism is worth a clean fold-local reproduction. Public final checkpoints are not valid evidence for honest repo OOF.**

The strongest surviving direction is a small first candidate based on **Qwen3-8B-Base, truncated to the first 24 layers, with LoRA plus learned plain/centered attribution heads**. The important distinction is that the centered head is trained jointly on same-prompt groups; this is not the failed `prompt-residual-style-v1` LDA projection.

Do not copy the public Stage-2 final checkpoint into local OOF. The public recipe fine-tunes Stage 2 on all 2,699 host passages for final inference, so that checkpoint has seen every official row.

## Sources inspected

Pulled public Kaggle source notebooks using the Kaggle API in temporary research workflows:

- `cosmorin/name-that-llm-how-to-reproduce-0-636`;
- `nishantdahal/two-stage-lora-on-the-public-pool`;
- `lucasboesen/name-that-llm-train-lora24-s7`.

`lightningv08/ntl-lora24-fivefold-training` was referenced by the public notebooks but returned a 403 to the connected Kaggle account, so its exact source was not independently read. Lucas's public reproduction of the LoRA24 recipe was readable and contains the fold-local implementation needed for this audit.

## FACT — public two-stage method

Nishant's public notebook describes this two-stage procedure:

1. Stage 1 trains on public passages only, to learn model-specific writing/style signal.
2. Stage 2 fine-tunes on the host competition train distribution.

Reported historical local results in that notebook:

| Model | Local grouped CV |
| --- | ---: |
| n-gram SVC on train + public pool | 0.435 |
| Qwen3-8B LoRA, public Stage 1 only | 0.483 |
| Qwen3-8B, Stage 1 + Stage 2 | 0.569 |
| Qwen3-14B, Stage 1 + Stage 2 | 0.569 |
| 8B + 14B + n-gram + centered twin + balancing | 0.593 |

The same notebook reports Public LB `0.592` for that historical ensemble and says the 8B/14B LoRAs disagree on about 22% of passages, explaining why blending them can help even when their standalone score is similar.

A newer public 0.636-style reproduction uses the same broad family of mechanisms with stronger public data/checkpoints. That is useful evidence that the mechanism remains competitive, but Public LB is not a local tuning target and does not prove Private transfer.

## FACT — LoRA24 architecture/training recipe

The public recipe uses:

- base: `Qwen3-8B-Base` for the smallest version;
- keep the first `24` transformer layers;
- LoRA rank `16`, alpha `32`, dropout `0.05`;
- target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`;
- pooled hidden states from layers `[2, 4, 6, 8, 12, 16, 20, 24]`;
- mask the first token before mean pooling;
- one LayerNorm per pooled layer;
- concatenate the eight normalized pooled vectors;
- two linear 18-class heads: `plain` and `centered`.

Training batches are same-prompt triplets. The loss is:

```text
CE(plain(z), y) + CE(centered(z - mean_triplet(z)), y)
```

This directly trains the centered representation to separate different model answers to the same prompt.

Original public Stage-1 recipe:

- one epoch over about `34,940` public passages;
- class-weighted cross entropy because the pool is unbalanced;
- AdamW;
- LoRA LR `1e-4`;
- heads/norms LR `1e-3`;
- 10% warm-up then cosine;
- six triplets per optimizer step / equivalent accumulation;
- public notebook uses a filtered pool spanning all 18 labels.

Original public Stage-2 recipe:

- two epochs on official train;
- LoRA LR `5e-5`;
- heads/norms LR `5e-4`;
- the notebook explicitly says more Stage-2 epochs over-fit and mixing public pool rows back into Stage 2 was worse.

## FACT — honest fold-local training is technically supported

Lucas's readable LoRA24 training notebook implements five outer prompt-group folds rather than only a final all-row fit.

For each non-final fold it sets:

```text
fit   = official rows whose prompt group is not the held-out fold
valid = official rows whose prompt group is the held-out fold
```

It trains the LoRA, norms, plain head, and centered head only on `fit`, then produces representations/readouts for `valid`. A separate `final` job fits all official rows only for inference.

Therefore the mechanism can be reproduced honestly in this repo by initializing every outer fold from the **same Stage-1 public-only checkpoint**, then performing Stage 2 only on that fold's official training prompt groups. The repository's existing permanent prompt-group folds should be used for historical comparability; do not replace them with Lucas's seed-0 fold mapping.

## FACT — centered twin is not equivalent to the failed residual-LDA candidate

The closed repo candidate `prompt-residual-style-v1` used:

- frozen ModernBERT embeddings;
- prompt-centering;
- supervised LDA projection;
- LinearSVC.

It failed Gate A with mean held-out-source delta `-0.03579`, worse on 11/13 sources.

The LoRA24 centered head is different:

- the encoder itself is adapted through LoRA;
- the representation is multi-depth Qwen hidden states;
- the centered head is trained jointly on same-prompt triplets;
- plain and centered heads coexist;
- centered evidence is a secondary signal, not the only representation.

So the LDA failure does not falsify LoRA24.

## FACT — important transfer-risk warning

The published public pool is not distribution-neutral. The notebook says its preprocessing removes exact duplicates and applies model-specific style filters derived from official train observations — for example, removing a punctuation mark from a model's public passages when that mark never appears for that model in host train.

This is not row-level validation leakage, but it deliberately makes public data more like the host train distribution. Given this repo's Phase 15 diagnosis — strong local gain that did not transfer to Public hidden data — that is a material risk, not a harmless detail.

The public notebooks themselves also report that stronger local CV can still drop on Public LB. Therefore the public recipe is evidence for the **mechanism**, not proof that its exact public-pool construction transfers to hidden data.

## INFERENCE

The strongest next experiment is **not** another sparse external concat and not another post-hoc centering transform. It is a clean two-stage neural attribution model where:

- Stage 1 tests whether LoRA adaptation on public data learns model signal that transfers across public sources;
- Stage 2, if authorized, adapts that representation to official train using strict fold-local prompt groups;
- final scoring evaluates plain-head OOF first, with centered-head complementarity reported separately rather than automatically blended.

The smallest credible backbone is 8B. The public notebook reports 14B did not improve standalone Stage-2 CV over 8B in its original experiment, so starting with 14B would add cost without establishing the mechanism.

## UNKNOWN

- Whether the current public pool's train-informed style filtering improves or harms true hidden transfer.
- Whether a source-held-out Stage-1 test predicts competition hidden transfer strongly enough to serve as a reliable gate.
- Whether the centered head adds honest repo OOF value when evaluated separately under the repo's permanent folds.
- Whether Public-LB gains from the current public notebooks will survive the final Private split.

## Required boundary for the next candidate

Before any new model result is observed, freeze a candidate contract with these constraints:

1. use Qwen3-8B only for v1;
2. Stage 1 must not see official competition rows;
3. Stage-1 transfer gate must hold out entire public sources, not random rows;
4. remove any prompt-bank overlap between Stage-1 train sources and held-out source using both available prompt identifiers/keys;
5. compare LoRA adaptation against a frozen-backbone/head-only control under exactly the same source split;
6. only if Stage 1 passes may official OOF run;
7. every Stage-2 fold initializes from the same frozen Stage-1 checkpoint and sees only official training prompt groups for that fold;
8. no prefit public Stage-2 checkpoint may be used for OOF;
9. plain head is the primary candidate score; centered-head gain and error complementarity are secondary diagnostics;
10. no Public-LB tuning loop and no Kaggle submission before the full promotion contract passes.

This audit authorizes **contract design**, not model promotion.