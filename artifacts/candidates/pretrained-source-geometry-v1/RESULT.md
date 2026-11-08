# Frozen pretrained source-geometry factorial — completed CPU result

**Date:** 2026-11-08 MYT. **Scientific verdict:** DeTeCtive H-M **REJECT**; LUSR H-R **REJECT**. No promotion, new Kaggle Notebook or submission. This records [reproducible successful GitHub Actions run 42454362197](https://github.com/kolihhan/name-kaggle/actions/runs/42454362197), commit `c07916d4ed5425c48f383361535d2c6e90ec4b29`, with 20/20 focused tests passed, all pinned input/model SHA guards passed, 3 frozen checkpoints loaded, and both evaluations executed using only CPU. The full aggregate JSON artifacts are downloadable from that run (artifact ID **13608485634**); not all per-source numbers are reproduced in this prose summary.

## Real grouped official training OOF (18 labels, 2,699 rows)

| Arm | Exact-label grouped OOF balanced accuracy |
|---|---:|
| Original own LoRA24 hard-OOF anchor | **0.5467611235** |
| SimCSE centroid, published 512 pooling B0 | 0.108889 |
| SimCSE density, published pooling B1 | 0.122593 |
| SimCSE centroid, masked pooling B0 | 0.116667 |
| SimCSE density, masked pooling B1 | 0.130746 |
| DeTeCtive centroid, published pooling M0 | 0.078889 |
| DeTeCtive density, published pooling M1 | **0.092222** |
| DeTeCtive centroid, masked pooling M0 | 0.079630 |
| DeTeCtive density, masked pooling M1 | 0.087407 |
| SimCSE + linear R0 | 0.234860 |
| LUSR + same frozen linear R1 | **0.3511981109** |

**H-M rejected** with own-LoRA paired prompt bootstrap 97.5% CI **[-0.478610, -0.430181]**; all principal promotion gates false, except within-DeTeCtive density-vs-centroid +0.005 improvement. Real masked mean does not rescue DeTeCtive; SimCSE density remains higher. The frozen source-transfer and error checks are in the aggregate artifact.

**H-R rejected** with own-LoRA paired prompt bootstrap 97.5% CI **[-0.217822, -0.172035]**. LUSR genuinely beats its matched SimCSE linear control by +0.11634 OOF, and the mean external source-gain gates were positive, but **does not beat LoRA24**; provider and label guardrails fail. This is **evidence of representation signal but not a competitive standalone model**. The study compares released encoder packages, not a causal ablation isolating their pretraining objectives.

## Engineering/evidence boundary

- Experiment runner finished **SUCCESS** at 2026-11-08T07:07:27Z; 20 focused tests passed; pinned runtime, official folds, public external pool, original LoRA24 archive, and three pretrained weights passed validation. Encodings completed for SimCSE (890s), DeTeCtive (891s) and LUSR (891s) on CPU.
- All relevant predictions and model caches were kept in temporary Actions runner workspace, with **only aggregate JSON** uploaded. These private row-level caches are not retained by the aggregate archive; do not claim they can be reused after runner shutdown.
- These are **official-training grouped OOF metrics**, not competition Public LB scores; the existing own-account **Public best remains 0.668 from a reproduced external solution**. No new submission ID exists from this experiment.
- The [conditional-denoising18 frozen proposal](../conditional-denoising18-v1/CONTRACT.md) is the next materially different hypothesis. It has NOT been implemented, executed, or validated by this workflow. Its own training and transfer gates apply; do not claim that it will beat 0.668.

**Decision:** Close both evaluated frozen geometry routes. Next research target is trainable, **context-conditioned source residual learning**, not another frozen-encoder geometry or TF-IDF reweighting adjustment. Actual execution requires an independent new stage with tests, fit, controls and source transfer, followed by official OOF and Kaggle submission only if promotion gates pass.
