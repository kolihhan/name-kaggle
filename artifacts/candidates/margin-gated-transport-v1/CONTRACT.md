# Margin-Gated Transport v1 — no-new-training one-shot official submission

**Date:** 2026-11-11, **FROZEN before new Kaggle submission and score**.

## PIC authorization and honest scope

The user explicitly requested **a new official Kaggle submission** after the earlier independent-18 transfer evidence gate was blocked. This is authorization for **one experimental submission of a materially changed inference rule**, not permission to train new models, tune on Public LB, bypass Kaggle limits, or claim scientific generalization. The existing best remains **0.679** (own submission `58210835`, an attributed external public reproduction). **One new candidate may score lower**.

Kaggle Name That LLM balances recall across 18 true classes; the published Public and hidden Private label mixtures can differ. Our scored baseline already performs unsupervised global Sinkhorn-like class-marginal correction for every document using the predicted marginal + competition-train prior. It can alter **high-margin** predictions even when the predicted class mixture is ambiguous. We can test a more conservative, data-agnostic correction that trusts strong passage-specific evidence while retaining full correction for ambiguous passages.

## Exact model/algorithm (source-frozen)

Start from our owned **private scored** `kolihhan/ntl-ozertuu-0683-guarded-no-train`, NOT a changed public upstream Notebook. Pin executable **code-cell SHA256 `28fd128b745f0d95b70f79ea9b2c546d981edf9be08207cbf71a2d9e80ee1c86`** and original Kaggle source metadata: 20 datasets, eight kernels, six models, all official competition inputs, no internet, T4. Guard original three forbidden training paths: supervised Gaussian fit, Atlas train mode, core `--train-final`.

**Only scientific code delta:** Within the pre-existing `core_math.py` module's `balance_logits(P,train_labels,strength=.30,target_mix=.52)` (in existing Notebook cell 9), replace the *uniform per-row* balance blend:

```python
total = (1 - strength) * np.log(P + 1e-12) + strength * np.log(Q + 1e-12)
```

with **class-independent margin-adaptive shrinkage**:

```python
ordered = np.sort(P, axis=1)
margin = np.clip(ordered[:, -1] - ordered[:, -2], 0.0, 1.0)
row_strength = float(strength) * (1.0 - margin)
base_log = np.log(P + 1e-12)
total = base_log + row_strength[:, None] * (np.log(Q + 1e-12) - base_log)
```

- `P` = original 18-class pre-balance per-row probability distribution, `Q` = original 18-class Sinkhorn-adjusted probabilities. Both are **unlabeled test-time quantities**. `margin` between top-1 and top-2 is in [0,1].
- Retain pre-existing `strength=.30`, `target_mix=.52`, GSQ share `.15`, native share `.43`, typography evidence, all neural readouts, saved weights, grouping, and remaining code exactly. **No new tuned numeric hyperparameter**.
- At tied top classes, correction equals champion; at certainty margin 1, correction tends to zero. Continuous effect elsewhere. This is **not** a class-specific rule, trained error fixer, label-prior estimation, or per-family gate.
- The intervention acts in each of five native-fold finalization calls. It does not alter the prefit data or produce new model weights.
- Physical source license/credit retained; the source is an attributed Ozertuu/Lightning-based pipeline with *our own small inference-only margin gate*, **not** a new trained model architecture.

## Risks and preflight

- The incumbent's global balancing may improve macro recall, even on confident examples; conservatively reducing it could **lower** 18-label balanced accuracy, especially minority/sibling recall. Margin confidence can be miscalibrated under domain shift; it is not ground-truth correctness.
- This candidate is not validated on an unused complete 18-label benchmark (none currently qualified). **Official submission is exploratory competition evidence**, not independent proof of transfer. Previous 347 external and Kimi 80 examples are already inspected and **must not** be used to set parameters or select follow-up patches.
- Preflight checks frozen owned source SHA, exact single replacement of balance expression, no other executable code delta; preserved three no-training guards; original 18 labels, input metadata, valid Notebook syntax, and synthetic purely mathematical controls proving the new gate differs from original. **Synthetic tests establish implementation only**, not expected score.
- Strict duplicate guard: refuse if destination Notebook slug already exists or Kaggle submission description exists. Never retry submit after receiving an ID. No repeated seeds/weights/variants or score-based parameter changes. Preserve the 0.679 champion unchanged.
- Private Notebook, internet disabled, T4, prefit public assets, no official raw passages in Git/Action artifacts or logs. Preview format must be `submission.parquet` containing id/model. Hidden rerun subject to 9h limit; if rejected, report actual reason rather than train or silently resubmit.

## One-shot plan and result acceptance

1. Create one private Notebook `kolihhan/ntl-margin-gated-transport-v1` using only original upstream notebook code + exact snippet patch.
2. Run standard Kaggle public fixture preview; verify original output `submission.parquet`.
3. Submit that Notebook version 1 **once**, description:
   `MGT-v1 unlabeled confidence-adaptive prior transport; frozen 0.679 prefit baseline; no new training 20261111`.
4. Record **new official submission ref, status and eventual publicScore**; do not claim a score until Kaggle returns numeric `COMPLETE`. If score <=0.679 keep champion unchanged. If >0.679 label Public improvement as exploratory and await separate Private confirmation.
