# NTL: new 0.691 official champion and bounded proof of expert-cache shape

Verified MYT 2026-11-16 (Kaggle read-only tool run 2026-11-15 17:55 UTC).

## Actual official result

- **Own official `name-that-llm` submission `#58375806`**, `SubmissionStatus.COMPLETE`, **Public BA 0.691**, official description `Lightning0694 pinned V2 no-fit research bet 20261115`.
- Previous exact scored best `#58295802`: COMPLETE, Public BA **0.683**. Genuine official **+0.008 absolute**, ~1.17% relative. Gap to **0.700** is **0.009**.
- Exact authenticated account ledger [read-only Action #43743764085](https://github.com/kolihhan/name-kaggle/actions/runs/43743764085). Submitted from pinned user's private Lightning V2, no new training; submission [#43692500733](https://github.com/kolihhan/name-kaggle/actions/runs/43692500733). Upstream author notebook title `0.694` is **not** our score.
- Reproducing a pretrained public ensemble with execution/no-supervised-fit guards is **attributed reproducibility**, not an independently learned 0.691 architecture. Public BA is an external competition outcome, not a clean independent source/prompt transfer or private BA estimate. Do not train or tune thresholds against this public score.

## Competing mechanism explanations, not causal attribution

[Read-only existing output plus source inventory #43744050448](https://github.com/kolihhan/name-kaggle/actions/runs/43744050448):
- 0.691 Lightning: **50** code cells / **23** dataset inputs / **6** model inputs / **9** source-notebook inputs, modules including `ensemble.py`, `v1_stage.py`, `core_worker.py`, GSQ, family and native-fusion helpers.
- Prior 0.683 Areeb8: **19** code cells / **17** dataset inputs / **8** model inputs / **8** source-notebook inputs; different collection of stages, including `gsq_stage.py`, `comp_stack.py` and native-Q8.
- This is a *many-factor* comparison. Neither source structure nor the +0.008 between submissions isolates gains from a single expert or any parameter. Treat the current 0.691 combination as frozen.

## Saved-cache ablation investigation: genuine feasibility limit

- 1,200-row successful existing inference SHA256 `926dffc6ab0909f63767e8fcec7ca9488f7d1dc7c49656b0eaa8cf50064e0826`. It retained `v1/flash_scores.npz` and `v1/glm53_scores.npz`.
- [NPZ read-only structural inspection #43744548740](https://github.com/kolihhan/name-kaggle/actions/runs/43744548740) **SUCCESS** using safe NPY *header parsing only*, no unsafe object unpickling. Both NPZ have identical field schema:
  - `ids` object dtype shape **[240]** (unsafe to deserialize blindly)
  - `n_prefix` int64 **[240]**
  - `offsets` int64 **[241]**
  - `scores` float32 **[27869,4]**
  - `tokens` int32 **[27869]**
- Exact archive SHAs: Flash `e0eafd51e12124807d7d3dacb463ee54e3e7ff61ec7736516085732dab6e510b`; GLM53 `6f0c041c089d5bf360a96dadef32d02f40f49a96bd018761352681abfa0297c3`.
- An earlier [safe `np.load(allow_pickle=False)` attempt #43744212202](https://github.com/kolihhan/name-kaggle/actions/runs/43744212202) stopped with `ValueError: Object arrays cannot be loaded when allow_pickle=False`. We deliberately did **not** set `allow_pickle=True`; this is a security boundary, not missing computation.
- These are **token-level 4-channel scores on a selected 240-row subset**, *not* complete `[1200,18]` per-row ensemble scores. **No valid component-off ablation has been run and no marginal correctness improvement can be claimed** without safely reconstructing id-to-input mapping, reading aggregator semantics, and paired final logits on the same rows. Do not invent them.

## Prospective next test

1. Freeze and preserve 0.691. Research source-stage relation by static inspection, especially how token scores are converted into per-row candidate odds. Determine whether a clean, reproducible, source-independent full 18-way paired test is feasible.
2. If not, stop ablation and change research/data representation rather than optimize on the 1,200 already-spent train examples.
3. Only consider another official submission after a **distinct hypothesis** passes a frozen evaluation; do not spend leaderboard slots on public titles or micro-blends.

No GPU, new training, or competition submission from this source/cache audit.
