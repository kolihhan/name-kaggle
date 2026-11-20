# 2026-11-20 — Scored NTL 0.702 vs newer public 0.704: original-code mechanism and executable constraints

## Scope and outcome

**Executed** authenticated, read-only Kaggle API source and output audits; **NO** model GPU inference, training or competition submission. Owner verified frozen **0.702 Public (#58514818, COMPLETE)**, live Public top-10 cutoff **0.761** (gap **0.059** on 2026-11-20). [Authenticated scoreboard run](https://github.com/kolihhan/name-kaggle/actions/runs/44674191096) returned top 20, not owner exact rank.

The user specifically asked to work on the strong **0.702** champion, not further optimize historically low-performing Phase7/14/15 SVM pipelines. This audit checked **original scored source**, not the mutable public notebook title, and tested whether an honest new stage-level evaluation could be done without rerunning the expensive strong stack.

## 1 — Authenticated exact source diff

[Successful original-code read-only Action #44675311464](https://github.com/kolihhan/name-kaggle/actions/runs/44675311464), comparator public Kaggle notebooks downloaded with official API (NO inference or GPU). Script: [source-diff mechanism](../../scripts/ntl_top_public_kernels_vs_owner702_source_diff_20261120.py).

Scored **owner 0.702** `kolihhan/ntl-0702-apex-pinned-nov18`:
- 29 executable code cells.
- Entire executable source SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`; identical to original successful scoring receipt.

Latest inspected **public Lightning 0.704-title** source `lightningv08/0-704-glm-fable-routing-qwen3-8-27b-lora`:
- 29 executable code cells, SHA256 `b16429704ebc7e7baf1b81163b0ed604463a6ea0a81cf5a35782ea86e6a45c5b`.
- **27 of 29 exact matching code parts. Only two changed: `/kaggle/temp/v1/v1_stage.py` and `apex_postprocess.py`**.
- `v1_stage.py` GLM likelihood work selection changed from `order = argsort(-mass)` to:
  ```python
  fable_mass = np.exp(zl[:, labels.index('claude-fable-5')])
  priority = mass + 0.35 * np.minimum(fable_mass, mass)
  order = np.argsort(-priority, kind='stable')
  ```
  This prioritizes Fable/GLM confusion rows in a **budgeted GLM likelihood stage**, **not newly trained discriminative model weights**. Whether it increases valid GLM/classification accuracy in hidden tests is NOT established.
- `apex_postprocess.py` removes older text similarity `prompt_pair_adjustment` code and changes GLM winners' probability path to use aligned post-GLM logits; **this is a second independent behavior change**. Therefore 0.702 vs 0.704 cannot be causally credited to Fable routing alone.
- Crucial: **public notebook title/current code and any best-scoring version are not necessarily identical**, so `0.704` is an **author/notebook report**, not independently scored on our account; even if same version, **+0.002** is below previously observed identical-source Kaggle variability.
- Public `areebbb/0-704-fingerprint-sibling-specialis-reproduces` 39 code cells: 24 named shared parts, 18 identical, 6 modified, substantial visualization/analysis additions; uses same Fable priority adjustment, with other additional GLM/Fable override logic. Its notebooks contain illustrative OOF charts, not independently source-lineage-clean winner 0.702 OOF.
- `lumibee/ntl-kiro-v44-apex-grandmaster` 24 code cells, less clean comparison (9/18 matching parts) with runtime and packing modifications and a different pipeline lineage. Not evidence of an independent ≥0.761 recipe.

**Decision:** Copying all of 0.704 to chase a **+0.002** title delta has very low expected explanatory value. Do not submit or claim a breakthrough.

## 2 — Can we do paired 0.702 stage counterfactual without GPU?

[Original saved-output manifest, Kaggle API authenticated Action #44675693564](https://github.com/kolihhan/name-kaggle/actions/runs/44675693564) **SUCCESS**. The original scored private 0.702 notebook output lists:
- `submission.parquet` (the **Kaggle preview kernel run**, not the labeled hidden scorer outputs);
- script files such as `apex_postprocess.py`, `infer.py`, and three `smoke*.bin` files.
- **NO exported full-class `final-logits.npz`, `post-glm.npz`, component probability cache or 18-way matched stage array**.
- Final official scoring runs in a separate hidden evaluation environment; the saved preview output does not expose hidden-test logits or truth.

Thus an honest labeled, same-row *champion pre/post-Apex* real counterfactual would require **a fresh full-strong-model inference run on a suitable labeled cohort**, or earlier genuine original intermediate arrays that have not been located. Re-labeling hard predictions as probabilities, or treating official TRAIN inference from pretrained assets as independent OOF, is invalid.

## 3 — Can the recently updated Open Pool supply a clean after-training 18-class cohort?

[Successful SHA-locked CPU metadata-only Action #44675959849](https://github.com/kolihhan/name-kaggle/actions/runs/44675959849); source [script](../../scripts/ntl_0702_posttrained_18label_holdout_capacity_20261120.py). Current public dataset `crtproduction/name-that-llm-open-pool`, SHA256 `7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492`, **231,036 rows**.
- Publisher `gen_date` parseable on **33,603** rows. **Zero** dated on/after **2026-11-16**, so also zero on/after 11-18/11-19, despite the current snapshot having more rows than 11-08.
- `gen_served_model` is nonempty on **85,470**, syntactically matching publisher label in **85,470**; these are **publisher claims**, not independently verified model-route identity.
- Nonempty publisher `prompt_canon` **225,110**.
- Chronological post-training/date-disjoint cohort in this version: **0 labels / 0 candidate rows**, not a prospective 18-class eval. Do not infer file update or extra rows means **new generator calls** after pretrained champion weights were trained.
- Date and metadata tests are necessary, not sufficient: even a recently dated row could reuse training prompts, and publisher can misstate route/license.

**Decision:** do **not** launch expensive private 0.702 GPU rerun against the older, previously inspected Oleg or another contaminated pool and call it clean transfer validation. Existing 216-row Oleg data were repeatedly reused and likely differ from official host style. Neither that nor official TRAIN offers a genuine held-out 0.702 evaluation by itself.

## 4 — Scientific priority now

- **Stop** more Phase7/Phase15 old-OOF tuning, source-only SVM leaderboard submissions, k=2 resampling, Apex typography sweeps and 0.704 one-shot imitation solely for score.
- The strongest **near-term testable bottleneck** is an independent, **newly collected** (not merely newly published) full 18-label prompt-disjoint cohort with verified generator route/version, named source and genuinely untouched prompts; only then run **one full 0.702 inference** saving pre-GLM/post-GLM/pre-Apex/post-Apex **full 18-way logits and fixed ID-aligned final predictions** to get per-class errors, rescues/regressions, and intervention costs. Legitimate source authorization matters for training, while read-only diagnostic evaluation on public data follows source/data rights.
- This benchmark must not be silently supplied from the already repeatedly consumed old Kaggle training set. If a genuine new holdout cannot be acquired within time and budget, **mark full strong-model candidate uplift UNKNOWN** and prefer preserving score 0.702 over publishing unfounded private-medal claims.
- To close **0.059 BA** to currently observed Public top10 requires a materially broader new signal than changing GLM/Fable sorting and Apex soft typography, based on current best public evidence.

**Actual outcome:** source/score comparison and holdout feasibility done, no GPU, no submission, no new Public points. This is a **negative feasibility diagnosis**, not an accuracy experiment.
