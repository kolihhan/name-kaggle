# NTL — Apex typography original vs disabled: source-faithful paired-readiness result

**Date:** 2026-11-19. **Decision:** executable comparison is now implemented and synthetic-source verified; **real paired BA cannot yet be measured** because actual same-row 18-way pre-Apex logits are absent from accessible scored output. No GPU, no Kaggle submission. Authenticated best Public BA remains **0.702 (#58514818)**.

## What was executed

- [GitHub Action #44487815258](https://github.com/kolihhan/name-kaggle/actions/runs/44487815258) **SUCCESS**. Exact `kolihhan/ntl-0702-apex-pinned-nov18` code SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc` is asserted before importing its original `pngy_residual.py` and `apex_postprocess.py`. **No substitute inference stage** or fake accuracy metric.
- [Executable paired evaluation / smoke script](../../scripts/ntl_apex_typography_paired_20261119.py) calls the unmodified `apex.apply_arrays` twice on otherwise identical inputs: `strength=.10`, `corrected=False`, original `penalty=.15` vs disabled `penalty=1.0`. Every other operation is preserved, including `refine`, `prompt_pair_adjustment`, GLM original-label preservation and ordered punctuation rules. It is a **test harness**, not a modified Kaggle competition Notebook.
- **Synthetic, source-faithful behavior:** Kimi top probability .60 vs GLM .35 plus text containing `" — "` yields original `glm-5.3` but no-typography `kimi-k3`. Kimi .97 vs GLM .02 remains Kimi in both; allowed Claude Opus 5 remains Claude Opus 5 in both. In this controlled two-class comparison, a disallowed top class can be overtaken by an allowed challenger if the original logit gap is below `ln(1/.15)=1.89712`. This condition is **not a universal threshold** for all ordered rules or distributions. Artificial scores **do not establish real model rescues/regressions**.
- [Read-only score-output census #44487147476](https://github.com/kolihhan/name-kaggle/actions/runs/44487147476): current scored 0.702 private kernel has only top-level `submission.parquet` and Python/smoke outputs (no `final-logits.npz`); frozen old 0.691 Oleg216 likewise has `submission.parquet` and logs without per-row logits. The planned `ntl-oleg216-final18-diagnostic-nov17` returned **inaccessible/possibly nonexistent private kernel** through authenticated API; do not claim its 18-way vectors exist. This matches historical [run #44077987686](https://github.com/kolihhan/name-kaggle/actions/runs/44077987686) which reported postV1 logits **NOT AVAILABLE**.

## Why this matters and what it does *not* prove

Earlier [Apex typography source audit](RESULT.md) found **23/216 historical true-model label exposures**, including **5 old-0.691 previously correct labels**, under the original marker/forbidden-class rules. The new exact-code test confirms **actual label flips are possible**, but none of those 23 cases has a trustworthy *current* before/after score vector to establish a realized correction or regression. Stage source preserves family mass under residual adjustment, but typography can change family decisions afterward.

**Do not report:** "23 misclassifications", "+5 rescued", or "+0.023 BA", or "remove typography improves 0.702". None was observed.

## Precise gate for the next real experiment

Only run **`--mode paired`** if four unmodified, same-row provenance-pinned inputs exist:

1. `data/train.csv` official 2,699 labeled rows used by the pinned Apex residual fit (no validation/test fit).
2. `query.csv` containing permitted **previously unused** 18-class labeled `id,text,model` with the exact required row identity and disjointness. The historical Oleg216 is spent and only diagnostic.
3. `pre_apex_original.npz` with `ids,labels,logits` from the **actual ensemble's `FINAL_LOGITS` stage** (the production final call has `corrected=False`, meaning Apex reads the **pre-GLM** score file, not `post-glm.npz`).
4. `current_post_glm.parquet` with `id,model` at the **exact moment before Apex**. These can differ from the argmax of `FINAL_LOGITS`.

The script checks IDs, label set, finite arrays and alignment, fits the original Residual on official training data, and computes original vs `penalty=1.0` same-row 18-class BA, rescues, regressions and cross-family changes, without publishing row-level text/truth. It is **not** a prospective clean test unless the input cohort itself passes provenance and reuse gates.

**Next choice:** Acquire/recover a complete authentic score snapshot from an already-authorized run, or instrument **the next separately justified scored execution** to persist these two pre-Apex stages. No new Kaggle GPU under today's work; no official submission of a no-typography patch based only on synthetic behavior.
