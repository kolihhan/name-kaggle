# NTL 0.691 Lightning V2 — pinned prefit lineage audit (2026-11-18 MYT)

## Decision: NO HONEST OFFICIAL-TRAIN OOF FOR FROZEN INCUMBENT

The exact scored **0.691** Lightning system is an attributed third-party frozen ensemble, not a fold-retrained model. **Do not run its published/prefit package over 2,699 official train rows and report the result as OOF**, even if fold IDs are attached. The pipeline contains a directly observed trained-artifact declaration that reports use of official data; exclusion of each held-out prompt from all fitted modules is not demonstrated. This is an evaluation-lineage restriction, **not an accusation of Kaggle competition-rule violations**.

## Actual pinned source and execution evidence

- [CPU-only Actions #44221042287](https://github.com/kolihhan/name-kaggle/actions/runs/44221042287), SUCCESS. Workflow: `.github/workflows/ntl-lightning-prefit-lineage-cpu-20261118.yml`.
- Pulled owner notebook `kolihhan/lightning-0-694-authentic-no-fit-inference/2`, not a mutable unversioned author Notebook.
- **Exact code SHA256** `cae1074e9cea0b3c5f03e221b93e622858c4cbd8183309ba0a9a02e0ee73558c`; matches the 2026-11-15 original scored V2 one-shot contract. 50 executable code cells, 37 generated Python modules.
- Metadata mounts: 23 external datasets, 6 base model refs, 9 upstream notebook refs; these are **source references, not 38 independently audited training datasets**.
- Both `NO_NEW_TRAINING` Gaussian/Atlas guards present; static downstream data fitting can include unsupervised TF-IDF on inference inputs. **No supervised model training or Kaggle official submission launched by this audit.** GPU = 0, official training/test rows downloaded by audit = 0.
- Notebook contains **the exact declaration assertion**:
  ```python
  assert complete["rows_seen"] == {"external": 119899, "official": 5398}
  ```
  This is a **self-reported trained-artifact contract inside the exact pinned current Notebook**. The counts are recorded training-row exposures, not independently verified **unique** documents; 5,398 equals 2 × 2,699, so do not treat it as 5,398 independent competition examples. It does **not** prove every checkpoint saw all competition prompts, but it defeats any claim that *all components were demonstrated held-out-safe*.
- Static source census: `trained_artifact_contract` 1, `rows_seen` 2, `train.csv` 3, `GroupKFold` 0, `StratifiedGroupKFold` 0 (textual occurrences). Literal search counts are **not proof** of absence of historical upstream cross-validation: only fold-specific trained-data manifests/checkpoint hashes could establish that.

## Confidence boundaries and competing explanations

- **FACT:** Source declares at least one fitted artifact has official training-data exposure. No per-fold model provenance or exclusion ledger is available for the complete 0.691 system. Therefore fake OOF is disallowed.
- **FACT:** 0.691 is a genuine previous Kaggle **Public** score (official submission #58375806), not affected or re-scored by lineage inspection.
- **UNKNOWN:** Whether any individual trained module could be validly reused for some folds; degree of direct prompt overlap for each external source; whether training-source/channel artifacts explain historic Phase7→15 CV/Public reversal.
- **Do not infer:** the Public gain came from contamination, that the submitted system violated Kaggle rules, that all 2,699 official rows were uniquely seen by every checkpoint, or that the Phase15 external training support *caused* observed per-class changes.

## Research decision and stop rule

1. **NO RUN:** no official-train Lightning "OOF" inference, no 0.96 historical train accuracy used as selection evidence, no 4-model CV↔Public ranking, no retraining its multi-component upstream ensemble just to manufacture a local metric.
2. **Only reopen genuine OOF if** each fitted checkpoint/encoder/SVC/Gaussian/specialist has a verifiable per-fold official ID/prompt exclusion manifest and exact weight/dataset lineage. Lacking that, mark incumbent OOF `N/A — fit lineage exposed`.
3. **Alternative path:** a genuinely untouched, source-disjoint, 18-label, served-version-verified same-row external panel **and** one frozen incumbent run could provide a comparable paired test; currently no admitted full-18 panel satisfies that contract. Avoid more Oleg/external mechanism-search loops.
4. No new model, training, Kaggle submission or score change. NTL Public = **0.691**; Gemma Public = **0.18**, keep Gemma on low-investment lane **b** without development GPU or polling.

## Source controls

- Original official result: [one-shot V2 receipt / code fingerprint](2026-11-15-pro-lightning0694-authenticated-one-official-submission.md).
- Earlier V47 source independently listed official training counts: [mechanism audit](ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md). The present result now pins the same declaration directly in the scored V2 source.
- Grouped OOF vs Public mismatch: [ruler audit](ntl-ruler-2699-prompt-oof-reliability-20261118.md). It does not establish causation.
