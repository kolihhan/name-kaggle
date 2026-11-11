# WeMM hidden-rerun failure and actual full-path smoke — 2026-11-12 MYT

## Official terminal result — NOT a score

- Competition ref: **`58244483`**. [Authenticated account status job #43070833655](https://github.com/kolihhan/name-kaggle/actions/runs/43070833655) returned `SubmissionStatus.COMPLETE` but **empty Public and Private scores**.
- [Authenticated single-submission detail job #43071058170](https://github.com/kolihhan/name-kaggle/actions/runs/43071058170), job `130698716843`, returned **`Error: Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset`**. This is a failed hidden competition execution, **not a valid scored `COMPLETE`**.
- Own best authenticated official Public BA remains **0.679**, ref `58210835`. `58233227` MGT and `58167751` Lightning were still PENDING in the same read-only status check. No new submission has been created by this investigation.

## Root cause established and unresolved

**Proven preflight defect — preview skipped ALL meaningful full-model inference.**

- Read-only pull of the actual private WeMM Notebook [static audit #43071497601](https://github.com/kolihhan/name-kaggle/actions/runs/43071497601) revealed 43 executable cells, approximately 304,649 characters of code.
- Code cell 1 defines `emit_known_preview`: it matches exact public fixture IDs, byte lengths and text fingerprints; when matched, it writes a **precomputed** 3-row `submission.parquet` and tags `model_weights_loaded: False, hidden_runtime_verified: False`.
- Code cell 42 runs the genuine `ensemble.py` pipeline **only under `if not preview:`**. The 3-row preview's `COMPLETE` never loaded the actual ensemble, native Qwen/WeMM embeddings, or other large assets. The `assert len(frame)==3` is confined to the matched preview function; this assertion is **not** proven to be the hidden exception.
- The hidden rerun uses a different test file and enters the full inference path, which may throw for model load, GPU memory, source path, runtime guard, shape, timeout, or other conditions. **The exact hidden exception/line remains unknown.** Kaggle gives a generic error by design; do not fabricate a traceback or conflate static suspicion with confirmed root cause.

**Dependency hypothesis test — original 'asset directory missing' hypothesis NOT supported.**

- [Metadata-only full listing #43072187230](https://github.com/kolihhan/name-kaggle/actions/runs/43072187230) traversed **six pagination pages / 1,056 files** in `lightningv08/ntl-olmo2-native24-final-inference-assets`.
- Both native embedding directories were present, with **42 entries each** under `ntl-native-q8-multidepth-fivefold-0679/` and `ntl-native-wemm4-fivefold-0679/`. Earlier first-page-only file lookups incorrectly appeared empty; those negatives are superseded.
- File-list existence is not proof every weight loads, fits in GPU memory or has a valid runtime schema.

## One isolated PRIVATE diagnostic created, no official submission

- [Creation action #43072428946](https://github.com/kolihhan/name-kaggle/actions/runs/43072428946), job `130703543968`: **SUCCESS**.
- Private Notebook **`kolihhan/ntl-wemm-fullpath-3row-smoke-v1`**, version 1, initial Kaggle status `QUEUED` at **2026-11-11 17:53:23 UTC**. The notebook is *not* a competition submission, and no metric has been produced.
- Exactly **one executable source edit** against the already-owned private WeMM Notebook: replace `preview = emit_known_preview(DATA,None,OUTPUT)` with `preview = False` to force the actual pipeline on the publicly visible 3-row input. All 43 source code cells and 4 no-new-supervision guard checks were verified before publishing the private diagnostic.
- Notebook remains private, internet disabled, T4, **60-minute private runtime cap**, no new model training, no hidden test labels, no GitHub storage of official text/weights. GPU quota read at launch: **13.13 hours remaining**; the action itself pushed the Notebook successfully but did not run GPU inference on GitHub.
- Source combined code SHA256 `bdbbba6ac34ffa114f3c1c9d2c14299821e38d20f25842104d520ce697359979`; smoke variant `0a45fae0284eec2bf5d723b48035c7b37aa9e87f936459c7635d64fdc4198bcf`.
- **No unverified future outcome:** `QUEUED` is not `COMPLETE`, and the smoke may itself error or exhaust its bounded runtime. After completion, inspect the **actual private Notebook user-facing logs**, not hidden competition labels; if they contain a deterministic exception on the same inference path, diagnose and only then consider a new, separately frozen corrected submission. Do not automatically resubmit from this evidence.

## Separate scientific evidence recorded in this turn

The [114 paired sibling feature diagnostic](ntl-sibling-signals-paired114-20261112.md) was completed descriptively on *already-consumed* official train, with [reproducible read-only auditor](../../scripts/audit_official_sibling_signals.py) and [6/6 passing local tests](../../tests/test_audit_official_sibling_signals.py). Six comparisons passed descriptive BH q<.05; **these are not model accuracy, independent-transfer validation or permission to tune weights.**

**Overall decision:** WeMM is an **unscored runtime failure**, not evidence for or against embedding diversity. The operational blocker is absence of a real full-model pre-submit smoke; scientific model-class and source/protocol questions remain separate. No PPO, new training, coefficient retuning or competition resubmission was performed.
