# Frozen 0.715 Pre/Post GLM Stage Trace — source-only READY, actual inference NOT run (2026-11-23)

## Verified execution

- [GitHub Actions #45188580008](https://github.com/kolihhan/name-kaggle/actions/runs/45188580008) **terminal SUCCESS**.
- **Authoritative source SHA**: `202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8`.
- Original 32 cells **byte-for-byte preserved**, same all **9 model**, **26 dataset**, **10 kernel** mounts and GPU/runtime metadata.
- Scorer's `infer.py` invocation at original notebook **cell 25**; own GLM `v1_stage.py` call at original **cell 29**. Inserted one sidecar immediately before original cell 29, appended one read-only summary after all original cells.
- `NTL_0715_STAGE_TRACE_SYNTHETIC=PASS`: fake 3-row 18-label setup tested one sibling-only and one cross-family transition, source SHA unchanged, snapshot duplicate refusal, and failing closed on ID/logit misalignment.
- `NTL_STAGE_TRACE_BUILDER_SMOKE=PASS`. Actual logged builder result `PRIVATE_UNPUSHED_CPU_STATIC_BUNDLE_VALID`. No private Kaggle kernel pushed or launched; transient exact source and private notebook scrubbed in workflow cleanup. This is *not* a real GPU or performance test.

## Algorithm sidecar (what it does and does not do)

`scripts/ntl_frozen0715_stage_trace_20261123.py`:
1. Reads the already-written frozen source `/kaggle/working/submission.parquet` **just before** GLM V1, matches row IDs to existing `/kaggle/temp/final-logits.npz` (18 classes); saves the original pre-GLM labels only under `/kaggle/temp`, not publicly committed.
2. After the existing Notebook finishes all original cells, reads its untouched `submission.parquet`, classifies changed predictions as **cross-family** or **within-family sibling**. Checks identical row order/IDs, validates 18-class logits alignment, reports whether native V1 report existed and writes **aggregate counts only** to `/kaggle/working/ntl_stage_trace_aggregates.json`.
3. Never edits `submission.parquet` or `final-logits.npz`, never reads true labels, never computes accuracy/BA. If snapshot or alignment fails, logs `NOT_VERIFIED` and original scorer continues unchanged (diagnostic fail-open, model-output no-change).
4. This is not a stand-in for separate strong model confidence calibration or a post-GLM 18-class probability array. Pre-GLM `final-logits.npz` is **not** the final post-V1 posterior. No extra expert or GLM likelihood run.

`scripts/build_ntl_0715_stage_trace_private_20261123.py` builds exact scored owner code plus only the two sidecars on ephemeral storage. The fixed source SHA prevents silently porting the instrumentation onto an unrelated newer 0.719-titled Notebook.

## Cost and evidence decision

- Public scored strong 0.715 is frozen and intact.
- Notebook's frozen GLM `V1_DEADLINE_H=8.8` in a 9-hour official limit. Running its true full-GPU end-to-end path merely for *unlabelled* post-stage changes could consume nearly the same resources as a competition rerun; a cheap three-row preview won't establish the real hidden-size stage routing or full-18 BA.
- Independent 18-label cohort has **not** passed provenance/contamination gate. New publisher C1/C2 are 12/18 only and specifically exclude both GLM versions, Kimi and Qwen. Thus a direct C1/C2 score would not validate the suspected missing-family bottleneck.
- **Decision: keep this as ready-to-run private sidecar; DO NOT dispatch expensive GPU inference or another Kaggle official submission merely to get unlabelled counts.** Reconsider when a meaningful independent matched source (or already-available trustworthy full-output logs) is available, OR when a separate explicit time-budget approval and fully frozen diagnostic sample justify the cost.
- This is a **mechanism observability artifact**, not a new competitive 18-way model and not evidence of accuracy gain. Public best **0.715** unchanged.

## Exact files

- [Sidecar stage trace source](../../scripts/ntl_frozen0715_stage_trace_20261123.py)
- [Preserving two-cell builder](../../scripts/build_ntl_0715_stage_trace_private_20261123.py)
- [Verified CPU workflow](../../.github/workflows/ntl-0715-stage-trace-cpu-20261123.yml)
