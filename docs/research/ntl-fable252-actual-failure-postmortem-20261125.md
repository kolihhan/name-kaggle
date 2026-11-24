# Fable252 actual ERROR: implementation NameError postmortem and recovery

**Status as verified at 2026-11-25 01:28 MYT:** Kaggle private kernel `kolihhan/ntl-fable-preservation-252-20261124` ended `KernelWorkerStatus.ERROR`; **no** valid 252-row BA, rescues or regressions. NO evidence against the scientific `fable_return` hypothesis.

## Authentic observations
1. [Kaggle status/log forensic Action #45447729382](https://github.com/kolihhan/name-kaggle/actions/runs/45447729382) verified terminal `ERROR`, zero `NTL_FABLE_252_PREPARED`, zero `PRE_APEX_SNAPSHOT`, zero `FABLE_PRESERVATION_REAL252`, no output aggregate. Kaggle log was reachable but machine regex missed embedded exception and notebook output not materialized.
2. [Full failed log / source inspection #45447910487](https://github.com/kolihhan/name-kaggle/actions/runs/45447910487) retrieved 9,822 bytes failed-kernel logs, confirmed the log contained **`Traceback` and `NameError`**, and checked uploaded 38 code cells including the injected helper and first data preparation. No output result.
3. First-shot actual original scored 0.718 source audit previously recorded the first-ever `DATA` assignment in notebook code **cell 20**:
   `DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))`.
   The prelaunch builder accidentally **REPLACED this first assignment** with:
   `DATA = str(ntl_fable_probe.prepare(DATA))`.
   That is a deterministically invalid first reference to unbound `DATA`, causing `NameError` before the diagnostic could print PREPARED. This is a definite code defect and the **strongly supported proximate failure cause**. Caveat: archived Kaggle NameError statement was not obtained verbatim, so do not claim the full failed-kernel traceback explicitly named `DATA`.
4. [CPU executable regression #45448903613](https://github.com/kolihhan/name-kaggle/actions/runs/45448903613) proved the **v2 patched assignment executes successfully** with a fake `train.csv` and real Python `exec()`. The corrected generated cell retains the original `DATA = next(...)` and then calls `prepare(DATA)`, followed by setting the competition rerun environment before `FAST_COMMIT`. Original block and method are unchanged. This executed regression PASSED.
5. The **full donor notebook rebuild did not complete** because an authenticated `kaggle kernels pull kolihhan/ntl-evgen-forensic-eda-repro-20261123` unexpectedly returned HTTP **404 GetKernel** in that same Action. This is a separate input-access blocker; it was previously available at the original launch. Do not infer whether the private kernel was deleted, visibility or authentication changed, or the API is transiently failing without additional evidence.

## Recovery changes
- Branch: `research/ntl-fable252-nameerror-recovery-20261125`.
- Source: `scripts/build_ntl_fable_preservation252_private_20261124.py`.
- Patch: keep original data-directory discovery line, add a second `DATA = str(ntl_fable_probe.prepare(DATA))` line, preserve original `FAST_COMMIT` semantics; do NOT use an undefined `DATA` on assignment RHS.
- New executable CPU regression in `smoke()` proves this exact first-use case; originally smoke only checked text/magic syntax and missed the NameError.
- New intended private diagnostic ID `kolihhan/ntl-fable-preservation-252-v2-20261125` distinct from failed artifact; **it has NOT been submitted**.
- Do not blind-push a stale or alternate private Kaggle kernel. First ensure original 0.718 source can be pulled with exact notebook SHA256 `74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56`, CPU rebuild all 61 notebook cells, verify actual first-assignment runtime, and inspect full resource availability. Only then schedule one resource-budget-gated private non-submission rerun.

## Scientific interpretation
This is an **ENGINEERING FAIL before experiment**, not a negative GLM/Fable mechanism result. The owner Public best remains an attributed 0.718 reproduction, not the original candidate improvement. Pending original hypothesis deserves a valid test but is not yet supported.
