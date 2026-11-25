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


## Recovery execution receipt — 2026-11-25 11:42 MYT

- Previous source 404 was overcome **without** using an unverified latest public fork. Kaggle's historical internal source service returned 38 archived revisions of Evgen's public notebook. Original freeze log (`NTL_EVGEN_*_CELL`, 2026-11-23) has 58 notebook cell records; [version-by-version CPU comparison #45541411817](https://github.com/kolihhan/name-kaggle/actions/runs/45541411817) found **public version 36 matched 58/58 recorded cell prefixes and original lengths**, whereas versions 34,35,37,38 differed. Historical v36 source SHA256 `5988d64922650bf94490810effb7278907b301c07f581efbc6d4865d84e63d52`. The old audit truncated six code-cell bodies to 13,000 characters and a long markdown cell, so full bytewise equality to the original downloaded 0.718 `.ipynb` cannot be asserted; original downloaded SHA remains `74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56`.
- [Full source CPU recovery/61-cell preflight #45541570603](https://github.com/kolihhan/name-kaggle/actions/runs/45541570603) SUCCESS: recovered version 36 SHA matched, 58/35 original notebook structure matched, original **9 model / 26 dataset / 10 kernel** mounts matched, `DATA` first assignment executable regression passed, 61 final cells and mutated cell order compiled. Original `V1_DEADLINE_H=8.8` preserved (the prior shortened 5.8-hour deadline could incorrectly suppress GLM likelihood if native inference takes its allocated 7 hours).
- [Real one-shot private Kaggle GPU v2 #45541703599](https://github.com/kolihhan/name-kaggle/actions/runs/45541703599) SUCCESS (submission **of the private experimental notebook only**): Kaggle quota remaining **20.59 GPU hours** before start. The private, offline, no-training diagnostic `kolihhan/ntl-fable-preservation-252-v2-20261125` was **accepted**, and worker reported `KernelWorkerStatus.RUNNING` at 2026-11-25 11:41:46 MYT. Timeout is 32,400 seconds (9 hours), original 8.8-hour GLM deadline is unchanged. No official competition submission was made.
- **Scientific status:** `RUNNING_NOT_SCORED`. There are no 252-row paired rescue/regression results yet, no proof of Fable protection benefit, and no new official Public score; account reference remains the externally reproduced **0.718**. The old failed v1 is not a negative scientific result. v2 is now the single valid active recovery experiment. No extra GPU duplicate is warranted.
