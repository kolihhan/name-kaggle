# NTL #58268344: hidden rerun failed; true fullpath CUDA evidence

Date: 2026-11-12 (Asia/Kuala_Lumpur).
This is a diagnostic and corrective execution record, not evidence that accuracy improved.

## Confirmed observations

1. Official NTL code submission `58268344` returned **COMPLETE but no Public or Private score**. The authenticated per-submission details identify **an unhandled error during the hidden rerun**, so a successful submission receipt did not mean a valid scored submission. [Exact authenticated details](https://github.com/kolihhan/name-kaggle/actions/runs/43206823389).
2. The private source Notebook's successful 3-row preview used `preview = emit_known_preview(DATA,None,OUTPUT)`, a known-preview fixture branch; the real inference occurs under `if not preview`. Its success **did not test full-model inference**. [Static audit](https://github.com/kolihhan/name-kaggle/actions/runs/43206823389).
3. An isolated version of the own already-submitted Notebook changed **only** the known-preview assignment to `preview=False` and preserved all three no-training guards. In the resulting first genuine private 3-row GPU run, the Kaggle kernel reached terminal `ERROR`, with **no `submission.parquet`**. The outcome was a diagnostic, **not a new competition submission**. [Actual private smoke workflow](https://github.com/kolihhan/name-kaggle/actions/runs/43207179461).
4. [Read-only inspection of actual downloaded private runtime logs](https://github.com/kolihhan/name-kaggle/actions/runs/43208484087) located the deepest stack at **`core_worker.py` → `l24_features` → `AutoModel.from_pretrained` → Transformers `caching_allocator_warmup` / `is_accelerator_device`**, ending in the precise exception **`RuntimeError: Cannot access accelerator device when none is available.`**. The upstream `PapermillExecutionError` and `CalledProcessError` are wrappers, not the originating exception. [Sanitized exact error](https://github.com/kolihhan/name-kaggle/actions/runs/43208672527).
5. [CPU-only static source inspection](https://github.com/kolihhan/name-kaggle/actions/runs/43208997263) confirmed `ensemble.py` passes `CUDA_VISIBLE_DEVICES='0,1'` to each `core_worker.py` subprocess, whose model load requests `device_map='auto'`, `max_memory={0:'13GiB',1:'13GiB'}`. There is **no demonstrated GPU-visibility fix yet**. [Source-only installer check](https://github.com/kolihhan/name-kaggle/actions/runs/43209189190) found the wheel install occurs **after** the first core-worker stage, so this later operation cannot explain this first-stage allocator failure.

## Still unproven

- Why PyTorch/Transformers inside `core_worker.py` reports an unavailable accelerator: incorrect Kaggle GPU allocation, interpreter/torch build, a child-process CUDA visibility bug, or Transformers device-mapping/memory offload are still candidate explanations.
- That the hidden scorer's only issue is identical to this initial realpath error; additional issues can arise after the first blocker is fixed.
- That this attributed pretrained ensemble beats current valid champion **0.679** on an independent full-18-label sample or official Public BA. No accuracy claim is warranted.

## Discriminating next execution

One dedicated **model-free**, **no-training**, **no official submission** private Kaggle T4 hardware probe was launched as [Action 43209367647](https://github.com/kolihhan/name-kaggle/actions/runs/43209367647), checking `torch.version.cuda`, `torch.cuda.is_available()`, device count, `nvidia-smi`, and child Python subprocesses with `CUDA_VISIBLE_DEVICES` masks `0,1`, `0`, `1`, and empty. At the time this record was written, no terminal probe result had been verified.

Decision: If the minimal probe has no CUDA device, fix the Kaggle kernel hardware/runtime configuration first. If the probe sees GPUs, instrument only the actual core-worker launch and compare `torch.cuda` visibility and effective `max_memory` before changing the model's source or allocating another full run. **Do not resubmit unchanged source or spend another competition entry** until the complete realpath test succeeds, writes all rows, and satisfies the frozen no-training constraints.
