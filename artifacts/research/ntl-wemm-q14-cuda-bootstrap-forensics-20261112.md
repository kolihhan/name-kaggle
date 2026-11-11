# WeMM hidden failure — Q14 model-load traceback and CUDA bootstrap investigation

**2026-11-12 MYT.** Real private test evidence. This report supersedes speculative claims that WeMM lacked asset files, that preview success proved GPU inference, or that a `COMPLETE` Kaggle submission with blank score meant a valid scored result.

## Official outcome and confirmed failure

- WeMM competition entry **`58244483`**: `COMPLETE` with **hidden re-execution error** and empty Public/Private scores, [authenticated detail #43071058170](https://github.com/kolihhan/name-kaggle/actions/runs/43071058170). **Not a score**.
- Best authenticated scored competition result remains **`58210835` Public BA 0.679**.
- First owned private full-path 3-row Notebook `kolihhan/ntl-wemm-fullpath-3row-smoke-v1` intentionally bypassed the original public-fixture precomputed answers and did real ensemble execution. Its **terminal state was ERROR**; [read-only forensics #43076209088](https://github.com/kolihhan/name-kaggle/actions/runs/43076209088), no output. `core-q14v5` returned **exit code 1** around 278 seconds into Notebook logs. Parent `ensemble.py:18–26` redirected child stderr to `/kaggle/working/inference-logs/core-q14v5.log`.
- Second private one-shot `kolihhan/ntl-wemm-core-q14-error-probe-v2` changed **only safe failure instrumentation**, retained 43 original code cells and 4 no-new-supervised-training guards, and similarly ended **ERROR**. [Authenticated v2 terminal evidence #43077627100](https://github.com/kolihhan/name-kaggle/actions/runs/43077627100).

## Recovered exact failure — no further GPU necessary

Unexpectedly, `kaggle kernels output --file-pattern '.*core-q14v5[.]log$'` **retrieved the failed worker's stage file**, even though the regular `kaggle kernels files` listing did not list output files. [Read-only recovery #43078644928](https://github.com/kolihhan/name-kaggle/actions/runs/43078644928).

Precise stack on v2:
```text
core_worker.py:195 main → l24_features(...)
core_worker.py:57 AutoModel.from_pretrained(..., device_map='auto', max_memory={0:'13GiB',1:'13GiB'})
transformers/modeling_utils.py:4435 caching_allocator_warmup(...)
transformers/modeling_utils.py:5057 device-map accelerator filtering
transformers/modeling_utils.py:5005 is_accelerator_device → torch.device(device)
RuntimeError: Cannot access accelerator device when none is available.
```
The failure is **in the first Qwen 14B model loader, before WeMM, SVC blend, or writing predictions**. No evidence of a child `NO_NEW_TRAINING` guard violation or literal CUDA-out-of-memory traceback. The exact accelerator context state in this *child process* is **not yet measured**, so do not claim the kernel was CPU-only.

## Independent GPU facts and competing interpretations

- Separate **private GPU environment canary** `kolihhan/ntl-wemm-cuda-env-probe-20261112` executed successfully: 2× Tesla T4 (15,360 MiB each), CUDA available, CUDA tensor operation PASS, CPU pinned-memory PASS, PyTorch `2.11.0+cu130`, Transformers `5.20.2`, Accelerate `1.15.0`. [Verified readout #43079478140](https://github.com/kolihhan/name-kaggle/actions/runs/43079478140). No competition data or model loading.
- Separate private `kolihhan/ntl-wemm-int-device-probe-20261112` confirmed `torch.device(0/1)` and `transformers.modeling_utils.is_accelerator_device(0/1)` both work in **a normally initialized CUDA session**. [Verified readout #43080221106](https://github.com/kolihhan/name-kaggle/actions/runs/43080221106). A blanket patch against integer device IDs is not justified.
- Source check found `ensemble.py:20` sets `CUDA_VISIBLE_DEVICES=gpus`, and the Q14 stage call passes `gpus='0,1'`. The owned private Notebook metadata really has `enable_gpu=true, machine_shape=NvidiaTeslaT4`. [Source/metadata audit #43080464521](https://github.com/kolihhan/name-kaggle/actions/runs/43080464521).
- **Working hypothesis (unproven):** Q14 child reaches `torch.device(int)` before its CUDA/current accelerator context is initialized, whereas independent canaries use CUDA before the integer-device check. Alternative remains an intermittent/missing child accelerator context or another loader-specific mechanism; do not extrapolate a platform-wide GPU outage.

## One changed-variable private repair experiment

- Created `kolihhan/ntl-wemm-q14-cuda-bootstrap-smoke-v3` from the **hash-pinned private v2**; [one-shot creation #43080709095](https://github.com/kolihhan/name-kaggle/actions/runs/43080709095).
- **Only intended runtime intervention:** in `core_worker.py` just before the existing Q14 `AutoModel.from_pretrained(...device_map='auto')`, check CUDA availability, call `torch.cuda.init()` and `torch.cuda.set_device(0)`, verify `torch.device(0)` and log safe CUDA-count/device metadata. Existing model weights, head, features, all other code and four no-training guards remain unchanged. New full path stays private, no internet, T4, **1,200-second GPU cap**, no Kaggle competition submission.
- Source digest `0f0fabff8abbb4a96216ef74a0ecdb4e3be3e5599efff111631f30fb5988882a`; v3 changed digest `72430e7d613a870276c1d36238993e5ab496299a2fa2dd13e4846219e921a306`.
- **V3 result PENDING at this report's creation.** Do not say bootstrap repaired Q14 unless the authenticated terminal status and `core-q14v5.log` prove that. [Bounded read-only status workflow #43080939912](https://github.com/kolihhan/name-kaggle/actions/runs/43080939912) checks this without GPU.

## Decision boundary

- If v3 CUDA bootstrap succeeds past Q14, continue to test whole-model preflight. A 3-row real-path success still does **not** prove 4,000-row hidden runtime, memory, checkpoint rights, or 18-class transfer quality.
- If v3 fails with the **same** exact exception, reject lazy-init patch and investigate child accelerator visibility without altering classification/weights.
- If v3 encounters a different stage error, identify it from its stored private stage log before another experiment.
- Only a full independently reproducible, within-budget preflight can justify another one-shot competition submission; none was made here. Current valid best is 0.679.
