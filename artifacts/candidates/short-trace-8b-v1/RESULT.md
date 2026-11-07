# Short-TRACE 8B v1 — first executed checkpoint

**Date:** 2026-11-07  
**Status:** Initial GPU v1 run **ERROR — EXECUTION BLOCKED by missing offline dependency**, verified by authenticated read-only Kaggle status at **2026-11-07 13:11:49 UTC**. No rank features or H/S/T scientific scores were generated. This is **not scientific FAIL**.

## Frozen science
- [Contract](CONTRACT.md), committed before any 8B scores.
- The same pretrained frozen Qwen3-8B-Base evaluator yields H = rank histogram, S = histogram + shuffled adjacent-rank matrix, T = histogram + true adjacent-rank matrix.
- Shuffling preserves the **same** rank frequency and passage length while breaking token adjacency; fixed seed based on source text hash only, never model/label.
- A fixed identical StandardScaler + LogisticRegression readout is trained per arm on external source-disjoint data only.
- Train/holdout: the **already frozen** (not fresh) alperen/hailey/willrice source split of Kaggle public pool v4, 17 eligible labels; 20 train and 12 heldout prompt-distinct rows per label. The pilot is a *mechanism feasibility test*, not an independently valid 18-label Kaggle accuracy estimate.
- Pilot precommit gate: T exceeds both S and H by 0.03 absolute BA; 1,000 prompt-bootstrap lower 95% bound on T-S above zero; and no per-source T-vs-S BA decline more than 0.05. This can only authorize **larger 8B validation**, not 14B or competition submission.

## Verified engineering
- Implemented `scripts/short_trace_core.py`, `scripts/run_short_trace_8b_pilot.py`, `scripts/build_short_trace_8b_pilot.py`.
- Implemented deterministic/null-control/prompt tests in `tests/test_short_trace_core.py`.
- GitHub Actions [run 42367277419](https://github.com/kolihhan/name-kaggle/actions/runs/42367277419): `Frozen unit checks no real source/competition data` **SUCCESS**, `Build a private internet-disabled frozen GPU kernel` **SUCCESS**, `Push exactly one Kaggle 8B pilot kernel` **SUCCESS**.
- Kaggle private, internet-disabled research kernel **actual resolved slug**: `kolihhan/ntl-short-trace-8b-frozen-representation-pilot` ([Notebook URL](https://www.kaggle.com/code/kolihhan/ntl-short-trace-8b-frozen-representation-pilot)). The metadata requested `kolihhan/ntl-short-trace-8b-pilot` but Kaggle rewrote the slug from the title, causing 28 status polls to show a misleading permission denial. The exact v1 push output is in [run 42367277419](https://github.com/kolihhan/name-kaggle/actions/runs/42367277419); [read-only corrected-slug check 42367892615](https://github.com/kolihhan/name-kaggle/actions/runs/42367892615) returned `KernelWorkerStatus.QUEUED` (verified). **This is not a scientific failure or a model runtime error.** No official competition inputs, no official submission, no LLM fine-tuning or 14B.
- **Scientific outcome still unknown** until actual GPU execution completes with result JSON. A passing CPU suite or successful push is not an 8B feature-gain measurement.

## Next gate
Retrieve the exact Kaggle GPU execution result, check output `short_trace_8b_result.json` and the frozen contract; mark **PASS_FOR_LARGER_8B_VALIDATION_ONLY**, **NO_ADVANCE**, or **BLOCKED**, strictly by observed result. Do not change bin thresholds, shuffle seed, classifier or split after scores.


## Verified execution failure — 2026-11-07 13:11 UTC

- Read-only [Actions 42375838167](https://github.com/kolihhan/name-kaggle/actions/runs/42375838167) checked actual Kaggle slug and received `KernelWorkerStatus.ERROR`. Kaggle logs showed the Python 3.13 kernel raised:
  ```
  importlib.metadata.PackageNotFoundError:
  No package metadata was found for bitsandbytes
  ```
- The exception occurs in `BitsAndBytesConfig(load_in_4bit=True)` while configuring 4-bit NF4 quantization, before loading/running Qwen3-8B evaluator and before any H/S/T predictions, scores, or artifacts. The Kaggle Notebook v1 *upload* succeeded; its actual model execution did not.
- Science controls and data split remain untouched. Fix requires a compatible **Python 3.13 offline bitsandbytes wheel**, tested in the exact Kaggle GPU environment. Respect `enable_internet=false`; mount prepackaged dependency in an approved Kaggle dataset/runtime and pin its sha/version. A new *infrastructure-only* kernel version can then be preflighted and run with the same frozen H/S/T science.
- No 14B escalation or official competition submission; 8B remains **BLOCKED_BY_ENVIRONMENT**. A green GitHub Actions wrapper run must never be interpreted as successful GPU inference.

