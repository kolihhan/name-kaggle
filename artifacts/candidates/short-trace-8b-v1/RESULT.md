# Short-TRACE 8B v1 — first executed checkpoint

**Date:** 2026-11-07  
**Status:** CPU checks and Kaggle upload **VERIFIED**. 8B scientific scores **NOT YET VERIFIED**.

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
- Kaggle private, internet-disabled research kernel: `kolihhan/ntl-short-trace-8b-pilot`. No official competition inputs, no official submission, no LLM fine-tuning or 14B.
- **Scientific outcome still unknown** until actual GPU execution completes with result JSON. A passing CPU suite or successful push is not an 8B feature-gain measurement.

## Next gate
Retrieve the exact Kaggle GPU execution result, check output `short_trace_8b_result.json` and the frozen contract; mark **PASS_FOR_LARGER_8B_VALIDATION_ONLY**, **NO_ADVANCE**, or **BLOCKED**, strictly by observed result. Do not change bin thresholds, shuffle seed, classifier or split after scores.
