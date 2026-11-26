# NTL 31B inference-only + strongest public Ensemble — ONE official scoring bet

Date: 2026-11-26 MYT. User explicitly authorized **no newly trained model** and an actual **Gemma 4 31B + Ensemble official Kaggle submission** to test whether Public BA can reach **0.760**. This supersedes earlier no-official-submission preference only for this genuinely new 31B candidate. Exactly one code submission; NEVER recycle failed Confidence Fusion, never adjust after seeing Public result.

## Pinned source and intervention

- Attributed public baseline: Lightning v52 0.724 ([previous independent source/score audit](ntl-lightning-v52-mechanism-result-20261125.md)); exact Kaggle Notebook `lightningv08/0-724-glm-pb115-sibling-odds-qwen3-8-27b/52`, pulled source SHA256 `b4c9715e32d27463a399f8715684c8403f057c83f1f77e0039ff0da2427df343` ([real pinned source audit](https://github.com/kolihhan/name-kaggle/actions/runs/45765840657)). Own current official Best remains 0.718 (#58755304).
- New 31B expert: unmodified **Gemma 4 31B Instruct Q4_K_M** 18,323,726,560-byte GGUF from Kaggle `batuhankska/gemma-4-31b-it-q4-k-m`, plus CUDA Turing SM75 `llama-server` from `nixty2/llama-cpp-turing-sm75`. Both current public Kaggle assets, source/manifests checked in [#45765548592](https://github.com/kolihhan/name-kaggle/actions/runs/45765548592), [#45765840657](https://github.com/kolihhan/name-kaggle/actions/runs/45765840657). Their availability does **not** prove runtime/quality.
- **Cost replacement rather than additive 31B:** set original `--gsq_share 0.035` to `0.0` to disable expensive 27B GSQ stage; reduce previous 7h stage budget to **6.25h**. Keep all other prefit ML components, weights and source mounts of v52. Add 31B final expert for at most **600 lowest-margin** top-two eligible predictions, no more than 1,700s model-inference stage.
- 31B inference is **not trained/fine-tuned**. It reads one official common-prompt set of 18 labeled reference passages **in context only**, requests the model to choose one of two existing candidate labels, and changes final labels only when model returns an exact legal alternate. No gradient updates or LoRA. The inherited v52 pipeline has its original **small logistic/LightGBM heads fitted on official TRAIN at inference**; do not claim those original operations ceased to exist. No NEW weights or fitting have been added.
- Input labels and row IDs must match the official Kaggle test. No internet, external APIs, private data, test truth or hidden label use. Actual GGUF/model-server load and at least one real labelled prediction required even for 3-row Kaggle preview; no cached known-label shortcut for the new expert. Final hidden candidate must change at least one model label, or fail closed rather than falsely claim a new scoring test.
- More than one major decision factor changes when replacing GSQ with 31B and reducing time budget. This is a **composite replacement experiment**, not a strictly isolated additive 31B effect. An official BA result therefore cannot be causally attributed to the 31B alone.

## Hypothesis and mandatory interpretation

**Optimistic H1:** In-context 31B inferences discriminate a subset of ambiguous same-family generators more accurately than the removed 27B GSQ contribution and compensate for running the prior stages with a reduced deadline. Public BA from base 0.724 might improve sufficiently to reach 0.760. This is an **unverified risky hypothesis**, not a forecast.

**Competing H0:** The stronger LM does not know original generating API variants from prose, prompt references overfit to source protocol, or reduced GSQ / time budget causes more regressions than rescues. The 31B can also fail to load / generate under dual-T4 memory or exceed the 9h rerun limit.

**Cost:** 18.3GB Q4 GGUF offline + dual T4 GPUs, max 1,700s additional inference stage, baseline stage budget 6.25h plus V1, total real Kaggle scored runtime must be <9h. Check CPU/GPU viability on a *real model-running* 3-row private preview before consuming official submission slot. A working preview alone does not prove hidden throughput.

**Score gate:** One unique official submission only after CLI push, new version, `COMPLETE` private preview, nonzero `submission.parquet`, actual 31B provenance proof, legal 18 labels, authenticated quota >=2, and no active other scorer. Authenticated submission ID required, and eventually status COMPLETE + numeric Public BA. Goal >=0.760 vs own >=0.718, strong public comparator 0.724; if <=0.724, malformed, or timed-out, RETIRE; no weight/threshold retune using Public LB. If no scoring slot/preview, fail safely without a submission.

## Files

- `scripts/ntl_gemma4_31b_ensemble_stage_20261126.py`
- `scripts/ntl_gemma4_31b_ensemble_submit_once_20261126.py`
- Workflow `.github/workflows/ntl-gemma4-31b-one-official-submit-20261126.yml`

**No claim of a new numeric Kaggle score before an actual Kaggle official Submission ID and terminal score exists.**
