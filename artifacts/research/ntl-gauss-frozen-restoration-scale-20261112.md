# NTL 0.700 — Frozen Gaussian recovery + realpath scalability

Status: **EXPERIMENTS IN PROGRESS; NOT SCORED**. Date: 2026-11-12, Kuala Lumpur.

## Official truth

Authenticated Kaggle owned-submission ledger [Action 43249505659](https://github.com/kolihhan/name-kaggle/actions/runs/43249505659), 2026-11-12T15:36:06Z:
- Highest **scored Public BA: 0.679**, 2 tied submissions #58233227 and #58210835.
- Only 4 competition submission slots remaining at that check.
- One Lightning #58167751 still PENDING, #58268344 and #58244483 COMPLETE but blank Public score (not valid scored improvements).
- This project has **NOT** yet demonstrated >=0.700.

## GPU experiment A: 32 original synthetic passages, Gaussian OFF

[Action 43243311525](https://github.com/kolihhan/name-kaggle/actions/runs/43243311525) / private Kaggle kernel `kolihhan/ntl-native-clean-synthetic32-nov12`. It used the full real-model source, 21+8+6 mounts, clean Nvidia Tesla T4 metadata, zero new training, 32 **original synthetic** rows (no real test text or labels), a single new wall-clock marker, and the preview bypass disabled. At latest check **GitHub Action in_progress** polling Kaggle; no terminal time/parquet claim.

An independent static runtime source inspection [Action 43249142049](https://github.com/kolihhan/name-kaggle/actions/runs/43249142049) confirms the computation includes **six sequential core models with per-adapter feature loops**, separate complementary stage with batch size 4, GSQ two-GPU workers, and a final **five-fold Native LoRA expert** running batches of 4 with long-text windowing. Thus the full ~4000-row hidden batch may be much more expensive than the successful private 3-row run (~51 minutes). The 32-row synthetic texts are homogeneous, not representative of all official target lengths, so their timing is a *screen*, not proof of the 9h cap.

## GPU experiment B: restore two truly PRETRAINED Gaussian readouts

- Exact source `CONFIG.paths.gaussian_q14v5` and `gaussian_q8v5` inspected read-only in [Action 43245484759](https://github.com/kolihhan/name-kaggle/actions/runs/43245484759).
- Authenticated private CPU-only mounted filesystem evidence [Action 43246439378](https://github.com/kolihhan/name-kaggle/actions/runs/43246439378): **both** `metadata.json` and `parameters.npz` exist (42,619,056 bytes and 34,099,376 bytes); both metadata contracts identify **`fixed-no-test-redo`, `test_redo=0`, 2699 training-provenance rows, proper source version**. No weights/competition rows copied to GitHub; no fit executed.
- Native `core_readout.py` explicitly supports **`gaussian_state(folder, spec)`** loading frozen prefit state without fitting if model branch `fixed-no-test-redo`. Its other `source-test-conditioned` path calls `gauss.fit` and must be stopped under the current no-new-training boundary. [Original source contract inspection](https://github.com/kolihhan/name-kaggle/actions/runs/43246125175).
- Implemented immutable-source builder `scripts/ntl_native_restore_frozen_gauss_no_fit_20261112.py` with restored source weights `q14v5.gauss=0.7`, `q8v5.gauss=0.44`, no supervised train-row featurization, no runtime `gauss.fit`, fixed Gaussian state loading for zero-redo, and a zero contribution fallback on a redo. Five synthetic regression tests and exact-source CPU builder passed in [Action 43247215386](https://github.com/kolihhan/name-kaggle/actions/runs/43247215386).
- One private genuine full-model 3-row T4 test [Action 43247615098](https://github.com/kolihhan/name-kaggle/actions/runs/43247615098), Kaggle `kolihhan/ntl-native-frozen-gauss-readonly-nov12`, push succeeded, **in_progress** at latest check. No official scored competition submission.

## New quality-evidence limitation (important)

Even if private 3-row inference completes and its 3 predictions differ from the Gaussian-disabled version, **that is not 18-class accuracy evidence**. Conversely, matching 3 predictions does not show that the head is useless: `core_readout` activates Gaussian scores only on topical prompt groups of size **2 to 18**. A read-only check of the previous 3-row Kaggle output [Action 43248884740](https://github.com/kolihhan/name-kaggle/actions/runs/43248884740) found **no exported `Gte.npy` or feature branch metadata files**. Therefore group activation could not be determined; *unknown* is not *zero*.

No candidate is promoted until (1) true full-model execution reaches terminal COMPLETE with valid output and no fitting; (2) realistic input throughput makes 9h plausible; (3) a meaningful label-aware source-independent or bounded Kaggle result establishes a better 18-class macro recall. The incumbent 0.679 remains protected. Separate scoring and runtime evidence.
