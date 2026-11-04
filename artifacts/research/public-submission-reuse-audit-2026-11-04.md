# Public submission reuse audit — 2026-11-04

**Scope:** static examination of published Kaggle notebook source and authenticated dataset metadata; **no training, inference, model-weight download, Kaggle competition submission, or hidden-test scoring**. This report implements the user's decision to stop self-training and examine actual submission evidence first.

## Decision

**Prioritize the historic 0.622 Cosmorin reproduction as a minimal inference-only reproducibility candidate, not as a new competition champion.** Its publicly available code consumes an already-prefit SVC cache and final two-stage LoRA/DeBERTa weights. It has substantially fewer declared dependencies than the later 0.636/0.666 ensembles. This recommendation concerns *feasibility of reproducing the published inference pipeline*, not predicted leaderboard uplift over our submitted 0.520.

**Do not promote the 0.666 latest-source notebook from its public-page runtime or claimed best score.** Its fast three-row public-sample branch writes hard-coded predictions without loading the heavy ensemble; the observed 0.666 notebook score belongs to an earlier version than the currently published modified code. A hidden-size rerun is necessary to establish runtime, integrity and transfer and has **not** been performed.

## Canonical evidence

All workflows ran from `exp/dct-temporal-fingerprint-v3` using sparse checkout and authenticated Kaggle public-source API. They were read-only with respect to Kaggle: no training, model inference, official data fetch, or submissions.

- Successful source-pull/static-manifest workflow: [Actions #41809765887](https://github.com/kolihhan/name-kaggle/actions/runs/41809765887), archive `13311506779`.
- Public notebook source archive: [Actions #41810135445](https://github.com/kolihhan/name-kaggle/actions/runs/41810135445), archive `13311562148`.
- Extra 0.636 code-only public guide: [Actions #41810333036](https://github.com/kolihhan/name-kaggle/actions/runs/41810333036), source archive `13312630277`.
- Public dataset metadata/file enumeration: [Actions #41809970435](https://github.com/kolihhan/name-kaggle/actions/runs/41809970435), archive `13312380580`.
- Notebook-source checksum, exact dataset/model refs and static feature markers are retained in the Actions archives; notebook source came from authenticated `kaggle kernels pull`, not a third-party description.
- Public notebook pages and user-visible best scores are *not* proof of final Private LB, exact rerun reproducibility, model weight license, or 9-hour hidden-data feasibility.

## Comparison: the actual current source, not only title or Public LB

| Source | Advertised / reported Public LB | Source-level inputs | Training vs inference | Verified limitations |
|---|---|---|---|---|
| [Cosmorin historic 0.622](https://www.kaggle.com/code/cosmorin/name-that-llm-public620-reproduction) | 0.622 in notebook historical record; page Version 1 | 2 datasets, 2 base models, 0 other notebooks | Prefit SVC + 2 final LoRA adapters + DeBERTa; no new neural fit in notebook; offline inference | Three-row GPU sample historically 324.4 s (author reported), hidden runtime unmeasured; final adapters saw all official train rows, **not legal as honest OOF** |
| [Cosmorin 0.636 code guide](https://www.kaggle.com/code/cosmorin/name-that-llm-how-to-reproduce-0-636) | Author states official **0.636** from separate **private** scored V1, version 361562324; current public guide itself has **no score-producing run** | 16 datasets, 8 other notebooks, 7 base models, 1 competition ref (32 total refs), many pinned versions | Loads existing neural checkpoints but **does fit a CPU SVC**, then larger complements, ATLAS, group adjustment | Author reports 2002.1 s on **three** sample rows, T4x2; complete hidden runtime and all exact-32-asset access unverified; rights vary |
| [Lightning 0.666 weighted ensemble](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble) | Public notebook page best 0.666 at **V47**; current code was edited beyond scored version | 16 datasets, 8 other notebooks, 5 base models plus large prefit caches/GSQ3 assets | In full path, heavy sequential inference from prefit neural / SVC assets; **special three-row preview** returns fixed predictions without running models | UI 21-second sample **not** measured full inference, large output/input dependency and 9-hour risk; latest source is not the scored V47 |
| [Alex Chilton 0.667-title all-public](https://www.kaggle.com/code/alexchilton/name-that-llm-0-667-all-public-what-failed) | Page best **0.666 V11**, despite title 0.667; author text reports another 0.667 elsewhere, not independently verified for current page | 16 datasets, 8 other notebooks, 8 base models | Full process launches heavy inference; local public-sample branch writes placeholder `gpt-5.5` without full execution | Fast-preview runtime cannot establish hidden re-run viability; author reports nested-CV/leaderboard transfer regressions |
| [Lightning 0.642 historical](https://www.kaggle.com/code/lightningv08/0-636-lora-llama-qwen3-8-27b) | Public page reports 0.642; not independently reproduced | **Not source-verified** | Unknown | Kaggle API pull using old slug failed; no source analysis conclusion |

**Source counts = declared `kernel-metadata.json` inputs; they are *not* a verified list of components actually exercised by each execution path.**

## Concrete findings about transfer and scoring

1. Our submitted champion remains `two-stage-lora24-v1`, Public **0.520**, OOF **0.5467611235**, submission **57908683**. Earlier Phase 15 scored OOF **0.40532** but only Public **0.357**. That negative transfer is why the current research is read-only and does not optimize against Public LB.
2. The published 0.666 and Alex latest code contain public-test-only shortcuts. Lightning directly recognizes the visible three example rows and returns hardcoded labels before model loading, whereas Alex uses `KAGGLE_IS_COMPETITION_RERUN` plus `len(test)<50` to choose a placeholder path. They are **source facts**, not allegations about their historical scored versions. Do not conclude the scored hidden rerun took 21 s / other public-page times.
3. The historic 0.622 notebook **does not have that hardcoded-prediction branch** in reviewed source. It loads SVC cache and both LoRA/DeBERTa models on sample execution. Its `len(test)<=3` watchdog only aborts late sample processing (600s) and `len(test)>=50` controls Sinkhorn. Thus its 324.4 s sample report is a more meaningful *sample feasibility* observation, not a trustworthy 4000-row runtime estimate.
4. The 0.636 notebook is a **public documentation-only code guide**, not the exact private completed V1. It claims a measured 3-row **2002.1s** T4x2 run, **26/26** artifact checks, about 46.9 GB mounted inputs and 32 fixed refs (16 dataset, 8 kernel, 7 model, 1 competition). The **source explicitly says** hidden-run time was not captured, and some hidden-sized branches (Sinkhorn, group paths) are untested on 3-row sample. It is not truly zero-training, because it fits a CPU text classifier, albeit no fresh large-neural-model training.
5. Alex's source includes reported negative results: nested CV **~0.6865** for a five-backbone model translated to only **~0.56 Public**, and some additional group assignment/blending steps regressed. These are author-reported observations, not reproduced. They directly falsify “maximize local CV or add more models = best Kaggle hidden transfer”.
6. **Validation contamination:** the externally published *final* two-stage LoRA weights were already adapted on *all 2,699 official training rows*. They may be used for final inference subject to Kaggle rules/licensing, but **cannot be evaluated as honest held-out OOF on those rows**. An independently disjoint/legitimate evaluation would be needed if considering alterations, and the user has currently paused training/submission.

## Public asset accessibility

Authenticated Kaggle `datasets metadata` and `datasets files` calls **both succeeded for 11 manually inspected refs**, without downloading weights or raw passages. This verifies *file enumeration for this connected Kaggle account at audit time*, **not anonymous access, byte integrity, exact pinned-version consistency, distribution rights, or inference success**.

Notable verified names:

- `cosmorin/name-that-llm-0-622-reproduction-svc-cache`: `ntl620_svc_prefit_v1/manifest.json`, `sample_svc_probability.npz`, `svc_prefit.joblib`.
- `nishantdahal/ntl-handwriting-blend`: Qwen3-14B and Qwen2.5-14B LoRA adapters and heads plus DeBERTa model files.
- `nishantdahal/ntl-two-stage-lora`: other LoRA14B assets.
- `josephadamski91/ntl-atlas-qwen3-8b-v1`: explicit separate code/weight license files (`LICENSE-CODE-APACHE-2.0.txt`, `LICENSE-WEIGHTS-CC-BY-SA-4.0.txt`).
- Also enumerated: `lightningv08/ntl-olmo2-native24-final-inference-assets`, `dhruvtibarewal/nlm-l24-adapters-v5`, `dhruvtibarewal/nlm-enc-v5`, `pjmathematician/ultrabert-380m-ntl`, `bbucxi/qwen35-9b-safetensors`, `kirilltym/ntl-spectral-style-heads-paired-4404`, `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head`.
- Some larger listings yielded empty extracted filenames due CLI CSV formatting; **do not treat their individual file manifest as verified**.

**Rights:** The notebooks' Apache-2.0 code license does not automatically cover third-party weights, competition data, prompt pools or saved predictions. Confirm exact asset cards/versions and redistribution conditions before copying, rehosting or recommending public outputs. The current user's public GitHub repository is not certified history-safe for official competition data; never add any official or private Kaggle rows/files to repo or public Actions artifacts.

## No-training next step / stop conditions

1. **Keep the trained v1 as reference, no GPU/CPU retraining, no v3 resume job, and no Kaggle competition submission.**
2. For the narrow no-training track, **freeze the historic 0.622 notebook's exact published version, dataset versions, base models and existing SVC cache**. Read-only validate availability/license/version/manifest of only those exact inputs; keep code separate from weights.
3. Check the 0.622 inference's hidden-size execution path statically, preflight 9h T4x2/scratch budget and whether competition code sharing rules permit the dependencies. Do not infer 4000-row runtime by linearly extrapolating the three-row sample.
4. Treat 0.636 as a second option only if willing to accept new **CPU SVC fitting** and 32 inputs; treat 0.666 as a more research-heavy option with higher potential published Public but **unverified current scored-code identity and hidden runtime**.
5. If all reusable public assets cannot be verified or rights/runtime preflight fail, **stop**; do not silently drop a model, change weights/inputs, tune based on Public, or submit.
6. To advance to a real competition submission or GPU inference later, obtain a **separate explicit user decision**; this audit does not authorize it. The current competition's private leaderboard is not available in this source.

## Research interpretation

**FACT:** More complex published ensembles advertise higher Public than our current 0.520, and the public source contains real pretrained assets, some accessible by authenticated API.

**INFERENCE:** A small, fixed inference-only reuse of final prefit checkpoints is the least expensive way to test deployment feasibility without self-training, but not an honest new validation signal or assured ranking improvement.

**UNKNOWN:** True Private leaderboard transfer; the exact inference runtime on the full hidden test for the 0.622/0.636/0.666 branches; licenses for every third-party asset; exact scored-vs-current source equivalence for 0.666; whether any future score would improve over our champion.

**REJECTED:** Treating a notebook title, 3-row fast-preview time, latest mutable source, or pretrained final checkpoint as proof of independently validated hidden improvement.
