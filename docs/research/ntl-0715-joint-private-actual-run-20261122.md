# NTL 0.715 Champion / new label-only joint inference

Date: 2026-11-22; snapshot as of 14:41 UTC.

## Scientific signal (historical official TRAIN prompt-grouped 5-fold)

- Exact scored champion: official Public 0.715, #58680531 COMPLETE. Champion exports final parquet labels, not 18-way logits.
- Genuine Phase5b strong lexical+style classifier 0.3670892369; weak char-SVC 0.3159582401.
- Joint decoding using strong final labels + weak score matrix + blind inferred Prompt clusters: 0.4226721352 BA18, delta +0.0555828983, 348 rescues, 198 regressions, paired prompt-bootstrap 95% CI [+0.038597, +0.072593].
- Incomplete five-response/prompt sets (750 sampled rows, fold-mean BA): strong alone 0.3655324922 versus anchored joint 0.3814227131 (+0.0158902), 4 of 5 folds positive.
- The group's randomized negative control 0.2937583893. PREDECLARED mechanism gate passed on spent train dev data, not the 0.715 champion itself.
- Authentic [paired experiment #45032497796](https://github.com/kolihhan/name-kaggle/actions/runs/45032497796).

## Actual LIVE Kaggle private source, score unknown

- Confirmed exact original 0.715 scorer Notebook code SHA-256: 202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8, original 32 cells and model/data/kernel mounts unchanged, one added final label-only joint decoding cell. Authenticated [source gate #45033095936](https://github.com/kolihhan/name-kaggle/actions/runs/45033095936) PASS.
- Actual private Notebook push succeeded in [#45033527199](https://github.com/kolihhan/name-kaggle/actions/runs/45033527199), 2026-11-22 14:34 UTC. The Kaggle backend uses the title-derived URL slug, not the metadata ID; a second unique slug --no-run attempt produced a 409 Conflict, no GPU or submit. DO NOT repeat pushes for this existing experiment.
- Real owner Notebook at https://www.kaggle.com/code/kolihhan/ntl-0-715-label-only-joint-decoder-research
- Authenticated actual status is RUNNING, proven [#45034068448](https://github.com/kolihhan/name-kaggle/actions/runs/45034068448). New [snapshot #45034183067](https://github.com/kolihhan/name-kaggle/actions/runs/45034183067) at 14:41:49 UTC: KernelWorkerStatus.RUNNING, no final inference marker or file output known. The private GPU Notebook has begun running, but the new decoder has NOT yet been verified to finish or change predictions.
- No official Kaggle submission, no new leaderboard BA. The best remains 0.715.
- Candidate implementation branch: research/ntl-0715-private-live-run-20261122. Corrected builder slug is kolihhan/ntl-0-715-label-only-joint-decoder-research.

## Guard for next phase

Check terminal status and final output; require actual postprocess completion marker NTL_0715_LABEL_ONLY_JOINT_INFERENCE and correct final parquet and nonzero modifications before considering a single official submit. If the last cell fail-opens, unchanged original score is not evidence for the proposed mechanism. Check hidden test group assumptions, time, and the 4500-row dense clustering memory limit. Do NOT tune from repeatedly reused official prompt folds.

Gemma: latest #45025647137 attempt 3 at 14:35 UTC found private existing two-issue 31B Candidate History QUEUED, grade unavailable. No duplicate 31B GPU. Best official Public 0.18 unchanged.


## 2026-11-22 23:14 MYT — TERMINAL update, NO official uplift

- Exact private Kaggle title-derived kernel `kolihhan/ntl-0-715-label-only-joint-decoder-research` has **KernelWorkerStatus.COMPLETE**, confirmed by authenticated read-only Actions [#45034183067 attempt 2](https://github.com/kolihhan/name-kaggle/actions/runs/45034183067). Its `submission.parquet` exists.
- Correct the provenance of **official owner 0.715 #58680531**: the official code submission actually pointed to **external Estopadilla v5**, not this owner's private clone. Real official submission source proven by [#44931438259](https://github.com/kolihhan/name-kaggle/actions/runs/44931438259).
- Private exact-output comparison with **actual publicly scored Estopadilla v5** (`estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix/5`) in [#45037109580](https://github.com/kolihhan/name-kaggle/actions/runs/45037109580): both available Parquet files have **3 rows**, exact same IDs and **0 changed predictions**, each with one unique class among these 3 rows. The owner's private reproduction kernel had no saved Parquet; that explains the prior false-negative download attempts [#45036868186](https://github.com/kolihhan/name-kaggle/actions/runs/45036868186), [#45036961014](https://github.com/kolihhan/name-kaggle/actions/runs/45036961014).
- The final cell DID execute: actual private run log markers `NTL_0715_LABEL_ONLY_JOINT_SYNTHETIC=PASS`, `NTL_0715_JOINT_INPUT_COUNT=3`, and `NTL_0715_LABEL_ONLY_JOINT_INFERENCE=` are present; no fail-open marker found. Read-only log evidence in [#45037214221](https://github.com/kolihhan/name-kaggle/actions/runs/45037214221).
- This 3-row public Notebook output does **NOT** measure hidden competition BA, and an unmodified 3-row output does not rule out potential differences if the scorer runs a larger hidden test input. Conversely, it proves **no scored advantage** or reliable mechanism generalization. `KernelWorkerStatus.COMPLETE` does not imply official Kaggle submission.
- **Decision:** Do NOT claim an NTL improvement. Preserve official best **0.715**. No official competition submission was made from this candidate. An official code submission would be required to test hidden rerun, but has quota/structure risk and must be separately approved as a deliberate experiment, not inferred from the CPU source gate. Before that, verify actual competition code-submission runtime and decoder behavior on hidden-size clusters and compare budget to the scored original.
- Gemma latest 2026-11-22 23:10 MYT: two-issue Candidate History private kernel remains **QUEUED** with no grade, [#45025647137 attempt 4](https://github.com/kolihhan/gemma-4-kaggle/actions/runs/45025647137). Public best **0.18**.
