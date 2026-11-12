# MGT-v1 — terminal official result: 0.679, no gain

**DONE — authenticated at 2026-11-12 11:33 MYT:** `58233227` is COMPLETE with Public BA **0.679**, no error, Private Score unavailable. Frozen baseline `58210835` is also **0.679**: delta **0.000**. [Read-only official evidence](https://github.com/kolihhan/name-kaggle/actions/runs/43149001011), job `130953475412`.

**Decision:** retain baseline; no repeat submission or leaderboard-driven margin-gate adjustment. This result supplies no evidence of expected advantage. The dated acceptance/pending observations below are historical.

---

# MGT-v1 — first authenticated official Kaggle submission (2026-11-11)

**VERIFIED:** Kaggle formally accepted a **new model code submission `58233227`**, UTC `2026-11-11 09:44:14.517` (Malaysia 17:44:14), status **`SubmissionStatus.PENDING`**, `publicScore` empty at independently checked 2026-11-11 09:47:26 UTC. This is **not** a result of 0.700+, or any other unverified score.

## Evidence

- Authenticated [read-only quota, submissions and owned Notebook status workflow #42998099464](https://github.com/kolihhan/name-kaggle/actions/runs/42998099464), job `130447129398`, **SUCCESS**. Its `OFFICIAL_SUBMISSION` ledger entry exactly matches:
  - `ref=58233227`
  - `description="MGT-v1 unlabeled confidence-adaptive prior transport; frozen 0.679 prefit baseline; no new training 20261111"`
  - `status=SubmissionStatus.PENDING`
  - `publicScore=""`
  - `date=2026-11-11 09:44:14.517000`.
- The same authenticated workflow returned `kolihhan/ntl-margin-gated-transport-v1 has status "KernelWorkerStatus.COMPLETE"`, confirming **three-row Kaggle private Notebook preview** completed; hidden scorer runs separately. Preview completion must not be mistaken for hidden-scoring completion.
- [One-shot Kaggle execution workflow #42997421326](https://github.com/kolihhan/name-kaggle/actions/runs/42997421326) completed source preflight **SUCCESS** and started the unique private Notebook/version and competition submit. Preflight exact original code-cell SHA and candidate hash were published in job logs; existing scored anchor submission `58210835` was reconfirmed `0.679`, `COMPLETE`.
- Static synthetic-test gate passed for 18 dimensions: row strength is `0.30 × (1 − (P_top1−P_top2))`, preserving full correction at margin 0 and suppressing it at margin 1. Synthetic tests prove a code/math delta only; they are **not** model-quality evidence.
- Generated private Notebook SHA256 `7eba8c853f4756a18f559eddcebb10fc245caf73ea43ab4273656ad2ca1a35fa`, based on fixed owned champion code-cell SHA256 `28fd128b745f0d95b70f79ea9b2c546d981edf9be08207cbf71a2d9e80ee1c86`. Precisely one scientific code cell (old index 9 `core_math.balance_logits`) changed; original 20 dataset, 8 kernel, 6 model dependencies and all 3 fail-closed no-supervised-fit guards preserved.

## Result policy

This was an **explicitly user-authorized one-shot exploratory official submission**, not a scientifically promoted new 18-class validated model. Clean independent full-18 heldout remains **unavailable**, as recorded in [source readiness gate](../../research/ntl-18class-independent-validation-gate-20261111.md) and [Issue #1](https://github.com/kolihhan/name-kaggle/issues/1).

- If `58233227` later scores **higher than 0.679**, update Public best only after authenticated numeric `COMPLETE`; still do **not** claim independent generalization or original pretrained model.
- If score is <=0.679 or submission fails, keep scored champion `58210835` `0.679`. Do not retune this margin gate on Public LB or repeat submit.
- No new training, weight fitting, per-label override, use of the already-inspected 347/Kimi rows, source copying into GitHub or Kaggle official raw data in Actions.
- Previous Lightning submission `58167751` was still `PENDING` with blank Public Score in the same authenticated ledger. This is unrelated to new MGT-v1 submission.

## Follow-up authenticated read-only official status — 2026-11-11 10:29:35 UTC

- **Evidence:** [GitHub Actions #43004028969](https://github.com/kolihhan/name-kaggle/actions/runs/43004028969), job `130466936098`, **SUCCESS**. The existing read-only status checker was extended to print three exact submission refs; no competition submission, GPU inference, model fitting, or parameter change was made.
- **MGT-v1 `58233227`: `SubmissionStatus.PENDING`, Public Score blank, Private Score blank.**
- **Champion `58210835`: `SubmissionStatus.COMPLETE`, Public BA `0.679`, Private Score blank.**
- **Earlier Lightning `58167751`: `SubmissionStatus.PENDING`, Public Score blank.**
- Private MGT-v1 Notebook status `KernelWorkerStatus.COMPLETE` remains a **preview** result only, not official hidden scoring.
- No leaderboard-driven MGT-v1 retuning, duplicate submission, or promotion. The latest verified Public champion is unchanged.
