# Official Kaggle outcome — guarded public Ozertuu 0.683-title reproduction

Date verified: **2026-11-11 10:28 MYT** (GitHub Actions log 2026-11-11 02:28:39 UTC).

## Verified result

**Own new confirmed Public Balanced Accuracy: 0.679**.

- Exact competition: `name-that-llm`
- **Our own submission ID:** `58210835`
- Kaggle status: **`SubmissionStatus.COMPLETE`**
- Public Score: **`0.679`**
- Private Score: **blank** (not yet available)
- Previous confirmed champion: `0.668` (submission refs `58022340`, `58045266`)
- Absolute gain: **+0.011 BA = +1.1 percentage points**; relative gain ≈1.65%
- Other pending candidate `58167751` (Lightning publicly titled 0.681) remains `SubmissionStatus.PENDING`, Public Score empty as of this check.

## Authenticated evidence

- [Read-only Kaggle Actions run #42874538415, attempt 2](https://github.com/kolihhan/name-kaggle/actions/runs/42874538415) — job `130279696484` completed successfully.
- Kaggle official `competitions submissions name-that-llm -v -q` returned `{"ref":"58210835","status":"SubmissionStatus.COMPLETE","publicScore":"0.679","privateScore":""}`.
- Independent exact-ID `kaggle competitions submission 58210835` returned `Status: COMPLETE`, `Public Score: 0.679`, no Private Score.
- [One-shot private Notebook push/preview/submission #42873288053](https://github.com/kolihhan/name-kaggle/actions/runs/42873288053), accepted submission ref 58210835.
- [Hash-pinned no-new-supervised-training preflight #42872819394](https://github.com/kolihhan/name-kaggle/actions/runs/42872819394), source SHA `14d48eb60e4201cd749e0db7f6b9fae89796d4cbb10a4f2f7006300f267535c6`, guarded own version SHA `49d4fdce69b4586afff6d240969ffe443f6537a5f810803e758313bb62b63574`.

## Interpretation and restriction

- **This is a real, independently verified improvement on our Kaggle account.**
- It is an **attributed reproduction** of the author's public Ozertuu Notebook titled 0.683, with frozen safety guards and original prefit assets. It is **not independently developed science**, and the upstream title `0.683` was **not independently verified**.
- Observed own `0.679` is **0.004 BA below the upstream title claim**. This discrepancy has not been causally attributed to the no-training guards, code path, hidden dataset sampling, scoring variability, or another factor. Do not treat the four-digit difference as license to tune against LB.
- Public score is based on the competition's public slice, not proof of private leaderboard transfer or full-model error structure.
- No new GPU training, LoRA supervision, Gaussian fitting, or repeated competition submission was performed by us beyond the permitted one-shot **inference-only** Kaggle job.
- Maintain `58210835` as the champion until another **our-account terminal official scored submission** legitimately exceeds 0.679. `58167751` remains pending and should be checked on future request but not duplicated.
- Research agenda remains: independently demonstrate a source-robust complementary exact-generator attribution signal against this strongest anchor using same-row held-out data. This score alone is not that demonstration.

## Durable outcome

**PROMOTE** external-reproduction Kaggle champion `0.668 → 0.679`. Do not rewrite historical diagnoses, frozen evaluation policy, or completed negative controls.
