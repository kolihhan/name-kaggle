# NOW — Name That LLM
2026-11-18 22:13 MYT · One owner-authorized Kaggle T4 run ACTIVE

- **Official score remains 0.691**, ref #58375806, until authenticated new score exists. **No new 0.702 score yet.**
- **Candidate:** attributed Lightning Apex 0.702 *title only*, source SHA256 `f9948b633b22bd5e63b8253e4d8027c073b44f57eb85bc60b775857e396b38c2`, executable SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`. Frozen 26 data mounts, 9 model mounts, 9 kernel mounts; no modifications to original executable code.
- **Owner explicitly approved one private GPU production run** on 2026-11-18. [GitHub Actions #44321758700](https://github.com/kolihhan/name-kaggle/actions/runs/44321758700), job 134862256553 **Preflight SUCCESS → Version 2 T4 launch SUCCESS → waiting for Kaggle completion before one official code submission**. [Private Notebook](https://www.kaggle.com/code/kolihhan/ntl-0702-apex-pinned-nov18).
- Two guard failures were corrected before GPU: provider changed only model mount ordering (9 refs identical), and Quick Save v1 reported COMPLETE despite no output. No earlier private GPU launch.
- **Automated scorer path:** existing GitHub Action waits up to ~310 minutes, requires completed Kaggle version 2 and visible `submission.parquet`, checks authenticated quota/no duplicate and submits **exactly once**. If Kaggle version 2 remains running when CI wait expires, do NOT launch version 3 or re-submit blindly; use the existing `scripts/ntl_apex702_official_once.py finish_gpu_once` only after confirmed complete, official ledger absent, and no CI job already submitting. Stop on failure. All scores must come from Kaggle.
- Historical full results before this incident: [NOW.md at prior commit](https://github.com/kolihhan/name-kaggle/blob/8c1b7963feb5ced1f83616f6a64a59c6e8459bf3/NOW.md). Goal official >=0.72. No more weak char/style tuning.

