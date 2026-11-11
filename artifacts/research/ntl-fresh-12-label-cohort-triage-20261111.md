# Fresh-Generation External Cohort Survey — Source Transfer Gate

Date: 2026-11-11 10:51 MYT. Status: **DATA PROVENANCE VERIFIED FROM PUBLIC README/MANIFEST; FULL 18-CLASS TRANSFER GATE BLOCKED**.

## Motivation
Current **own official Public champion is 0.679** (Kaggle submission `58210835`); this is a protected reproduction, not our independent model-attribution breakthrough. The next target is a **material improvement ≥0.70** supported by independent transfer, not repeated public-leaderboard tuning.

Prior same-source matched-pair fingerprinting failed its shuffled negative control. Prompt-conditioned length direction reversed on official vs SWEeper sources. New evidence must disambiguate generator behavior from generation-channel/source artifacts.

## Evidence / executed, no GPU
- [Read-only Kaggle dataset and leaderboard survey #42948264726](https://github.com/kolihhan/name-kaggle/actions/runs/42948264726): dataset discovery success; public leaderboard CLI returned exit 2, so **no live leaderboard rank or top score verified**.
- [Dataset metadata and filenames #42948416413](https://github.com/kolihhan/name-kaggle/actions/runs/42948416413): source manifests checked, no data rows downloaded.
- [README and generation provenance #42948566225](https://github.com/kolihhan/name-kaggle/actions/runs/42948566225): public README and JSON provenance read; no raw response text.
- [Exact 12-label manifest coverage #42948801556](https://github.com/kolihhan/name-kaggle/actions/runs/42948801556): cohort label counts and source limitations verified without reading passages.

| Candidate | License in public README | Rows | Exact target-label coverage | Prompt support | Distinctness & caveats |
| --- | --- | ---: | --- | --- | --- |
| `nimithashreea/ntl-proxy-candidate-generations-20261110` | CC BY 4.0 (attribution to public Seyit prompt bank) | **240** | **12/18**, precisely 20 per label | **20** prompt IDs | New Model Proxy responses after Nov 10. Exact-text collision exclusions with official train and existing proxy pool claimed by source. Prompt bank from already-public `seyitkaangunes/ntl-extra-api-passages`, so **not demonstrated disjoint from champion pretraining/prompts**. No served model version evidence verified. |
| `toanplam/ntl-fresh-generation-pilot-nov08` | CC0-1.0 | **107** | **12/18**, 8–9 per label | **9** authored prompts | Generated Nov 8, declared not admitted to training; author source QC screened 2702 host rows; lexical screening does not prove semantic disjointness. Exact served-model ID not exposed, request settings differ across proxies. **0 full 18-label panels**. |

The **same six target labels are absent from Toan's cohort**: `claude-fable-5`, `gemini-3.7-flash`, `glm-5.3`, `glm-5.3-flash`, `kimi-k3`, `qwen3.8-max`. Nimitha's 12 available labels match the other 12 from this set. Therefore neither supports standalone **18-class** balanced accuracy or adequate coverage of GLM/Kimi/Qwen sibling errors. A 12-label conditional BA result must never be represented as 18-label transfer.

**What is verified:** datasets are public and their README license/provenance and source-defined aggregate row counts have been read; no raw passages committed to GitHub and no new training. Champion's direct public input list does not include these two dataset refs.

**What is NOT verified:** champion model/pretrained pool has never seen their prompt/text equivalents, exact served model revisions, 18-label balanced accuracy, model gain, or clean champion aligned predictions. **No GPU or Kaggle inference executed here.**

## Scientific decision

1. **Reject both as sufficient full-stack evaluation sets**. They may be used for narrow, predeclared **12-label / source-transfer diagnostic** (not score promotion).
2. Do not train/retrain, sweep thresholds or submit a new competition candidate based on this discovery. The current user authorization is `NO NEW TRAINING`.
3. Strongest defensible next experiment is **one same-row, frozen champion inference-only diagnostic** on the independently authored 107-response Toan dataset, plus the 240-response bank-sourced cohort as a distinct negative/source-control. Measure 12-label conditional BA, per-family confusions, prompt overlap/hash screen, and same-row rescue/regression for any independently frozen candidate. Before any Kaggle GPU use, prepare source overlap/permission/compute contract and show a viable scoring path that cannot activate supervision.
4. For a claimed breakthrough, separately acquire genuine 18-label, source/channel-documented, non-overlapping validation evidence including the six missing labels and independent champion scores, then freeze a materially novel mechanism. Existing ATLAS, family gates, prompt grouping, SVC readouts and GSQ mixing are **already in champion**; renaming/reweighting them does not constitute new information.

## PIC decision
**We have new independent-candidate provenance, but no scored research improvement yet.** Do not claim 0.70 or originality until stronger evidence. Current Kaggle terminal champion remains 0.679.
