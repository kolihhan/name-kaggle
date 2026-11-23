# NTL 2026-11-23 — Actual high-score public-source lineage audit (NO TRAIN)

**Verdict: STOP treating different 0.719 notebook titles as distinct model architectures. Compare real final decision execution with 0.715 and focus only on incremental evidence.**

Authenticated code-only read-only experiments:
- [Actions #45136005059](https://github.com/kolihhan/name-kaggle/actions/runs/45136005059) — complete SUCCESS: pull six exact refs, SHA and scoring-module diff.
- [Actions #45136282372](https://github.com/kolihhan/name-kaggle/actions/runs/45136282372) — complete SUCCESS: trace user-prefix and Fable decision scopes in six public refs; no GPU, model inference, dataset/model download, training, or competition submission.
- Scripts: `scripts/ntl_highscore_realcode_audit_20261123.py`, `scripts/ntl_highscore_decision_scope_20261123.py`. Source pulled ephemerally to GitHub runner; code/output summaries only. Running them is **not** an accuracy test.

## 1. Verified source identity and delta

| Source code requested | Exact pulled code SHA256 | Verified observation | Claimed/not independently re-scored |
|---|---|---|---|
| Owner's previously scored 0.715 `kolihhan/ntl-estopadilla-glm53-0716-repro-20261122` | `202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8` | Matches previously independently pinned own scored source | Official owner Public 0.715 (#58680531), no new scoring now |
| Estopadilla public V5 `estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix/5` | **`202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8`** | **Code identical to 0.715 owner; 26/26 same parts, no changed lines** | V5 URL requested, Kaggle pulled metadata does NOT expose returned version; matching SHA confirms actual fetched content, not official author score |
| Sanpier requested V3 `sanpier/0-709-320-token-glm-likelihood-qwen-ensemble/3` | `d526209d0eda5138867324fbb35e11345f1b2bb8dc74d772fcdfb929753a7434` | 39 code cells, ~253k chars, different from owner; GLM-segment pipeline and `apex_postprocess.py` | Public Kaggle page says best **0.717 V3**, not score checked on owner's account |
| Sanpier latest | `9d4b79ab02d63f9152d57bb6b0cc6448c6423f0b678b06f781225812b58e9173` | Different SHA from requested V3; must NOT substitute newest code as V3 scoring proof | Latest does not inherit V3's best-score attribution |
| Dennis `dennis6886/glm-user-prefix-guarded-fable-return` | `85b472b84211f58b26e870f552047e329bdf64958fc2c343ec0952a51f47cf68` | 29 code cells; 22 of 26 overlapping parts unchanged vs owner, adds prefix variant stage and `prefix_decisions.py` | This tool did **not** verify Dennis's official score or exact scored version |
| Areeb `areebbb/0-719-preserve-glm-to-fable-corrections` | `1e23bd888da21a9d481946d315c1ba5b3231bf9bea8332f90cfff28ac677a063` | Same GLM/Fable routine names and decision path; 26 datasets, 10 kernels, 9 models as Dennis/Sanpier | **0.719 occurs in title; not authenticated official 0.719 receipt here** |
| Hcluo `hcluo8888/ntl-0719-glm-prefix-fable-gate` | `a82340ef14bcda3bbaee1d281ecabeee9f2a58b9924363c531c3cbc5182af7f8` | Same `prefix_decisions.fable_return` and `prefix_variant_stage` architecture; 26 datasets, 10 kernels, 9 models | Title is not a verified distinct scored breakthrough |
| Ranjeet `ranjeet258/name-that-llm-sibling-odds` | `15b1251b2c68e523fea3b9e91d2be06029db754e78663ccd4c89273a14deaef5` | Same inherited `fable_return`; extra non-GLM sibling-family decision logic in final `CELL_26` | No independently verified performance increment |
| Evgen forensic EDA latest | `eccba87f3c246d69364ec5affa2c6cdc264281875ebb14079866ca02f070d77b` | 35 code cells; additional upstream train EDA plus inherited Dennis strong scoring code | Displayed/referenced 0.719 is not proof EDA itself scored 0.719 |

**Important version caveat:** Kaggle `kernels pull ref/3` and `ref/5` returned different source contents where expected (Sanpier V3 vs latest), but the emitted `kernel-metadata.json` does not expose a canonical returned version. This supports, but does not independently cryptographically certify, the server-side version mapping. Do not claim a positive Kaggle score from source-only retrieval.

**Kaggle scoreDescending discovery caveat:** 2026-11-23 `kaggle kernels list -v --sort-by scoreDescending` returned 50 public notebook refs, but CSV fields `ref,title,author,lastRunTime,totalVotes` omitted score. Rank order is not an exact per-notebook scored numeric ledger. The visible names 0.719 must not be converted into real validated numeric scores.

## 2. The competitive mechanism is decision-stage evidence conservation, not a new small classifier

### Existing in our scored 0.715
- Large pretrained/LoRA inference stack with Qwen plus other trained expert assets; official train style readout; prompt/sibling conditional inference; GLM-5.3 vs GLM-5.3-Flash segmented causal likelihood; post-GLM protected decisions.
- `pngy_residual.py`: family mass-preserving sibling residual before/around final inference.
- `gauss.py`: soft inferred prompt grouping and nontrivial covariance; **not** blind Hungarian assignment.
- Already saves `/kaggle/temp/final-logits.npz` as pre-GLM 18-way scores; don't assume they are post-GLM final posterior.
- These existing mechanisms mean another style SVC, soft family residual, or hard prompt assignment is **not new**. Appended weak Hungarian previously degraded official Public **0.715 → 0.594**, a negative design warning.

### New/changed in the leading public descendants
- **Dennis** adds an explicit user-prefix variant GLM stage with model-config-aware prefix and a *conditional Fable return*: `prefix_decisions.py:fable_return(base,post,labels)` requires original GLM family argmax and new post-likelihood Fable argmax with margin improvement; then the final downstream stage adds protected rows to the GLM/return mask. Existing older report traces predeclared guards (gain ≥0.30, after-margin ≥0.25); verify exact module before ever reimplementing.
- Dennis's `prefix_variant_stage.py` changes **which examples receive GLM likelihood compute**, prioritizing by strong-base uncertainty/likely benefit and respecting separate Flash/Full GLM deadline estimates. Static code docs say GLM-5.3 scores the available highest-priority subset rather than aborting all work when the budget is tight. **This changes evidence coverage, not a new trained classification model.**
- **Sanpier requested V3** has `apex_postprocess.py` implementing GLM evidence routing and protected `claude-fable-5` return, own segmented `v1_stage.py`, actual gguf six Flash + 19 full GLM segment validation, `--priority-order --shrink`, 8.8-hour stage deadline, and a `160` max text tokens configured for each likelihood pass despite the 320-token notebook title (do not infer 320 per passage per model). Native Qwen stage route differs (not trivially equated to added predictive information).
- **Ranjeet** contains additional family sibling-agreement logic in final cell but no measured increment over the Dennis lineage. Do not treat code or a notebook title as win evidence.
- Areeb/Hcluo and Evgen are **not independent new backbone innovations** merely because their notebook names or code SHAs differ; they inherit the same GLM user-prefix and Fable-guard decision apparatus with the same attached model count (nine). Different code SHAs can reflect revised stages, metadata, chart cells, and budget changes.

## 3. What we can/cannot infer

**FACT:** We compared exact retrieved public code with an authenticated scored-source SHA. The most identifiable new technical levers are (1) more efficient directed likelihood coverage; (2) preserving GLM→Fable corrections through subsequent stages; (3) family-relative margins where evidence actually changes.

**NOT PROVEN:** Whether any specific lever causes +0.002 or +0.004 Public BA, because different runs change multiple factors and hidden per-row truth is unavailable. Also no clean 0.715 OOF on a new independent distribution; no row-aligned final post-GLM joint probability vectors were established in this audit.

**NOT PRIOR ART-FREE:** Dennis and Sanpier already have protected Fable return. Porting their exact decision gate alone is a reproduction, not our breakthrough. Avoid claiming original authorship.

## 4. Research choice from this audit

**Primary question:** Can one obtain genuinely better scoring decisions at constant inference budget by preserving the *provenance and calibrated uncertainty* of stage-specific evidence, particularly the GLM vs Fable transitions, **without** trusting a weak surrogate or doing hard global assignment?

**Cheapest honest next discriminator, still no training:**
1. Treat `0.715` as immutable and rebuild a strict *final decision-stage ledger* on source-only constants: pre-GLM 18-way `logits`, GLM score availability and model-specific budget/censoring, post-GLM score/protect mask, family residual, final label. This is instrumentation/specification only, not another classifier.
2. On user-owned already-existing matched legitimate outputs **if they are available**, assess how many Fable/GLM rows were changed at each stage and whether a new guard would change any labels. Do not claim accuracy without ground truth. A zero-change candidate is worthless even if its code is complex.
3. Evaluate any *materially different* evidence source only if it can be compared with the original 0.715 **on the same independently labeled examples** or with a single pre-frozen official score submission and an explicit public-overfit caveat. If unavailable, state `BLOCKED` rather than train an unrelated proxy or patch an ensemble.
4. More technical novelty needs a distinct inference signal beyond current assets, not another temperature/threshold/label-only override. Running 14B/QLoRA from scratch and low-scoring tiny surrogates is NOT the chosen next step.
5. Do NOT spend new Kaggle submission credits, GPU quota or train models just to compare static code.

## Bottom line

**The strongest public 0.715–0.719 family studied is mostly a *shared upstream strong ensemble* with refined GLM likelihood resource allocation and protected Fable return.** This is much narrower than 'every public score milestone requires a larger newly trained backbone'. It supports a research pivot to evidence-preserving, budget-aware high-fidelity inference. Neither the size nor the apparent number of named notebook variants is evidence of a new model family.

No action beyond ephemeral public-code inspection was executed; no competition score improved as a result of this audit.
