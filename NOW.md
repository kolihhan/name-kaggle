# 2026-11-28 — Research reset: proxy metrics are not the objective

**Current verified account Public BA: 0.718 (#58755304, attributed public reproduction). Private BA unknown.** Reference submission #59044358 last authenticated PENDING at 18:39 MYT [46136170973](https://github.com/kolihhan/name-kaggle/actions/runs/46136170973). Target >=0.760 final Private BA; public score is a separate intermediate outcome.

**Research decision:** [PB115 matched-window result](docs/research/ntl-reset-pb115-window-decision-20261128.md) retires only the exact global short-window readout as a demonstrated improvement; it does not establish hidden-score harm. CSIM first conditional-objective diagnostic [46139254989](https://github.com/kolihhan/name-kaggle/actions/runs/46139254989): same char 3–5gram linear readout, CE vs 50% matched-prompt column CE; official five-way 750 rows .492 -> .461333, external 4,820 rows / 964 groups .467635 -> .461203 (delta -.006432; prompt CI [-.012863, approximately 0]; 111 rescues, 142 regressions). This exact intervention failed its predeclared +.02 mechanistic gate, not a verdict against large-backbone CSIM, training objectives generally, or Kaggle hidden performance. External source uses public Open Pool prompts; no independent full18 champion comparison, no external fitting or neural training.

**Evaluation interpretation:** development CV, partial-class diagnostics, external transfer, official Public BA and final Private BA measure different populations. None can be substituted for another. No inference of public/private deltas from proxy deltas; a negative proxy does not prove a hidden-score decline. Positive proxy metrics do not authorize a submission. Keep the frozen evaluation policy, exact-intervention stop rules, and independent same-row evidence gate intact; do not tune against Public LB. Next: compare pinned scored strong architectures for a distinct mechanism rather than optimize this weak readout. No new submission or GPU authorized by this record.

---

# 2026-11-23 10:30 MYT — OFFICIAL NTL JOINT-DECODER SCORE 0.594, HARD NO-GO

**Authenticated Kaggle terminal receipt: official Code Submission #58729013 COMPLETE with Public Balanced Accuracy 0.594**, vs unchanged true account Champion **#58680531 COMPLETE 0.715**, delta **−0.121** (−12.1 percentage points). Independent authenticated read-only Kaggle ledger in [GitHub Actions #45100579731](https://github.com/kolihhan/name-kaggle/actions/runs/45100579731), executed 2026-11-23 10:30 MYT.

**Treatment:** exact original Estopadilla v5 0.715 scorer 32 cells + one final weak char-SVC scores / blind lexical Prompt-group Hungarian hard-label decoder, official submission marker `ntl-0715-joint-label-hard-anchor-once-20261122`. Original scored code source and mounted resources pinned. The private preview's 3 rows were unchanged, but the official hidden reexecution scored much worse. The hidden test's actual per-row predictions or task distribution have not been inspected; do not assert an exact causal mechanism for the decline.

**Scientific decision: RETIRE/NO-GO this particular final groupwise one-label-per-class reconciler on the 0.715 scorer.** Historical Phase5b TRAIN prompt-grouped 5-fold gain +0.0556 was a weak-proxy experiment, not transfer evidence for 0.715; real score falsifies its promoted hidden-gain hypothesis (predeclared ≥0.725). Possible explanations to investigate *read-only* if justified: blind cluster purity error, invalid one-each-model prior in hidden prompts, weak-model out-of-domain calibration, mismatch of strong-model hard-label error structure, and skip/fail-open behavior. None of those is proven by 0.594 alone.

**Do not repeat this Kaggle candidate, tune group threshold/anchor via Public LB, overwrite the current Champion or claim 0.594 as new best.** Historical best Public remains **0.715**. No official NTL improvement resulted.

---

# 2026-11-22 23:44 MYT — ACTUAL 0.715 + LABEL-ONLY JOINT OFFICIAL CODE SUBMISSION PENDING

**FACT:** Kaggle accepted real official Code Submission **#58729013** 2026-11-22 15:44 UTC, marker `ntl-0715-joint-label-hard-anchor-once-20261122`, **PENDING / numeric Public BA EMPTY**, authenticated [Action #45039930220](https://github.com/kolihhan/name-kaggle/actions/runs/45039930220). Previous official scored best **0.715 #58680531** untouched.

**Treatment:** exact original 0.715 Estopadilla v5 source (32 cells SHA256 `202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8`), plus one weak char-SVC + blind Prompt-set Hungarian hard-label reconciler Cell (runtime SHA256 `70d4b95eef7900c99927967ab63d60b6f20c21537b7090515a1de78893ba08d6`). The Kaggle title-derived exact private source `kolihhan/ntl-0-715-label-only-joint-decoder-research` is COMPLETE, 3-row preview output had **0/3 changed predictions**; ~4,000-row hidden rerun may be different. Scorer run is the actual prospective test, not the preview. Official quota before submit: 5 available, 0 active; one slot preserved.

**Research hypothesis:** Public BA ≥0.725 (+0.010) vs 0.715. Kill if ≤0.715, empty score or runtime failure; no repeated tuning against public results. Grouped 2,699-row official-train 5-fold evidence is historical adaptive DEV, cannot be equated to hidden score. No newer Public gain claimed yet. No new model training, no default Champion change.

---

# 2026-11-20 — ORIGINAL NTL 13-generator per-row metadata checked: 4,758/4,758 exact ID match

**REAL [Actions #44765273054](https://github.com/kolihhan/name-kaggle/actions/runs/44765273054) SUCCESS.** Downloaded publisher `lxdlxdlxd/ntl-synth-generations-v1` source parquet ephemerally; read only generation metadata, no passages or prompts. Original full source **4,758 rows, 13 labels**, all **requested_model = returned_model** by publisher-recorded ID. Claude 5×330 via Vertex; Gemini 4×390 with three settings per label; GLM 406/413 via Zhipu; Kimi-K3 369 (320 metadata settings explicitly thinking-off); Qwen3.8-max 360. [Per-family totals and scope](artifacts/research/ntl-original-13generator-row-route-verify-20261120/RESULT.md). Internal publisher ID consistency is stronger provenance than aggregated route checkbox, but does **NOT** certify API execution, lawful training rights, 18-class coverage, or host-domain parity. Main source was 4,270/8,347 in the filtered training cohort; full upstream has 4,758. No new GPU/QLoRA or competition submission; scored best **0.702** unchanged. NEXT potential competitive lever must clear original provider rights, family-specific settings and fresh same-row transfer vs 0.702. Do not blindly launch full 8,347 train.

---
# 2026-11-20 23:10 MYT — Official-only CE baseline already tested: stop duplicate retraining

**Historical completed baseline verified from repository Phase4 README and training contract**: exact official Kaggle TRAIN 2,699 rows / 150 prompts / frozen 5-fold prompt-disjoint OOF, **frozen ModernBERT 0.30224 versus full-finetuned ModernBERT 0.25781** (Phase4); official-only Phase5b stronger sparse anchor **0.36709**. Therefore repeating generic official-only trainable encoder/CE without a new mechanism, stronger independent label evidence, or GPU feasibility is not justified. This is historical evidence, **NOT** new experiment/score. GroupKFold official OOF has been reused adaptively, not a fresh heldout. Next competitive lever is nonredundant lawful model-identity signal surviving independent source/host transfer and paired same-row comparison vs actual **0.702** strong ensemble. [Original 4,270-row dataset generation and domain mismatch report](artifacts/research/ntl-4270-upstream-generator-domain-shift-20261120/RESULT.md); [provider terms constraints](artifacts/research/ntl-provider-output-use-terms-20261120/RESULT.md). Do not launch blind 8,347-row external QLoRA or blind duplicate ModernBERT. No Kaggle submission, new training/GPU or Public gain.

---

# 2026-11-20 22:55 MYT — provider terms checked against original source; full external CE remains blocked

**First-party provider policy review:** [results](artifacts/research/ntl-provider-output-use-terms-20261120/RESULT.md) using Z.ai official API terms, Anthropic output-training guidance, Google Cloud AI/ML restrictions and OpenRouter model-terms requirements. **Z.ai-origin GLM external output is not admissible for classifier training without explicit authorization** under current API terms; Anthropic explicitly allows noncompeting classifiers but original Claude-on-Vertex and third-party transfer terms still need confirmation. OpenRouter requires following each upstream model's terms. **No automatic approval** from publisher CC BY 4.0; no training or new submission. The 4,270-row main source original README also warned Gemini/Kimi synthetic domain shift. NEXT: focus permitted official TRAIN 18-label supervised baseline or an independently rights-cleared source cohort; freeze before Colab experiment. Official best **0.702** unchanged.

---

# 2026-11-20 22:49 MYT — original 4,270-row generator source confirms API lineage claims but exposes family-specific host domain gap

**Real original-source CPU checks SUCCESS:** [generator code #44756877019](https://github.com/kolihhan/name-kaggle/actions/runs/44756877019), [README domain/rights #44757193056](https://github.com/kolihhan/name-kaggle/actions/runs/44757193056). The dominant `lxdlxdlxd/ntl-synth-generations-v1` has original Vertex/OpenRouter/Zhipu/Anthropic/OpenAI-compatible generation code; public README reports per-row `requested_model`/`returned_model` match. **Not provider-authenticated run receipts.** Publisher explicitly documents Gemini synthetic em-dash **3–40% vs official 0–1%**, Kimi thinking-enabled **93% vs official 52%**, while Claude and some GLM/Qwen settings are closer. **Source CC BY 4.0 does not supersede provider output terms**, as original README says. [Full evidence](artifacts/research/ntl-4270-upstream-generator-domain-shift-20261120/RESULT.md). **NEXT:** verify provider training rights and freeze family-aware licensed data comparison instead of blind 8,347 full-source CE or match_weight. Public best **0.702**, no new training or Kaggle submission.

---

# 2026-11-20 22:05 MYT — exact 8,347-row original-source training audit completed

**Real CPU [Action #44749886505](https://github.com/kolihhan/name-kaggle/actions/runs/44749886505) SUCCESS**: original 8,347-row 18-label PC-CL cohort came from 8 upstream Kaggle datasets. One source (`lxdlxdlxd/ntl-synth-generations-v1`) contributes **4,270 / 8,347 = 51.16%**. Four sources totaling **1,820 (21.8%)** have zero nonempty per-row `gen_served_model` despite publisher protocol claiming a served-model check. Two single-label sources contribute 1,767. All eight source manifests claim CC BY 4.0; original provider-output rights/served versions are NOT independently cleared. [Actual source-by-source report](artifacts/research/ntl-8k-original-source-audit-20261120/RESULT.md). **Decision**: do not blindly train full 8,347 as verified 18-class truth. Validate original high-volume source and lawful routes before full-exposure CE; if unavailable, use permitted official TRAIN-only group-disjoint dev experiments, not false host-transfer claims. 0.702 #58514818 unchanged; no GPU/submission.

---

# 2026-11-20 17:28 MYT — Host-match proxy source-locked CPU audit completed

**Actual experiment [GHA #44710955318](https://github.com/kolihhan/name-kaggle/actions/runs/44710955318): SUCCESS.** Exact 231,036-row Open Pool SHA pinned; reconstructed 8,347-row external train/1,800 held-source valid. Matched high-weight 2,095 vs equal-N random 2,095 **with identical (source,label) counts**; match_weight median 1.8301 vs 0.8722. Publisher-source eta² 0.013801. **25% high-weight subset fails strict 18-label sufficiency:** 11 labels <100 rows, 7 labels <2 training sources. No model BA, GPU, or Kaggle submit. See [full result](artifacts/research/ntl-hostmatch-source-locked-20261120/RESULT.md). **NEXT research hypothesis:** preserve full 8,347-row 18-label cohort, compare uniform vs within-source/label match-weighted exposures on same capable trainable model only after original rights/serving-route checks; no blind 1.7B PC-CL rerun. Official incumbent 0.702 unchanged.

---

# 2026-11-20 15:53 MYT — Actual first PC-CL QLoRA complete; matched CPU source learnability falsifier

**DECISION: REJECT frozen 160-step PC-CL v0; PIVOT from loss tweaking to training sufficiency/source transfer.** [Full real result and follow-up](artifacts/research/ntl-pccl-20261120/RESULT.md). Authentic owner Colab T4 Qwen3-1.7B 4-bit r8 **same training batches and budget**: CE **0.078889**, CE+2×2 prompt-controlled InfoNCE **0.071667**, Δ **−0.007222**, 95% paired cluster CI **[−0.018839,+0.004112]**, rescues **62**, regressions **75**, 18 labels/1,800 external source-disjoint validation. [GHA #44683814370](https://github.com/kolihhan/name-kaggle/actions/runs/44683814370) **SUCCESS; real GPU weights optimized, Colab released; no official Kaggle scoring.** 160 steps × 4 = 640 presentations, only 13 classes present in real prompt rectangles; 12/120 rectangles cross training sources. Controls weak and under-exposed: do NOT claim this disproves Contrastive Learning generally.

**Actual CPU same-cohort discriminating diagnostic** [GHA #44698095061](https://github.com/kolihhan/name-kaggle/actions/runs/44698095061) SUCCESS: TF-IDF char+word LinearSVC on the **same 640 presentations (532 distinct texts)** → **0.122778 BA18** on same 1,800 heldout. Same SVM trained on all 8,347 approved train rows → **0.216111 BA18**. These are different classifiers, **not a causal proof** of optimization effect, but show some learnable cross-source signal and QLoRA training exposure insufficiency. The original PC-CL v0 **reject** is unchanged. Two informative diagnostics consumed; stop further weights/temperatures/seeds on this cohort. Candidate future test is stronger **balanced CE-only full-exposure** classifier on a genuinely independent source cohort before considering a new contrastive hypothesis. Never imply 0.216 vs 0.702 is same-data comparison.

**Competition unchanged:** own authentic Public best **0.702 #58514818**, top10 Public cutoff last authenticated Nov20 **0.761**. No NTL submission, no champion improvement from these experiments. Public publisher training data only, SHA-pinned, no raw passages/IDs/predictions uploaded to GitHub.

---

# 2026-11-20 12:40 MYT — True 0.702 champion vs latest 0.704 public code audited

**Actual original-source audit:** [complete results](artifacts/research/ntl-0702-vs-0704-true-source-diff-20261120/RESULT.md); GitHub Actions [#44675311464](https://github.com/kolihhan/name-kaggle/actions/runs/44675311464) SUCCESS. Owner's **official COMPLETE 0.702** source code SHA `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`. Public Lightning "0.704" Notebook: **27/29 identical source parts**, only **GLM-queue Fable-mass priority** and **Apex final GLM/postprocessing** changed. These two interventions are confounded and notebook title +.002 is not a reliably demonstrated model gain. Areeb 0.704 contains extra visualizations/analysis. This source-level audit rules out the thesis that public 0.704 introduced a novel large reader/backbone.

**Paired-strong comparison still blocked for a concrete reason, not vague gate:** [#44675693564](https://github.com/kolihhan/name-kaggle/actions/runs/44675693564) saw actual scored champion private Notebook output; **no 18-way component/pre-post GLM/Apex logits are exported**, only `submission.parquet`, code and smoke binaries. Can't calculate true champion rescues/regressions without rerunning actual strong inference on labeled rows. Preview output isn't official hidden test labels.

**Post-training fresh-source 18-label holdout feasibility:** Kaggle API [#44675959849](https://github.com/kolihhan/name-kaggle/actions/runs/44675959849) SUCCESS. Current 231,036-row Open Pool `gen_date` parseable in 33,603, **ZERO rows published as generated on/after 2026-11-16**. Metadata route claims in 85,470, but not independently authenticated. An independently dated after-training 18-class holdout simply is not present in that snapshot. Do not mistake a new dataset version for new training-source-independent generated examples.

**Competition:** authenticated 0.702 best still, live current top10 0.761 (gap 0.059). **No new NTL Kaggle scorer submission, GPU, training or achieved uplift.** No 0.704 clone submission, no Oleg spend-based proxy as fresh eval. Focus on acquiring a genuinely new route-authenticated 18-label prompt panel if further strong-GPU experiments are cost-justified.

---

# 2026-11-20 12:25 MYT — live Kaggle ranking target corrected

**Official authenticated Public leaderboard [Actions #44674191096](https://github.com/kolihhan/name-kaggle/actions/runs/44674191096) SUCCESS:** top 1 **0.776**, top 10 cutoff **0.761**, top20 cutoff **0.751**. Owner frozen **#58514818 COMPLETE 0.702**, **gap to current top10 = 0.059 BA**. 23 account submissions returned, 0 pending. This API returns 20 teams, **owner exact rank unknown**. Do NOT tell owner 0.702 is already near top10; earlier 0.692 top10 anecdote is stale. [Full authenticated decision memo](artifacts/research/ntl-live-lb-submission-priority-20261120/RESULT.md).

**Diagnostic submissions decision:** Phase14 0.4178 OOF includes publisher/provider-terms-excluded external Gemini/GLM/Kimi; do not Kaggle-submit without original-source rights review. Official-only Phase15-recipe 0.374857 true grouped OOF (historical CPU) remains **Public UNKNOWN**, but Phase7 Public 0.369 uses different architecture and cannot predict its hidden score. Two ~0.36–.42 SVM submissions do not address owner's gap to 0.761, and one-shot score noise means they would be weak causal evidence. **No NTL submission made**. Strong 0.702 ensemble still lacks fold-lineage-clean full OOF, so do not claim cheap existing-model complementarity from old 347-row/Phase15 data.

---

# 2026-11-20 11:35 MYT — Phase15 near-leak and Open Pool source-risks: real CPU outcomes

**New completed independent source checks:** [Full read-only/CPU report](artifacts/research/ntl-phase15-neighbor-pool-provenance-20261120/RESULT.md), and source [GHA #44667515988](https://github.com/kolihhan/name-kaggle/actions/runs/44667515988), near [GHA #44667893709](https://github.com/kolihhan/name-kaggle/actions/runs/44667893709), weak7 [GHA #44668450000](https://github.com/kolihhan/name-kaggle/actions/runs/44668450000). The first combined workflow overall failure was **only a missing-label empty-quantile reporting bug** in its near-neighbor job, patched and successfully re-executed once; source-only CPU and live Open Pool inventory jobs succeeded without reruns.

- **Real current Kaggle Open Pool:** `crtproduction/name-that-llm-open-pool`, 231,036 passages, 18 labels, 66 source names, sha256 `7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492`; older pinned 2026-11-08 version was 212,623/60. **All seven Phase15 external-unsupported labels now have rows.** Publisher metadata pre-screen 91,754 weak7 rows → **17,142** unflagged rows with license+served-model metadata, but **none have had original API output rights/host provenance independently cleared**. Never mix old/new pools.
- **Phase15 word-nearest detector, original SHA-locked 26,025 legal-filtered external versus 2,699 official:** cosine median all external .108914 vs other official train prompts .085954, but **equal-size external 30-run null .079402**. Ext closest fraction vs official **74.435% raw vs 40.0889% size-matched**. **Zero** near-neighbor word cosine≥.75 across 2,699; exact normalized duplicates zero. This is **no detected lexical near-duplicate**, not proof of semantic prompt independence or no host mismatch.
- **Real same-recipe four-view SVM**: official-only fivefold GroupKFold **0.374857 BA18**; Phase15 external-only (11 classes, official rows completely excluded) **0.300890 BA18 / 0.492365 BA11**; historical official+external **0.405322 BA18**. External examples demonstrably encode transferable signal on the official TRAIN sample, but **0.048322 OOF/Public gap still NOT explained**. Strong self-source data effects not demonstrated on unseen host.
- **Decision:** do NOT train a new source-matched SVM on Gemini/GLM/Kimi rows solely because `match_weight`/CC attribution strings exist. Prior original-provider output-terms exclusions and source routing uncertainty still apply. Inspect original-source rights/protocol before any prospected host-target experiment. No new competition submission, no GPU, no champion gain; preserve 0.702.

---

# 2026-11-20 — Phase7→15 source diff completed; 0.048 gap NOT causally resolved

**Sharper quasi-controlled historical mechanism found (Phase14→15):** Same four-view SVM and same frozen official folds, only provider-terms external admission changes: 43,781 18-label external → 26,025 11-label external. P14 OOF **0.417810**, P15 **0.405322** (−0.012488); omitted 7-class mean recall **−0.192381**, retained 11-class mean **+0.101989**. GLM5.3-Flash −0.380, Gemini3.6 −0.320, Gemini3.8 −0.300. [Source-pinned comparison](artifacts/research/phase7-phase15-source-diff-20261120/RESULT.md). **This quantifies an OOF effect of exclusion, not an identified cause of the 0.048 Public gap**: no Phase14 authenticated Public result; original exclusion was due to provider-terms compliance and must not be reversed on performance grounds. Own LoRA24 A1 347-row older-0.679 control is worse with 24 rescues/66 regressions (paired exact sign p≈1.09e−5 within old cohort); not a 0.702 complementarity claim.


**Source and dataflow conclusion:** [exact diff audit](artifacts/research/phase7-phase15-source-diff-20261120/RESULT.md) identifies **no group-dependent test inference** in either actual scored model. Phase7 is 18-way anchor plus row-level gated 8-class specialist (2,503 external rows; Gemini/GPT), Phase15 is **single 18-way 4-view LinearSVC** (26,025 external rows; only 11 labels Claude/GPT/Qwen). The two candidates change architecture, source and training population simultaneously; **not a one-variable experiment**. Existing actual Phase15 fold audit has 0 official prompt split and 0 normalized external/heldout text exact duplicate, but semantic prompt-bank/source/protocol mismatch remains unknown.

**Quantified strong support-conditioned OOF shift:** 11 externally covered classes average recall **+0.128017** vs Phase7, 7 unsupported classes **−0.110476** (Kimi −0.1933, GLM Flash −0.1867, GLM −0.1600). Full OOF moves 0.370052→0.405322, Public 0.369→0.357. Phase15 OOF−Public **0.048322** has no validated causal decomposition; claiming selection bias only ~0.01–0.02 lacks power/selection-path evidence. Phase7 OOF≈Public on a single score is not proof of unbiased calibration.

**k=2 claim corrected:** [previous CPU sampling](artifacts/research/ntl-sparse-prompt-k2-retrospective-20261120/RESULT.md) on **already fixed row-level predictions** is a low-information variance illustration; its k=2 mean delta approximately equals full OOF **by design**. It does NOT test sparse-group recomputation, and neither Phase7 nor Phase15 needs it. **Do not run more resampling** as a model-validity experiment. 0.702 honest OOF currently remains blocked by trained-asset official-data exposure; no fabricated pseudo-OOF, no GPU/official submission.

---

# 2026-11-20 10:22 MYT — actual NTL k=2 ruler experiment: universal fix REJECTED

- **CPU scientific result, not model gain:** [Actions #44660204293](https://github.com/kolihhan/name-kaggle/actions/runs/44660204293) SUCCESS: fetched original Phase7 and Phase15 authentic, row-aligned 2,699-response/150-prompt historical OOF predictions, 5,000 paired uniform k=2 sparse draws on the same fixed predictions. Full OOF Phase15−Phase7 **+0.035270** despite Public **−0.012**. k=2 mean delta **+0.035148**, with only **595/5000 (11.9%)** draws accidentally selecting the Public-winning Phase7. k=1 **21.69%**; k=4 **3.1%**. **k=2 does NOT systematically fix validation of row-independent models**. Not a refutation for group-dependent Hungarian/centering inference; only two historical models, adaptively spent OOF, no fresh independent cohort/correlation. [Detailed result](artifacts/research/ntl-sparse-prompt-k2-retrospective-20261120/RESULT.md).
- **F1 status correction:** Colab ADC Action [#44544600364](https://github.com/kolihhan/name-kaggle/actions/runs/44544600364) actually allocated a remote session but **CANCELLED** at approx 2h35m after `Connection was lost`. **No C1/C2, no result JSON artifact, no scientific outcome**; don't blindly rerun large uncheckpointed 2,699-row extractor.
- Public best still **0.702 #58514818**; no Kaggle submission, new training, GPU or change to champion. Full 0.702 honest OOF requires more than pretrained-in-sample predictions; defer costly full-fold replication until a candidate justifies it. No Apex 1.0 submission.
---

# NTL F1 decoder-probe execution — 2026-11-19 22:xx MYT

**REAL verified research result:** [fixed five-fold 2,699-row C0 lexical control #44541223988](https://github.com/kolihhan/name-kaggle/actions/runs/44541223988) **SUCCESS**, macro 18-class grouped OOF BA **0.291509**, 150 prompt groups, folds 540/540/540/540/539. [repeat + actual tiny Qwen3Model architecture interface #44541786043](https://github.com/kolihhan/name-kaggle/actions/runs/44541786043) also **SUCCESS**, same exact C0 BA; verified 4-layer hidden-state `AutoModel` API, masked pooling, and train-only scaler/PCA self tests. The tiny model has **random weights**, is CPU only, and is **not a real frozen-pretrained attribution experiment**.

**End-to-end experiment engineering committed:** `scripts/ntl_decoder_probe_extract_colab_20261119.py` (Qwen3-1.7B-Base Apache 2.0 revision SHA `658b6ee46470b73f3e0e04e57fc9fb7031fd7daa`, fp16 T4 512 tokens, 4 hidden layers, private NPZ only), `scripts/ntl_decoder_probe_eval_cpu_20261119.py` (C0 lexical, C1 decoder, C2 decoder + fold-training PCA8; paired prompt-bootstrap and 18 per-class recalls), [single-run Colab notebook](notebooks/ntl_f1_frozen_qwen3_decoder_probe_colab_20261119.ipynb) and [fixed prospective rules/score boundaries](artifacts/research/ntl-f1-decoder-20261119/PREREG_AND_STATUS.md).

**Crucial missing result:** No remote Colab T4 connector is available to this session. **Actual pretrained Qwen3-1.7B inference has NOT RUN, so C1, C2, per-class deltas and any new Public BA are UNKNOWN.** Do not call decoder toolchain correctness a competitive gain or pretend to have used T4. Official best remains **0.702 #58514818**; preserve unchanged. Owner may open the committed Colab notebook, select T4 and Run all to complete C1/C2 with an aggregate-only JSON; no hidden test, score submission, paid APIs or new source data.

**Decision:** continue hypothesis-driven F1; no new perfect external 18-label holdout required to start, but separate DEV mechanism evidence from unseen generalization. Avoid another Apex/postprocessing loop or repeated seed/split sweeps.

---

# NTL — 2026-11-19: first newly unblocked candidate passes genuine CPU diff gate

**Actual execution:** [GitHub Actions #44530471825](https://github.com/kolihhan/name-kaggle/actions/runs/44530471825) **SUCCESS**, after original public Lightning slug returned HTTP 404 in the first source pull. Recovered from the authenticated owner-scored private 0.702 reproduction, which retained the *exact* original executable SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc` over 29 code cells. Source notebook SHA `1a5f41518bb2a346bc86319497c4a93b2438639bb7cb9d6e9faa983d432e0630`. CPU builder made **exactly one executable cell change**, final Apex caller `penalty=0.15 → 1.0`; candidate executable code SHA `9fed26f2d4c0163a121982e0d1ab38ce29f7a70a3ced211d1bf3163898571f33`. Mounts and model unchanged. Builder: [`scripts/ntl_apex_typography_off_onevar_cpu_20261119.py`](scripts/ntl_apex_typography_off_onevar_cpu_20261119.py). **No development GPU, no Kaggle candidate uploaded or competition submitted, and no new score.**

**Decision:** An actual candidate can be engineered without new independent 18-class labels. This validates the distinction between *starting an experiment* and *claiming independent/official improvement*; it does **not** establish that removing typography increases BA. The old 23/216 potential true-label contradictions are spent-data diagnostic risk exposures. No official score claim or no-source model improvement claim. Before any potentially costly one-shot official score, confirm current quota, other active scorers, specific PROBE cadence and run economics; no score-facing act has occurred. Neither this code-quality pass nor continued exploratory data-free checks should be called a medal-level breakthrough. If a bounded score test is not worth the slot, move to a materially different learned attribution objective rather than return to the frozen no-experiment loop.

---

# NTL — 2026-11-19 20:56 MYT: UNBLOCK RESEARCH; previous blanket no-experiment gate was wrong

**Decision:** The owner correctly flagged a repeated deadlock: assistant repeatedly demanded a new provenance-perfect, full 18-class, same-row transfer set **before any experiment**. This confuses research development with final confidence. The repository already prospectively authorized NTL model training and the local sanity-only one-shot candidate gate on 2026-11-17. Canonical [RULES.md](RULES.md) now explicitly restores the correct two-stage standard; do not regress to the blanket pause.

**What can happen NOW:** substantial representation / objective / model architecture experiments using existing official TRAIN under proper prompt-group-disjoint DEV-only reporting, and permitted Colab GPU training on lawfully sourced external data. A *single frozen, one-variable, explicitly risky, official score-facing candidate* may proceed after legal/format/runtime/quota/no-active-scoring preflight; no pristine new full-18 holdout is an absolute requirement. This is NOT a waiver of leakage honesty, source rights, output legality, the seven-day PROBE cadence, or anti-LB-overfitting.

**Next candidate for a bounded, one-variable scored mechanism test:** On the *exact* 0.702 Lightning Apex champion, consider disabling only its hard-coded typography `penalty=0.15` to `penalty=1.0`, preserving all model and evaluation stages. The precise empirical hypothesis is that class-exclusive punctuation claims are generator-invariant when they are not: 23/216 true-label contradictions were observed on a **spent external** cohort, and the source-faithful Apex smoke proved the original 0.15 can cause a decision flip. Equally plausible competing hypothesis: original penalty corrects more errors on actual hidden test than it causes. **Unknown:** real rescue/regression and true 0.702 stage logits. The proper one-shot experimental readout is an authenticated official score vs 0.702, not retrospective pretend 18-way BA. **Kill:** Public <=0.702, no result, infeasible 9h runtime or any other unexpected change. Do not interpret an isolated hidden Public change as causal mechanism proof. Before running, freeze exact diff/code SHA, prove all other code/mounts unchanged and check authenticated 1-slot reserve + any active graders + allowed PROBE cadence. If this single change is not economical or safe, pivot to a materially different model class rather than wait forever for full-18 data. This is a prioritized candidate to assess, **not yet an official submission**.

**Verified account best:** 0.702 (#58514818). Neither Apex typography exposure nor old sparse prompt grouping provides new champion accuracy. Avoid additional no-new-data recovery workflows or weak lexical specialist loops.

---

# NTL — 2026-11-19: full retained-output recovery definitively negative (CPU)

**Official champion remains Public BA 0.702 (#58514818).** [Read-only Kaggle output retrieval + original Apex smoke, terminal SUCCESS #44511538578](https://github.com/kolihhan/name-kaggle/actions/runs/44511538578) recovered **23** scored-kernel output files (including **5 nested**), inspected array files without printing rows or labels, and found **zero NPZ/NPY stage arrays**. Specifically, no retained `final-logits.npz` and no `post-glm.npz`. The historical final18 diagnostic kernel is inaccessible through authenticated API. **No comparable real Apex-stage logits or new BA, no GPU/inference/official submission.**

**Decision:** Do **not** retry output listings, pseudo-logit ablations, weight sweeps or no-typography submissions on synthetic evidence. The source-faithful paired evaluator is ready; it can only score an admissible, new, 18-class labeled same-row cohort with genuine pre-Apex logits and post-GLM labels from a separately justified execution. This is an actual exhausted recovery path, not a promise of score gain.

---

# NOW — NTL 2026-11-19: source-faithful Apex Typography ON/OFF executed (CPU)

**Official Public best: 0.702 (#58514818); no new official score, GPU or Kaggle submission.**

- [Executed original-Apex exact-code smoke #44487815258](https://github.com/kolihhan/name-kaggle/actions/runs/44487815258), [production-accurate paired evaluator](scripts/ntl_apex_typography_paired_20261119.py), [complete bounded report](artifacts/research/apex-typography-20261119/PAIRED_READINESS.md). Original `penalty=0.15` vs absent `penalty=1.0` can flip a synthetic 0.60 Kimi vs 0.35 GLM prediction when text contains `" — "`; no flip when Kimi 0.97 vs GLM 0.02. **Synthetic scores, no real accuracy claim.**
- [Authenticated scored-output inventory #44487147476](https://github.com/kolihhan/name-kaggle/actions/runs/44487147476): available 0.702 and older 0.691 outputs **do not include authentic row-level 18-class pre-Apex logits**, and planned diagnostic final18 kernel was inaccessible. Therefore **NO honest same-row 0.702 original-vs-no-typography rescue/regression result can be computed now**. 23/216 historical true-label typography contradictions from [prior forensic audit](artifacts/research/apex-typography-20261119/RESULT.md) are exposures, not real flipped errors.
- Next single scientific gate: recover a genuine, provenance-verified same-row `FINAL_LOGITS` snapshot and `current post-GLM` labels **on a valid labeled cohort**, then run the pinned `--mode paired` evaluator. **Do not spend exploratory GPU or submit a no-typography fork from synthetic effects alone.** No external-source search, no weak reranker.

---

# NOW — NTL (2026-11-19, Apex forensics completed)

**Authenticated Kaggle Public BA remains 0.702 (#58514818)**, not 0.720; no new official submission, Kaggle GPU or model-quality gain. **Primary science decision: stop the cross-family Apex veto idea; examine true typography-induced flips only with real pre/post stage logits.**

- [Apex typography experiment result](artifacts/research/apex-typography-20261119/RESULT.md) and [terminal CPU Action #44485246316](https://github.com/kolihhan/name-kaggle/actions/runs/44485246316): imported exact 0.702 `pngy_residual.py` after full-code hash verification; `refine` **preserves the four family probability masses** (max numeric discrepancy 8.88e-16) and leaves other classes' scores unchanged.
- The original Apex typography rules potentially penalize the **true class in 23/216** historical 18-label Oleg rows; **5/216** also had a correct previous frozen 0.691 hard prediction. `" — "` alone accounts for 21 such true-label contradictions. **No actual 0.702 prediction flips measured**: the real Apex is a 0.15 SOFT penalty only when a disallowed proposed label occurs; current 0.702 stage logits/truth missing.
- **Correction to earlier quick-and-dirty residual pseudo-ablation:** Standalone `argmax(evidence)` is **invalid** because Kimi and Qwen retain default 1.0 evidence while the fitted residual has only four multi-label family heads. Do **not** cite prior `R=12/216`, `P0=13/216`, `P1=87/216` as actual Apex model performance or use them for algorithm selection.
- **Next needed real evidence:** actual same-row 0.702 `pre_typography` versus `post_typography` predictions with proper logits and reliable truth, preserving the exact strong base/Apex family-score context. Do not submit an unverified no-typography patch, rerun a forbidden exploratory Kaggle GPU, or mine more external corpora. The older Oleg216 is spent, source-constrained, and cannot establish hidden BA improvements.

---

# NTL — 2026-11-19 selective-inference source-path conclusion

**Official best remains Public BA 0.702 (#58514818); target 0.720. No new model training, GPU work or Kaggle submission.**

## Scored champion source and saved output verified
- [Exact source/output audit Action #44478355029](https://github.com/kolihhan/name-kaggle/actions/runs/44478355029) SUCCESS: matches full code SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`. The owned `kolihhan/ntl-0702-apex-pinned-nov18` kernel has only final `submission.parquet` and Python/temporary smoke files in its **listed top-level output**; **no accessible per-row 18-way logits/sibling expert outputs in that listing**. `infer.py` writes `/kaggle/temp/final-logits.npz`, while V1 saves `/kaggle/working/v1/post-glm.npz` (a nested path; listing alone cannot establish whether its bytes were retained).
- [Actual decision-source audit #44479302323](https://github.com/kolihhan/name-kaggle/actions/runs/44479302323) and [final-call trace #44478809220](https://github.com/kolihhan/name-kaggle/actions/runs/44478809220): the frozen code **already contains selection mechanisms**: ATLAS probability-margin option, GSQ agreement veto, GLM V1 ranking by combined selected-class mass with max cover **15%** (and <=90,000 estimated token rows), plus final APEX. Re-implementing bare confidence gating is redundant. The actual `--atlas_margin` setting in launch must be respected; having an option isn't proof it was active on hidden rerun.
- **Important correction to earlier 'inference-only' shorthand:** `apex_postprocess.py` line 97 executes `Residual(labels).fit(train.text, train.model)` in the full scoring path. Thus the reproduced upstream system **does fit an official-train supervised residual head during notebook execution**. It adds no owner-developed model weights, but it is **not literally a no-supervised-fitting pipeline**.
- Final APEX call (full branch) uses `strength=0.1, corrected=False, penalty=0.15`. `apply_arrays` computes a residual-driven new prediction for all rows, then **keeps the incumbent prediction for originally predicted GLM-5.3 / GLM-5.3-Flash** (unless later typography rules change it) and **replaces non-GLM predictions unconditionally** with APEX's preferred label. It also applies a reciprocal near-duplicate prompt-pair probability adjustment and fixed typography exclusions.
- `Kaggle kernels logs` of the private executable source show setup/V1 smoke/last module write, but do **not** contain full hidden-scorer before/after row decisions or a reliable complete hidden row count. Do not claim actual 0.702 hidden Apex change rates from this private run or confuse a private Notebook preview with the separately authenticated scored reexecution.

## Structural optimization target (NO launch yet)

**Apex override calibration, not another external corpus or standalone weak classifier.** Existing strong base+V1 scores and the final APEX residual could be **compared on identical labeled rows** under a truly held-out prompt- and source-disjoint protocol; only then define a simple no-new-model selective veto for overrides that are unsupported by base-family probability. Pin/retain:
- C0: exact scored 0.702 output and `pre-Apex` 18-way logits;
- C1: current unconditional APEX on non-GLM + GLM keep + typography;
- C2 (one candidate, not an optimization sweep): identical APEX but **disallow a cross-family override when the underlying pre-Apex logits prefer the original model family over the proposed family**; a class-independent, parameter-free test, preserving GLM handling and all other stages. This may preserve strong predictions or prevent justified corrections: **effect entirely unknown**.
- Gating requirement: actual side-by-side pre/post logits and truth on admissible same-row, prompt-disjoint, model-served-verified data, and a feasible 9h runtime. The present official hidden output doesn't expose its truth or reusable logits; older official folds are spent and upstream residual fit makes same-train counterfactuals in-sample. **No assertion C2 beats 0.702; no new GPU or submission without a justified prospective test.**

**Research decision:** Stop extra data searches, SVD/hard-negative weak-head training and more status-only workflows. One decisive next model-level action is a production-accurate **Apex-only selective veto ablation with genuine paired data**, not a whole new agent architecture.

---

# NOW — NTL 2026-11-19 (latest completed experiment)

**Official Kaggle Public best remains 0.702**, submission #58514818. Target >=0.720 (gap +0.018). No new Kaggle submission, GPU or independent full-18 transfer result.

## Executed, NOT merely planned: hierarchy+hard-negative five-fold mechanism trial

- [Preregistration](artifacts/research/hier-hardneg-20261119/PREREG.md), [complete scientific result](artifacts/research/hier-hardneg-20261119/RESULT.md), [terminal 5-fold GitHub Action #44477189113](https://github.com/kolihhan/name-kaggle/actions/runs/44477189113) SUCCESS; 15 matched CPU fits, 2699 official TRAIN rows, 150 disjoint prompt groups, 18 classes, no external data.
- Reused grouped-OOF **Flat C0 = 0.278914**, **hierarchy-only CH = 0.275963**, **hierarchy+hard-negative C1 = 0.273741**. Primary C1−CH **−0.002222 BA** (95% prompt-cluster CI **[−0.006667,+0.002227]**), **18 rescues/24 regressions**, cross-family error 918→921, within-family 1036→1039.
- **REJECT C1 v0** (predeclared ≥+.03, positive CI, cross-family preservation and within-family rescue requirements all failed). The experiment uses cheap char-TFIDF+SVD256 + trainable 128d MLP; it does not falsify every hierarchical/contrastive approach nor measure any 0.702 incumbent OOF. No future C1 weight/temp/seed sweep on spent folds.
- **Research pivot, not a new promise to score:** investigate real current 0.702 strong-ensemble *selective inference / disagreement* and whether exact per-stage scores can be read without new model passes. No new dataset search, no weak standalone classifier. Preserve champion until authentic Kaggle verification.

---

# NOW — NTL — 2026-11-19 (Kuala Lumpur)

**Actual official Public BA remains 0.702**, #58514818, improvement +0.011 from previous #58375806 (0.691). Target >=0.720; remaining +0.018. No new official submission or development GPU today.

## Scientific decision: NO QUALIFIED NEW 18-LABEL TRAINING / ENSEMBLE EXPERT

- [Rechecked Maniceet Kaggle API dataset #44472996386](https://github.com/kolihhan/name-kaggle/actions/runs/44472996386): public dataset page advertises 7,753 rows of OpenRouter/OpenAI generations and a `served_model` column, but account's **authenticated `kaggle datasets files` still returns access denied**; public search does not list a downloadable entry. No bytes obtained; **cannot use or infer its real per-row served-ID evidence**. No access bypass.
- [Oleg source-diverse strong-ID audit #43805000854](https://github.com/kolihhan/name-kaggle/actions/runs/43805000854): 9,085 rows across all 18 classes are reported as request-and-returned-model evidence in short view, but **8,658 are from sensitive_topics and just 427 are from prose domain**; non-sensitive prose covers **10/18 classes**, missing Claude Opus 4.6, Gemini 3.6/3.7/3.8, GLM 5.3 and Qwen3.8-Max. The previously tested Oleg216 sensitive-topic selection is spent; [unspent-capacity #44130273610](https://github.com/kolihhan/name-kaggle/actions/runs/44130273610) does not establish a genuinely source-disjoint 18-way panel.
- Other publicly visible GPT sibling traces mirror [Crownelius GPT-5.6](https://huggingface.co/datasets/Crownelius/GPT-5.6-Sol-Luna-Terra-Traces) has **14,467 Sol coding-agent traces, 886 Luna persona-distilled rows, zero Terra** (published metadata). Strong channel/protocol mismatch and missing class; unsuitable for contest-style 40–250-word paragraph sibling classification, even if authentic.
- Latest [OpenPool quality filter #44451612939](https://github.com/kolihhan/name-kaggle/actions/runs/44451612939): 8,338 exact label-string matches after license/protocol/safety/length rules, only 13/18 labels and zero accepted for the five GPT siblings. The [363-row free GPT proxy #44465398268](https://github.com/kolihhan/name-kaggle/actions/runs/44465398268) has 355/363 exact response duplicates with OpenPool, not novel data.
- [100 public notebook source scan #44469878828](https://github.com/kolihhan/name-kaggle/actions/runs/44469878828): only Kirderf 5 candidates have a novel pretrained Gemma4 12B mount. [All 5 are canceled #44471239606](https://github.com/kolihhan/name-kaggle/actions/runs/44471239606), and [required cousin-head asset unverified #44471611805](https://github.com/kolihhan/name-kaggle/actions/runs/44471611805). No safe promotion. [ModernBERT source-adversarial trial #44116912517](https://github.com/kolihhan/name-kaggle/actions/runs/44116912517) worsened independent BA 0.2144→0.0544; retired.

## Next action only when genuine new evidence exists

**Material lever:** independently verified real served GPT-5.5 / Luna / Sol / Terra / Astra labels on prompt- and source-disjoint, license-compliant single-paragraph outputs. For a change that affects *only those five classes*, a +0.018 overall BA improvement requires their mean recall to improve by **0.0648** with no regression on other classes. Gate identity+provenance and paired prompt coverage BEFORE training; then compare to fixed 0.702 champion on same holdout if feasible. Do not substitute inflated source-aware CV for official generalization.

**STOP:** no new 0.702 fork, proxy dataset, GPU preview, guessed 0.720, extra status workflow, or paid API usage. Resumption requires accessible, genuinely new data or an independently demonstrable expert. Historical [NOW at prior commit](https://github.com/kolihhan/name-kaggle/blob/fe52aa52d717dd811f0966d86e65e04383b76e7f/NOW.md); [RULES.md](RULES.md).
