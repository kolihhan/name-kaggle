# 2026-11-23 — ATLAS PAPER A0 REAL TOKEN-LEVEL CPU RUN DISPATCHED; DO NOT CLAIM PENDING SCORE

**REAL EXPERIMENT committed and queued:** [GitHub Actions #45152458986](https://github.com/kolihhan/name-kaggle/actions/runs/45152458986) **QUEUED** when last inspected. Exact author ATLAS bounded multi-temperature argmax-prod token readout implemented in [`scripts/ntl_atlas_exact_token_readout_a0_20261123.py`](scripts/ntl_atlas_exact_token_readout_a0_20261123.py). Reuses **2,699** official TRAIN responses' genuine GLM-5.3 vs GLM-Flash token logprobs from publisher versions v1/v2; compare **300** authentic GLM sibling-labelled historical examples on matched fixed prompts. C_mean = ordinary mean token logprob; C_prod = original ATLAS bounded readout T=1,2,4,8; C_sign = token wins; C_global = previously **0.9433333 binary BA** supervised global-feature strong control, require exact reproduction. Verify all eight published source-file SHA256 digests from prior [successful recovery #45016238278](https://github.com/kolihhan/name-kaggle/actions/runs/45016238278). **New measured C_prod BA: NOT YET AVAILABLE.** Cache is historic repeatedly-used TRAIN, 2-way only, NOT full source-conditioned ATLAS training, not current 0.715 final predictions; cannot assert LB gain. [Frozen experiment plan](docs/research/ntl-atlas-a0-real-20261123.md). No new token-model forwards, paid APIs, foundation-model training or Kaggle submission. Champion 0.715 unchanged.

**Other queued experiments NOW actually completed**: POS syntax [#45146166103](https://github.com/kolihhan/name-kaggle/actions/runs/45146166103) SUCCESS, cross-source historic 216 lexical-free POS order 0.125 BA vs old strong 0.412 BA; **reject as standalone competitive avenue**. SupCon synthetic engineering [#45149856087](https://github.com/kolihhan/name-kaggle/actions/runs/45149856087) SUCCESS with finite gradients; **not a real-label BA validation**. The 0.715 source availability audit [#45148589048](https://github.com/kolihhan/name-kaggle/actions/runs/45148589048) also completed, but did NOT produce independent same-row strong embeddings; real SupCon score remains absent.

---

# 2026-11-23 — ATLAS (2026) ADDED AS NTL RESEARCH CANDIDATE (CO-EQUAL WITH SUPCON)

**Author paper/source verified**: [arXiv 2610.19583](https://arxiv.org/abs/2610.19583), [official implementation](https://github.com/komseok0109/ATLAS). ATLAS uses Qwen2.5-7B frozen causal backbone + per-family source-control prefixes + shared-A/source-specific-B LoRA, trains next-token LM likelihood and compares all source-conditioned per-token likelihoods with bounded multi-temperature argmax-prod. **74.27% accuracy on 5-family MAABench joint domain/version shift; NOT 18-way NTL BA.** 

**NEAR-TERM FIXED-BUDGET RESEARCH:** test *only* the paper's clipped/bounded token-level readout on already-existing SHA-pinned aligned GLM vs GLM-Flash token-score caches, comparing plain mean logprob vs ATLAS argmax-prod T={1,2,4,8}, **ZERO new LLM calls or tokens**. This is **not faithful full ATLAS**, can establish at most a GLM 2-way mechanism result, and **HAS NOT RUN**. Full K-conditioned trained ATLAS would add local GPU computation and new LoRA training, conflict with user's current NTL no-QLoRA / existing-time-budget constraint: retain as DEFERRED, not silently reject the paper. SupCon remains a separate complementary discriminative experiment. [Verified architecture/cost/evaluation comparison](docs/research/ntl-atlas-2026-vs-supcon-vs0715-20261123.md).

**No new measured Public gain / no new competition submission. Official scored Champion = 0.715.**

---

# 2026-11-23 — NTL SOTA PAPER APPLICATION: STRONG EMBEDDING PROMPT-SCL CODE IMPLEMENTED, VALIDATION RUNS QUEUED

**Paper research selection:** Recent 2026 **ATLAS** (arXiv:2610.19583) has reported 74.27% dual-domain/version-shift MAABench accuracy but its multiple source-conditioned LoRA LM likelihoods are expensive (and NOT NTL's 18-class BA). ICML 2025 Sun et al. LLM2Vec model-identity classification gets 97.1% 5-way but requires encoder fine-tuning. Selected the lower-cost **domain-generalization supervised contrastive mechanism** Beigi et al. (arXiv:2407.21264): positives same generator across different prompts, hard negatives different generators within one prompt. This is paper-derived, not a claim of original invention or 2026 SOTA score.

**IMPLEMENTED:** [`scripts/ntl_paper_prompt_supcon_20261123.py`](scripts/ntl_paper_prompt_supcon_20261123.py) CE-only vs CE+SCL 18-class head on FROZEN actual strong neural embeddings; SHA-pinned feature and encoder revision metadata contract; `--external-only` if backbone was already fitted on official TRAIN labels, so cannot publish fake OOF. Same hyperparameters and paired prompts; no paid API, no QLoRA or new Kaggle submission. [Full reproducibility and scientific limits](docs/research/ntl-sota-paper-supcon-strong-head-20261123.md).

**ACTUAL RUN STATUS at last check**: GitHub [source availability #45148589048](https://github.com/kolihhan/name-kaggle/actions/runs/45148589048) QUEUED, [alternate ARM #45149509625](https://github.com/kolihhan/name-kaggle/actions/runs/45149509625) QUEUED, [SupCon code smoke #45149230647](https://github.com/kolihhan/name-kaggle/actions/runs/45149230647) QUEUED. Separate local PyTorch equation-only 36-sample test passed finite loss/gradient, but it DID NOT import this checked-in module and DOES NOT prove training or benchmark BA. Must not claim a job completed, a source feature cache exists, or score improved. Need real 0.715 feature extraction / 18-class independent labels. Champion Official Public **0.715 unchanged**.

---

# 2026-11-23 MYT — NTL PUBLIC HIGH-RANK NOTEBOOKS SOURCE CENSUS: 25/25 STRONG SHARED STACK

**Authenticated real [Actions #45142007503](https://github.com/kolihhan/name-kaggle/actions/runs/45142007503) SUCCESS**, requested Kaggle `scoreDescending` first page (NOTE: CLI CSV OMITTED NUMERIC SCORES), pulled and inspected 25/25 ranked public notebooks and three pinned comparators. **All 25** attach **9 model resources**, include GLM causal likelihood, stylometry, family residual and soft grouping; **22/25** have GLM user-prefix/Fable guard markers; **2/25** have Hungarian marker. **24/25** have exactly the same complete input manifests as our scored **0.715** clone; **17/25** have exact identical nine scoring-module source bodies to it. 25 total notebook code bodies yield **20 distinct SHA**. Markers can exist in non-executed cells, so do NOT infer all run at scoring. Not a census of leaderboard top-10 teams: their methods remain hidden; third-party 2026-11-22 leaderboard snapshot puts #1 ≈0.807, public notebook best ≈0.719. **NO verified individual score column** in API output, titles ≠ actual score receipts.

**Verdict vs 'Generator Fingerprint':** Ordinary stylistic fingerprint is ALREADY present 25/25, hence proposed title alone is **not novel**. Prior our independent token transition (0.1893 vs unigram 0.2445), pretrained token heads (0.1827 vs sparse 0.3671), matched prompt conditional style (0.3015 vs 0.3007) all failed suitable TRAIN OOF incremental gates. Thus no automatic promotion of vague discourse fingerprint; if investigating a novel higher-order cross-source invariant feature, **predeclare direct same-row paired gain** against 0.715, no paid target APIs, no QLoRA, no weak small-model replacement, fixed inference budget and prompt/source-disjoint evidence. Report absent 0.715 paired outputs as BLOCKED; do not silently use mismatched proxy. [Full table and mechanisms](docs/research/ntl-public-top25-mechanisms-vs-fingerprint-20261123.md). **Actual new gain: NONE; best official Public 0.715 unchanged.**

---

# 2026-11-23 MYT — NTL REAL ERROR TOPOLOGY CHECK: 216 SOURCE-SHIFT DIAGNOSTIC; 0.715 FROZEN

**Authenticated new CPU scientific reading:** [Actions #45140011226](https://github.com/kolihhan/name-kaggle/actions/runs/45140011226) SUCCESS, SHA-pinned original external 216 responses + exact historic **0.691** fully scored prediction output. Of 216 (12/18 balanced): **89 exact correct = 0.412037 BA**, **153 correct family = 0.708333 BA**, 64 within-family sibling mistakes and 63 cross-family mistakes; sibling-only perfect truth-oracle ceiling **0.708333 BA**, cross-family-only perfect truth-oracle ceiling **0.703704 BA**. OpenAI 60/60 family vs 33/60 exact; Gemini 31/48 vs 11/48; GLM 6/24 vs 3/24; Kimi 0/12, Qwen 2/12. This test is pre-spent ALL sensitive-topic external source, *not the present 0.715*, and oracle numbers are NOT attainable predictions. No new model or benchmark improvement.

**Access check:** [Actions #45139584293](https://github.com/kolihhan/name-kaggle/actions/runs/45139584293) SUCCESS: 0.715 source-clone kernel output lacks `submission.parquet`, compared named highscore public kernels expose **3-row single-label commit previews**, NOT real scored ~4000-row hidden outputs. **So actual 0.715-vs-0.719 paired errors remain UNAVAILABLE.** Do not infer rescues/regressions from code or preview. One earlier invalid workflow YAML was fixed before this run.

**ACTIVE TECHNICAL JUDGMENT:** model-identity errors are source- and family-dependent. A naive sibling-only target is insufficient on this frozen external cohort; blindly changing family labels also risky. Explore genuinely independent **family-recovery evidence** at a fixed runtime budget, then sibling disambiguation only within credible family; existing 0.715 already has GLM likelihood/style/family residual so a new threshold or queue policy alone is NOT novel. Actual original-generator reference/logprob evidence for Kimi/Qwen is not yet credential/version/terms-proven. [Full verified report](docs/research/ntl-real-family-vs-sibling-216-20261123.md). **NO QLoRA, NO small-model surrogate, NO GPU, NO new official submission, NO scored gain. Official best 0.715.**

---

# 2026-11-23 MYT — NTL 0.715 VS 0.719 PUBLIC OVERFIT AUDIT: NO PROOF; DO NOT CHASE

**Real read-only [GitHub Actions #45137623140](https://github.com/kolihhan/name-kaggle/actions/runs/45137623140) SUCCESS.** Kaggle authenticated owned submission ledger returned **27 entries (25 COMPLETE/1 ERROR/1 PENDING), 21 numeric Public scores, max 0.715**. The public leaderboard CLI returned only 23 lines, no named highscore authors: **their current exact scores and independence are NOT authenticated by this method**. Sanpier official public page shows 13 notebook versions, best 0.717 at V3, while V3 versus newest code are different SHAs. Another repo source audit verified multiple 0.719-title notebooks inherit identical GLM user-prefix/Fable guard structure. This proves shared lineage, not overfit or a competitive gain.

**Official test contract:** hidden roughly 4000; Public roughly 40% (~1600); thus +0.004 Public BA corresponds to ~6.4 net correct **ONLY under equal class sizes**. Actual 18-class count and paired changed rows unknown. Sensitivity example of six net under symmetric discordant null: 20 hypothetical changed rows => two-sided ~0.263; not an actual p-value, cannot infer significance. **No matched independently labeled 0.715 vs 0.719 final inference exists** from this audit. **VERDICT:** Public overfitting is a serious *risk*, not verified fact, and owner's 27 submissions are also Public-selection exposure. Keep 0.715 frozen.

**Active user constraint:** no new QLoRA, tiny model proxy, blind ensemble patches, GPU, or extra official submission. Full proof requires same-row independent labels, per-class rescues/regressions and source/prompt split; absent that, do not claim 0.719 promotion. [Short report](docs/research/ntl-0715-vs0719-overfit-audit-20261123.md). **New competitive score: NONE.**

---

# 2026-11-23 MYT — ACTIVE NTL: STOP SMALL-MODEL/QLORA LANE; HIGH-SCORE SOURCE REALITY CHECK COMPLETE

**LATEST USER DECISION supersedes earlier design-first 14B/8B TCAE and QLoRA suggestions for the CURRENT active research lane:** **NO NEW QLoRA TRAINING, NO DISTILGPT2/1.7B PROXY, NO LEARNING FROM PUBLIC 0.54/0.66 AS COMPETITIVE BASELINE.** Benchmark against existing **official own 0.715** and public candidate sources actually at least the same quality, while separating code-title claims from real scored receipts. Preserve old research documents as historical contracts, not current marching orders.

**NEW REAL EXECUTION:**
- [Authenticated public-code source diff #45136005059](https://github.com/kolihhan/name-kaggle/actions/runs/45136005059) SUCCESS. Owner scored 0.715 source SHA `202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8` **identical** to pulled Estopadilla public V5. Sanpier requested V3 SHA `d526209d0eda5138867324fbb35e11345f1b2bb8dc74d772fcdfb929753a7434`, different from latest SHA `9d4b79ab02d63f9152d57bb6b0cc6448c6423f0b678b06f781225812b58e9173`. Its Kaggle page says best **0.717 V3**. Do not attribute that score to later code.
- [Actual GLM/Fable decision code #45136282372](https://github.com/kolihhan/name-kaggle/actions/runs/45136282372) SUCCESS. Inspected Dennis, Sanpier V3/latest, Areeb 0.719-title, Hcluo 0.719-title, Ranjeet sibling odds. Main lineage is inherited powerful 9-model/26-dataset stack plus user-prefix GLM likelihood, budget-aware ranking and **protected GLM→Claude-Fable returns**. Other notebooks share the same functions; different names ≠ independently tested new models. Kaggle CLI sorted metadata omits numeric score, so 0.719 in a title is NOT a newly authenticated score.
- Full exact hashes, technical deltas, score-evidence limits and priority: [source-lineage report](docs/research/ntl-0715-highscore-decision-lineage-20261123.md).

**CORRECT RESEARCH TARGET:** Genuine independent stage-wise information and budget-sensitive likelihood coverage, with **no more weak-model override of strong predictions**. Existing user 0.715 ALREADY has GLM likelihood, per-family residual, style features, soft grouping; indiscriminate Hungarian override already lowered Public 0.715→0.594. Need row-aligned post-GLM final decision evidence and independent same-row truth to assert gains; if unavailable, say blocked, not claim an improvement. Porting Dennis/Sanpier's exact Fable guard = reproduction/prior art, NOT original innovation.

**RESULT THIS TURN:** code lineage audited and archived; **no training, GPU, new Kaggle submission, new BA score, or Champion modification**. Existing official best 0.715 unchanged.

---

# 2026-11-23 MYT — DESIGN-FIRST NTL PIVOT: FRESH18 IS NOT A PREREQUISITE FOR ARCHITECTURE RESEARCH

**Decision:** Do not block exploratory 18-way mechanism research because a brand-new, publisher-authenticated full-18 source is unavailable. That stricter evidence remains needed to claim **independent full-class source transfer**, not to design or test a new model on official disjoint TRAIN folds. No Public-LB tuning. **Official champion remains 0.715 (#58680531).**

**Critical correction:** subtracting the same common LM log-likelihood from every candidate score cancels under argmax; it cannot improve attribution. ATLAS-style source-conditioned likelihood already exists in prior art. Do not repeat tiny DistilGPT2 128-token heads (0.1827 OOF), hierarchical TFIDF hard-negatives (0.2737), or 0.715 weak-score Hungarian override (official 0.594).

**New research direction (owner-corrected, 2026-11-23):** TCAE-v0 (Token-Conditional Authorship Energy), genuinely *discriminative* token/context interaction on **Qwen3-14B-Base PRIMARY** (8B-Base only on measured GPU infeasibility; no fallback to 1.7B/DistilGPT2), trained against direct **C0** using full observed response hidden states and lexical-only **Clex** on the *same* official 150 prompt-grouped 5-fold splits. Do not compare these expected development OOF scores numerically with 0.715 Public. Freeze true model revision, GPU memory/time and one-shot optimizer budget BEFORE a real run. Planned per-response supervised 18-way CE; no 18× backbone inference or synthetic source data. One hypothesis, no serial patching.

**Executed engineering evidence only:** [final authenticated synthetic CPU Actions #45132778093](https://github.com/kolihhan/name-kaggle/actions/runs/45132778093) SUCCESS, exact commit `de422986ed7825562ad226e1a7b9025f665b4111`: prior-vs-current causal token alignment, masked padding, gradients, interaction vs lexical counterexample and near-matched parameters (**14610** TCAE vs **14624** direct for toy D=48). This is **NOT** real 18-class accuracy, source transfer, pretrained backbone loading, or GPU training; no Kaggle submission. Design/stop gate [here](docs/research/ntl-token-conditional-identity-design-20261123.md); code [here](scripts/ntl_tcae_v0.py). **NEXT:** pin 14B revision and measure actual CUDA/4-bit/hidden-cache feasibility (8B fallback only if this fails), then *one* frozen real 18-class same-row experiment only if viable. No model-quality result from this turn.

---

# 2026-11-23 MYT — NTL FRESH18 ACQUISITION GATE DONE: 12/18 PROXY; SIX NATIVE 404; NO NEW TRAINING

**New actual Actions:** [#45129347799](https://github.com/kolihhan/name-kaggle/actions/runs/45129347799) SUCCESS catalog/quota check (12/18 exact models; six missing; Kaggle Model Proxy remaining daily **$10**, monthly **$100**). [#45129637029](https://github.com/kolihhan/name-kaggle/actions/runs/45129637029) SUCCESS live six native-path probe: all six **HTTP 404**, zero exact returned models, observed **$0** quota debit. All routes bounded; no generated texts saved, no competition data published, no model training, no official submissions.

The six absent labels are **Claude Fable 5, Gemini 3.7 Flash, GLM 5.3, GLM 5.3 Flash, Kimi K3, Qwen 3.8 Max**. Public Vercel AI Gateway catalog lists exact routes for all six but currently connected Vercel **has no accessible teams**, no authenticated gateway/credit and no verified served model IDs. A catalog listing or route-specific 404 does not prove true version availability or identity.

**Decision: full-18 *Kaggle-only* fresh-source generation BLOCKED**. Do not pretend 13/18 outside corpus plus separate GPT source is source-comparable. Prior old **0.683** GLM family misses 33/128 are not measured on latest **0.715**. Next true research gate: a lawful, version-provenanced, previously unused **18-class same-prompt** panel across two independent sources; then blind same-row 0.715 vs one frozen family-identity candidate with recall AND false-positive capture. Actual [contract](docs/research/ntl-fresh18-source-readiness-20261123.md) / [result](docs/research/ntl-fresh18-proxy-readiness-result-20261123.md). **Public best 0.715 unchanged; no BA result this run.**

---

# 2026-11-23 MYT — REAL GLM64 CROSS-SOURCE NEGATIVE RECONFIRMED; MATCHED SOURCE-SHIFT FORENSICS COMPLETED

**Actual new read-only CPU [GitHub Actions #45127329258](https://github.com/kolihhan/name-kaggle/actions/runs/45127329258) SUCCESS**, fixed official TRAIN **150** GLM-5.3/Flash paired prompts versus old locked independent *public publisher* **64** matched-prompt GLM pairs; originals SHA checked, 0 exact response overlaps. Ten frozen style/length measurements plus prompt-cluster 1000x CIs. **Absolute source shift** exists in average word length for both labels (Full −0.1743 CI [−0.3143,−0.0405]; Flash −0.1666 [−0.2974,−0.0366]), Full sentence word count decreases ≈2.35, digits increase ≈1/100 words; however **ALL TEN Full−Flash prompt-paired cross-source signature-change CIs include 0**, and three observed sign flips are inconclusive. No evidence that these superficial shifts *cause* the likelihood routing mistakes. Source metadata: 2 setting values / 55 target word-length values; true served version/pretrain independence unverified. Full [frozen contract](docs/research/ntl-glm-family-source-shift-v1-contract-20261123.md) and [actual report](docs/research/ntl-glm-family-source-shift-v1-result-20261123.md).

**Crucial previously-existing score evidence, NOT a new score:** completed [#43612072194](https://github.com/kolihhan/name-kaggle/actions/runs/43612072194) external 64 pairs: former **0.683 Champion** exact **88/128** (two-label BA .6875) → real GLM conditional likelihood **77/128** (.6015625), **6 rescues / 17 regressions**. Of 128 true GLM texts, **only 95** predicted GLM family, **88 of these 95 already exact-right**, just **7 reachable sibling errors**; **33 true GLM missed family entirely**. Do NOT rerun 64 pair, tune threshold, assume GLM weight pairwise evidence generalizes, or equate the old 0.683 gating with the latest 0.715. Not enough information to distinguish likelihood-intrinsic transfer failures versus gate/distribution confounds. Historical 64 source contains no non-GLM negatives: **cannot validate new family recovery** on it.

**NEXT scientific gate:** rights-cleared genuinely unused *full-18* prompt/source/protocol-audited transfer with true GLM and non-GLM source examples to measure family recall AND false positives; otherwise BLOCKED. No new model training, GPU, official Kaggle submission or transfer promotion. Public Champion **0.715** retained.

---

# 2026-11-23 MYT — GCTH-v1 SOURCE-SPECIFIC PRETRAINED TOKEN-HEAD EXPERIMENT COMPLETED: STOP

**Authentic frozen 18-class 5-fold CPU execution [#45123925475](https://github.com/kolihhan/name-kaggle/actions/runs/45123925475) COMPLETED SUCCESS.** Same-row historical official TRAIN 2,699 prompt-disjoint OOF BA: model-specific Unigram **0.2022620**, genuine pretrained causal context → **18** source-specific normalized 128-token categorical heads **0.1826547**, same-capacity source-context shuffled training **0.1452374**, independent four-view Strong Sparse **0.3670892**. Matched B−A **−0.0196073 BA** (Prompt-bootstrap CI **[−0.0358866, −0.0047886]**, 240 rescues / 293 regressions, only 2/5 fold wins); B−C **+0.0374174** (CI [+0.0199022, +0.0551944], 5/5 fold wins); B−D **−0.1844345**. Mechanism gate **FAIL** (also class safety veto 0.1733 recall loss), competitive gate **FAIL**. Real test total **219.3s CPU**, 171,310 causal token contexts from frozen DistilGPT2 and **2446/2699 (90.6%)** passages truncated to prefix 64 BPE tokens. Shuffled donor integrity ≥98.95% different-response. Full [frozen contract](docs/research/ntl-gcth-v1-contract-20261123.md) / [real result](docs/research/ntl-gcth-v1-result-20261123.md). No leaked raw source content, external generated training, GPU, Kaggle submission, or 0.715 Champion dependency.

**Research decision: RETIRE this exact trained source-conditional 128-category prefix-token head; do not tune it on spent OOF/Public.** B>C is real but not enough: random-context C is deliberately OOD, and B loses to Unigram A (CI excludes 0). Pretrained global token-surprisal FCS-v1 also passed a *mechanism* test yet scored .1952, far under Sparse .3671. Accumulating weak feature heads does not advance official score. Next effort requires *distinct actual-generator* evidence and clean source/protocol lineage, not another surrogate model or ensemble over the same official 2699 rows. Fresh independent 18-class transfer and external generation rights remain blockers, not problems hidden by rerunning GroupKFold. Best authenticated Kaggle Public **0.715** unchanged.

---

# 2026-11-23 MYT — FCS-v1 DISTILGPT2 PRETRAINED NEXT-TOKEN RESEARCH: CONTEXT PASS, STANDALONE COMPETITION FAIL

**Completed authentic GitHub Actions [#45107714630](https://github.com/kolihhan/name-kaggle/actions/runs/45107714630), 5-fold 2699 historical official TRAIN rows / 18 labels / 150 Prompt groups.** Same-row OOF BA: BOS-only unigram **0.1748198**; genuine pretrained next-token contextual probabilities **0.1952026**; shuffled context **0.1718568**; fixed four-view Strong Sparse **0.3670892**. Genuine context vs BOS-only **+0.0203828** (prompt-bootstrap CI [+0.0051852,+0.0363173], 5/5 folds), vs shuffled **+0.0233458** (CI [+0.0066789,+0.0388889], 5/5 folds). Pre-frozen **mechanism gate PASS**, **competitive vs Sparse FAIL** (−0.1718867 BA). No use of 0.715 Champion's weights/predictions or official competition submission.

**Cost/coverage:** DistilGPT2 FP32 CPU real+shuffled forward 535.7 + 534.1 seconds, science 1226.2 seconds; **1381/2699 passages truncated to 127 BPE tokens (51.2%)**. This small pretrained LM is not one of the 18 target generators. Read [frozen contract](docs/research/ntl-fcs-v1-contract-20261123.md) and [verified complete results](docs/research/ntl-fcs-v1-result-20261123.md).

**Research decision:** Real contextual token surprise profile contains additional identity-related information on spent historical OOF, but the global 55-feature representation is severely insufficient for standalone 18-way attribution. Pursue verified source-specific conditional evidence and independent full-18 source/protocol transfer instead of another ensemble/routing tweak. No Public score prediction or new Kaggle submission. Verified account best **0.715** unchanged.

---

# 2026-11-23 MYT — INDEPENDENT GENERATIVE TOKEN-TRANSITION EXPERIMENT COMPLETED: CONTEXT DOES NOT BEAT MARGINAL

**Executed real 18-label, 2699-row, frozen 5-fold historical TRAIN OOF; no 0.715 ensemble, GPU or Kaggle submission.** Generative unigram **0.2444743**, real class-conditional bigram **0.1892916**, shuffled-token-order bigram negative control **0.1537534**, independent prior 4-view Strong Sparse LinearSVC **0.3670892**. Matched same-row bigram vs unigram **−0.0551827 BA** (prompt-bootstrap 95% CI [−0.0718519,−0.0370370]), bigram vs shuffled **+0.0355382** (CI [+0.0188433,+0.0522457], 5/5 folds). Full source and authenticated completed CI: [GitHub Actions #45106442160](https://github.com/kolihhan/name-kaggle/actions/runs/45106442160), 165s scientific elapsed. [Pre-frozen contract](docs/research/ntl-generative-transition-v1-contract-20261123.md) · [final analysis](docs/research/ntl-generative-transition-v1-result-20261123.md).

**Critical interpretation:** B beats shuffled synthetic *unnatural-order* train control, but **loses to the simple unigrams on authentic heldout texts**; the shuffled control alone cannot prove generator-specific contextual value. Structural mechanism gate **FAIL**: this exact ~99 abstract-style-token Markov model cannot replace better source evidence. Do not tune smoothing, bins, context window, model weights or Public LB to rescue it. Actual pretrained generator-specific likelihood is **not ruled out**; existing real GLM 5.3/Flash previous research gives stronger pairwise prior. Next effective target: pretrained contextual generator evidence on a **certified external, source/Prompt/protocol-disjoint full-18** benchmark; current full-18 independent gate is still **BLOCKED**. Historical Kaggle best **0.715** remains unchanged.

---

# 2026-11-23 MYT — INDEPENDENT MPCD v1 CPU STUDY COMPLETE: NO PROMPT-PAIR IDENTITY GAIN

**Real executed 5-fold 2699-row 18-class historical DEV (no 0.715 reuse):** row CE `0.3007407`; matched-prompt column CE `0.3014815`; equal-capacity **shuffled-prompt negative control** `0.3040741`. Genuine matched pairing is `+0.0007408` vs row, `−0.0025926` vs shuffled control; both prompt-bootstrap CIs cross zero, only 2/5 folds improve. Verified real [GitHub Actions #45104027619](https://github.com/kolihhan/name-kaggle/actions/runs/45104027619); [frozen pre-run contract](docs/research/ntl-mpcd-v1-contract-20261123.md) / [full result](docs/research/ntl-mpcd-v1-result-20261123.md).

**Scientific decision:** Retire this specific compressed-char/style *matched-panel conditional CE* setup, not entire original research. Null control outperforms real pairing. Prior prompt-residual ModernBERT LDA source-transfer failed (−0.0358 mean); cross-prompt prototype LoRA v2 transfer failed (+0.0027 but CI spans zero); repeating simple prompt-removal or more paired-loss tuning is not justified. Shift hypothesis to **source-robust generator-specific token likelihood / fine-grained sibling evidence**, with a certified unused 18-class source-disjoint truth benchmark before heavier training. **No Kaggle official submission, no new GPU, no champion changes. Official own 0.715 remains best.**

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
