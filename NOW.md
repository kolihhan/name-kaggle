# NOW

## V47 fixed-window SVC — full preflight passed, submitted once (2026-11-07)

**Official submission accepted, scoring pending.** Candidate `kolihhan/ntl-v47-length40/2` was submitted once as `58078151` at 03:54:38 UTC. Authenticated read-only recheck at **07:11:19 UTC on 2026-11-07** found `SubmissionStatus.PENDING`, blank Public/Private scores, and exactly one matching submission; no new score or scientific decision. [Latest read-only check](https://github.com/kolihhan/name-kaggle/actions/runs/42350194737) · [Accepted record and full-preflight proof](https://github.com/kolihhan/name-kaggle/actions/runs/42336749310)

**Concrete solution.** Apply the existing pooled-external SVC to evenly spaced 40-word windows and average raw margins before its original normalization and weight. Use the published task length bound for the fixed budget. This is the sole scientific change to V47. [Frozen contract](artifacts/research/v47-length40-20261107/CONTRACT.md)

**Verified execution.** All 37 code cells compile, three targeted tests pass, and a complete seven-cell executable patch applies byte-exactly. Kaggle version 2 completed the full original GPU ensemble on the three-row fixture with the preview shortcut bypassed and the final Parquet schema checked. Repeating one 40-word block 1, 2 and 4 times produced margin drift `0.218252` for the original SVC and `0` for the treatment on the real fixture. An independent fitted-weight CPU check passed with original drift `0.271407` and treatment drift `0`. The first private version's cache mount defect was independently diagnosed and repaired using original hub v18 / numeric ID `21594398`.

**Frozen decision.** Compare the official result against V47 `0.666`. Follow up at score ≥`0.671`, close at ≤`0.661`, and call intermediate scores inconclusive. Follow through on submission `58078151`. This is a single diagnostic under the user's latest build-and-submit direction. Independent V47 grouped OOF remains unavailable for a validated promotion. [Exact versions, checks and current result](artifacts/research/v47-length40-20261107/RESULT.md)

## Gemini sibling attribution — requested length is a shortcut (2026-11-07)

The external GPT/Gemini generator draws a target word count from the official training length distribution for the requested label. Each model keeps that target when the reasoning setting changes. In the existing Gemini 3.6/3.8 Flash pilot, the target differed between models for all 20 base prompts. [Upstream generator](https://github.com/RohanExploit/kaggle-name-that-llm-top20/blob/main/data/extra_gpt_gemini/make_tasks.py) · [Pinned data provenance](artifacts/phase2/PHASE2_EXTERNAL_DATA_RECON.md)

**Measured without training.** Match the two low-setting answers to the two default-setting answers for each base prompt. Observed word-length order alone matched **20/20**. Character 3–5-gram cosine matching got **16/20** on full answers and **17/20** after cutting all four answers to the same word count. Across default batches `d1028` and `d1029`, the longer request produced the longer answer on **208/209** unequal-target pairs. There was one answer-length tie and zero reversals. A separate implementation reproduced the character scores and the length count.

**Research decision.** Use a single requested word count shared across models for each base prompt in any follow-up identity test. The exact same-question reference comparison establishes a length shortcut in this source. Equal-length prefixes still come from different original requests, so the remaining matching signal cannot be attributed to model identity alone. The pilot has two model labels and 20 prompts and supplies no Kaggle score estimate.

## Lightning V47 core provider gate — shelved as a diagnostic (2026-11-07)

**Research decision.** Shelve the candidate at the evidence gate. The +0.00557 balanced-accuracy replay used already-consumed LoRA24 rows with known prompt groups, and its prompt-bootstrap interval included zero. It leaves the effect on V47's inferred groups unknown. On every provider disagreement, the rule trusts the no-group prediction even when grouping would correct the provider. Reopen only with paired, independently labeled V47 predictions using inferred groups that show how often grouping corrects versus introduces provider errors.

Built one Notebook from Kaggle V47 script version `365000313` (baseline submission `58008463`, Public `0.666`). Only `core_readout.py` changes. It retains grouped logits on provider agreement and falls back to logits without the `_twin` and `_gauss` votes on disagreement, with the original active-vote denominator. The downstream ensemble receives the selected row. [Frozen contract](artifacts/research/v47-core-provider-gate-20261107/CONTRACT.md)

**Verified locally.** All 37 code cells compile and the 30 input source IDs match V47. Seven focused gate checks passed. A mixed-batch fixture exercised the gate through the original final writer and matched the official Parquet schema. The three-row preview was bypassed in the test harness. The previously archived LoRA24 replay reproduced `0.54676112 → 0.55233159` with 207 rescues and 192 regressions.

**Blocked.** Kaggle authentication, the pinned input mounts and the required GPUs were unavailable in this session, so full V47 checkpoint inference did not run. The candidate lacks fold-safe V47 OOF evidence and a verified transfer cohort under the frozen promotion policy. No new fits, submission or score. Frozen decision is follow up at Public ≥`0.671`, close at ≤`0.661`, inconclusive in between.

Notebook SHA-256 `f33e67004315dcdad424d2230fbf8805caa235b441efeda3b6251637ca5dc836`. Core patch SHA-256 `92dd3179b48e86a5115686e9d1a2d71b4eea76f9b8e9aae9fbfb1db6d2947978`.

## V47 core grouping feasibility — closed (2026-11-07)

The original V47 grouper put **23 of 2,699 passages (0.85%)** in groups that mix prompts, below the predeclared **5%** threshold for preparing the ablation. It grouped **95.59%** of passages and achieved **99.75% pair precision** and **76.79% pair recall** against known prompt IDs. The grouping-off candidate stops here. The result measures prompt recovery on existing training text. The effect on model attribution and hidden test groups remains unknown. [Feasibility result](artifacts/research/v47-core-grouping-feasibility-20261107.md)

## Research PIC decision — close contextual choice CPU v1 (2026-11-06)

**Result.** Recovered the completed five-fold CPU comparison. Phase 7 scored `0.370052`, choice counts `0.370428`, and adding sentence position `0.370798`. Position added one net correct prediction over counts and lost to the baseline under typography normalization. The candidate failed every promotion threshold after baseline reproduction. [Frozen contract](artifacts/research/contextual-choice-cpu-v1/CONTRACT.md) · [Verified result](artifacts/research/contextual-choice-cpu-v1/RESULT.md)

**Decision.** Close the sentence-position proxy at the predeclared gate. The 44 pairs covered many passages, but zero pairs had both forms at both positions under the locked detector. Revisit contextual wording only with a different context signal and credible evidence that both choices occur within comparable contexts before fitting. Kaggle promotion for this candidate is closed. [Mechanism and support audit](artifacts/research/contextual-choice-cpu-v1/RESULT.md#what-the-negative-result-covers)

## Research PIC update — generator-versus-collection-channel identifiability (2026-11-06)

**NEW primary-sourced research decision:** [`artifacts/research/generator-vs-channel-identification-decision-20261106.md`](artifacts/research/generator-vs-channel-identification-decision-20261106.md). Existing 18-label OOF shows different error regimes (Gemini family right 97.5%; GLM 42%), but cannot establish whether underlying signals reflect **generator identity** or **generation/collection channel** (instructions, sampling, version, gateway, formatting). Prompt-disjoint CV does not by itself control channel. Do not derive model-fingerprint claims from per-source BA with different label coverage.

**Specific previous category-error correction:** DCT-v3 operates on a **surrogate Qwen3-8B encoder's token-position hidden states** for a completed passage, not on the original generator's real decoding steps. ICML 2026 `Every Step Counts` measures *diffusion-model decoding trajectories* and does not validate our surrogate-state DCT scheme. Frozen DCT contract remains unchanged and unscored. ATLAS is strong at *family* identity under version/domain shift, but its MAABench trains one version per family; it does not directly solve exact sibling identification in this 18-class Kaggle task.

**Research priority:** Before any architecture/new training, require a genuinely crossed `model × same prompt × generation channel` comparison with exact source-version metadata and an independent heldout design. One public proxy dataset documents some GPT/Gemini temperature variants, but does **not** demonstrate a complete 18-label independent factorial evaluation. No candidate promoted or experiment authorized from this work; no actions, submissions, GPU, or deployment.



## Current mode — Research PIC, no submission/deployment (2026-11-06)

**Current explicit instruction supersedes prior operational "Audit → Submit → Verify" reading**: independently research the Name That LLM science and audit existing evidence; do not train, run experiments, dispatch Actions, submit existing Kaggle notebooks, deploy, or chase live scores without explicit fresh authorization. Do not infer "go" means Kaggle submission.

**NEW completed research evidence (read-only):** `artifacts/research/independent-conditional-attribution-diagnosis-20261106.md` decomposes the **already-archived** original LoRA24 Gate-B 2,699 OOF predictions, independently verified ZIP SHA256 `fa675fcf8ac3860984b449748cf067c3da29389a921417e5e1047bb22a7be8ce`; no models were run or changed. Exact 18-label macro: plain BA `0.546761`, wrong sibling `0.259771`, wrong provider `0.193468`, provider-family correct `0.806532`. Provider-correct for Gemini `0.9750`, GPT `0.8800`, but GLM only `0.4200`. Centered head decreases sibling error `0.018188` but increases cross-provider error `0.017099`, yielding almost no net BA change. **Therefore one blanket "fix siblings" narrative and a global family gate are inadequate.** All these OOF slices are already consumed, not unseen transfer tests.

**Research decision:** Examine conditional identity evidence robustness across prompt/source/generation protocol and separate provider-recognition from within-provider sibling attribution. ATLAS/Profiler already cover likelihood/context approaches; do not claim simply adding them is novel. Do not use group one-per-prompt matching absent evidence test construction supports it. A new candidate requires independent source/version/prompt provenance and a genuine precommitted falsification boundary; **none is authorized for execution today**.

**Competition score context (historical):** own independently developed Public `0.520`; third-party published notebooks reproduced Public `0.666` and `0.668`. Previously sent Apex V1 ref `58045266` was PENDING at last observed 13:37 MYT; this research-only work did not recheck or resubmit it. Published leaderboard results are not causal evidence of method contribution.



## User-selected execution mode: AUDIT → SUBMIT → VERIFY, no own experiments (2026-11-06)

**Official Kaggle latest verified score: 0.668 (ref 58022340, COMPLETE), superseding V47 0.666.** [Authenticated status run 42161796795](https://github.com/kolihhan/name-kaggle/actions/runs/42161796795) confirmed this directly. The notebook author titled it 0.672 but our account scored 0.668. Attribution: externally authored Ozertuu public V3, not our independently developed model (our independent LoRA24 scored 0.520).

**NEW VERIFIED SUBMISSION ACCEPTED:** [GitHub Actions 42162219817](https://github.com/kolihhan/name-kaggle/actions/runs/42162219817) directly submitted exact existing **Ozertuu Dual-SVC Residual Refinement Apex V1** from source SHA `3cc78ce05de7304b1240b3b406f00909690e2183efc04b0ea76e267677c70dbd`, with fully verified published version 1 and appropriate Kaggle metadata. Official Kaggle **submission `58045266`** accepted 2026-11-06 04:23:43 UTC, first observed `PENDING` / blank score. **Do not invent score or resubmit the same version.**

**NEXT**: read-only status of `58045266` against prior 0.668 benchmark; if COMPLETE with blank score, investigate genuine Kaggle runtime exception rather than claim success. Exact audit and links: `artifacts/operations/audit-submit-verify-20261106.md`. No user-side ML training, ablation, new model or custom kernel. Public notebook may run its own fit code during Kaggle hidden execution; this is not a claim of zero remote computation.

**Prior research lane PAUSED by explicit user choice.** Do not resume DCT v3, external source-transfer experiments, sibling probes or prompt audits unless user changes goals. Scope is source audit, unchanged published notebook submission, and official scoring verification only.


## Source-transfer mechanism probe COMPLETED — 2026-11-06

**Actual completed one-shot CPU experiment** [GitHub Actions 42157902705](https://github.com/kolihhan/name-kaggle/actions/runs/42157902705), aggregate ZIP artifact `13475960270` (digest `ea2d07c5ececfb36657a4b218521ab72f911b7d9d1589562b4086b3f17dcd669`). Original frozen contract and full results in `artifacts/research/source-transfer-views-20261106/RESULT.md`.

Eligible-label BA (different labels per source): Aless FULL `0.464769`, LEXICAL `0.457431`, SURFACE `0.388482`; Elliot FULL `0.295586`, LEXICAL `0.297183`, SURFACE `0.199968`. These show SURFACE-only is weaker in both settings, while FULL vs LEXICAL is minimal/inconsistent. Crucial matched-label control: **all three shared GPT-5.6 models have at least as high FULL recall on Elliot as Aless** despite the lower overall Elliot score; do not call absolute score gap a uniform domain-shift penalty.

**DECISION: CLOSE this candidate as a negative mechanism diagnostic; NOT a competition result.** Previously examined source data, different label mixtures, unclear served provenance, non-capacity-matched feature spaces mean no generalization/causal/independent evidence. Do not reuse this pair to fit family-specific gates or search new view weights. No GPU, official data, competition submission, LB use, or adaptive sweep.

**Global next question:** Whether generator-specific information beyond surface and sparse lexical cues lives in token *trajectory* rather than arithmetic mean. DCT-v3 existing frozen candidate is the relevant distinct direction, but its GitHub `RESULT.md` only establishes feasibility, not any scored transfer. No new candidate/experimental claim until that scientifically controlled v1-v3 comparison can actually execute. Earlier prompt-residual LDA failed and v2 prototype contrastive did not pass CI; do not recreate either as a supposedly new idea.



## Independent research pivot — source transfer, not sibling gate (2026-11-06)

**User correction accepted:** stop optimizing an elaborate sibling-information-probe and repeatedly source-screening holdouts; no more source inventories, sibling readout fitting, or claim of independent performance from scrutinized sources.

**Actual action:** dedicated branch research/source-transfer-mechanism-20261106, frozen diagnostic contract at artifacts/research/source-transfer-views-20261106/CONTRACT.md; original Stage-1 external-only 30,133-row A1 training split, exact pinned public source alessandrolair (3,866 rows) and elliotthankinson (8,347 rows) under two generation protocols. One CPU run [GitHub Actions 42157902705](https://github.com/kolihhan/name-kaggle/actions/runs/42157902705) compares fixed Phase14 feature views FULL (char+word+shape+small), LEXICAL (char+word) and SURFACE (shape+small), same LinearSVC. No official Kaggle competition data; no GPU; no submission.

**Evaluation:** these two sources have been inspected before, so diagnostic directional evidence only; not clean independent evaluation or model promotion. Do not change sources, hyperparameters, or test the new idea repeatedly against these numbers. Mechanism bottleneck is unknown until real results; Public 0.666 externally authored whole system cannot be causally attributed to any single feature.

**Restriction status:** A prior prohibition on revisiting style/lexical model classes is REOPENED for one mechanism test. Sibling-expert-v1 is RETIRED as active research (historical audit preserved). No new model trained for competition or Public LB optimization in this pivot.



## Official Kaggle Ozertuu V3 score candidate submitted — 2026-11-05 20:11 MYT

**NEW COMPETITION SUBMISSION: `58022340`, accepted `2026-11-05 12:11:08.823000 UTC` / 20:11 MYT, status `SubmissionStatus.PENDING`, Public/Private scores blank on first authenticated query.** The accepted submission is actual Kaggle competition API `kaggle competitions submit name-that-llm -k ozertuu/0-672-dual-svc-calibrated-blend-proven-3-lora -v 3 -f submission.parquet` from [Actions `42024958522`](https://github.com/kolihhan/name-kaggle/actions/runs/42024958522) (SUCCESS). Source SHA256 explicitly verified before submission: `05367a7f8efd41b2f4959505bc7e2e07a5978b85ee426c08bb5969c60945e2c2`. Its version was discovered read-only in [Actions `42024702664`](https://github.com/kolihhan/name-kaggle/actions/runs/42024702664): public **V3** matches that exact hash. Current author latest notebook drifted to `be98b4e0755e8bf30982b46d9a2c1eb96ebda2f4386402a0476b8502c641ffb8` — don't submit unpinned latest. Kaggle My Submissions was 10 before, so new submission is **11th**; old actual completed and scored champ remains **0.666**, `58008463`. **DO NOT SAY 0.672 IS OUR SCORE YET**: 0.672 is author title/ranked public Notebook indication, and V3's actual score with our Kaggle account is pending. Do not retry unchanged V3.

**Execution:** direct submission of **already-published public Notebook**, no user-side large-model training, no Kaggle `kernels push`, no personal GPU quota use. The original published Notebook does run train-time CPU classifiers (SVC/LogisticRegression on official/pool) inside its hidden inference pipeline, so *not strictly zero fit calls*. It consumes the author's pre-trained multiple LoRA adapter weights. Hidden full rerun success, score, runtime, and GPU memory are not yet known. Published V3 source has `FAST_COMMIT = len(test)<50`, so three-row upstream public commit isn't proof of hidden execution.

**Next:** query exact `kaggle competitions submission 58022340` and `competitions submissions` read-only until terminal; report scored Public vs 0.666 or exact exception. If it scores, compare with pinned V47 only as between-whole-systems benchmark; avoid treating 0.006 as isolated SVC gain. A competing independent research candidate requires explicit mechanism and non-spent transfer evaluation; no new training/retuning simply because one public LB number improved.


## PIC course correction — 2026-11-05: shift from audit-only to score-improving candidate research

**User correction: previous NOT TESTABLE forensic loop is not the actual deliverable.** Actual objective is to advance Name That LLM competitively beyond reproduced public **0.666**, researching WHY a model would generalize and finding a **material new capability**, not running synthetic mechanism proofs or generating audits without a scoring path. Preserve existing source/evaluation provenance as historical evidence, **not** as the next-task gate/loop. Do not train new models or duplicate-submit public kernels merely to appear active.

**New executed market/candidate recon**: authenticated [GitHub Actions 42021952459](https://github.com/kolihhan/name-kaggle/actions/runs/42021952459) queried ranked Kaggle public Notebook candidates on 2026-11-05 (read-only, no GPU). Newly listed [ozertuu 0.672 Dual-SVC Calibrated Blend + Proven 3-LoRA](https://www.kaggle.com/code/ozertuu/0-672-dual-svc-calibrated-blend-proven-3-lora) (notebook title **claims** 0.672; version-specific score not independently checked), [omerfarukmerey 0.668 Verifier + Matched Rows](https://www.kaggle.com/code/omerfarukmerey/0-668-verifier-matched-rows-pool-prompts), and [zbigniewkirchner 0.667 Pairwise Verifier](https://www.kaggle.com/code/zbigniewkirchner/0-667-pairwise-verifier-matched-external-rows).

[GitHub Actions 42022226813](https://github.com/kolihhan/name-kaggle/actions/runs/42022226813) successfully pulled all three **public source notebooks** and verified complete/notebook source access, offline GPU flags, and competition source; published source artifact `13411503888` retained briefly. The 0.672 Notebook SHA256 is `05367a7f8efd41b2f4959505bc7e2e07a5978b85ee426c08bb5969c60945e2c2`, 9 code cells, 19 datasets, eight notebook sources, eight model sources. It has a `FAST_COMMIT = len(test) < 50` shortcut, so the public 3-row notebook run cannot prove the full hidden-sized path. Source contains sklearn fit calls: **do not claim it is strictly inference-only**. Version-pinned actual author score (as opposed to title) and real hidden runtime still need verification before even considering a user-side competition submission.

**Source diff of 0.668 -> currently published 0.672 confirms substantive changes:** `infer.py` adds externally pre-fitted `ours_svc4n` SVC with reconstructible TF-IDF views + optional `top34` fitted LightGBM readout; run config changes `svc_w` 12 -> 10, `ours_svc_w=1.2`, `top34_w=0.20`; `n14b` changes to a **three-adapter blend** with weights `[0.34,0.33,0.33]` and model weight 2.5 -> 2.8; Atlas gate becomes `atlas_w .22`, `atlas_margin .36` instead of .20/.32. These are simultaneous changes: **no causal attribution** or validated +0.006 gain from source diff alone. Other 0.672 code has full hidden-run branch and many attached packages. New candidates are for **competitive mechanism scouting**, not permission for an automatic copied public submission.

**High-value next scientific/competitive decision**: decide a **materially new, feasible pipeline** from source differences, with a fixed pre-declared transfer evaluation and an executable scoring path. Challenge the high-complexity public LoRA/SVC lineage rather than chase 0.001 leaderboard tweaks. A separate strong prior is the original independently developed 0.520 LoRA24. No training, inference run, or Kaggle competition submission occurred during this pivot.


## PIC follow-up: V47 disjoint transfer gate + exact-code family gate CPU test — 2026-11-05

**Completed read-only data availability gate: NOT TESTABLE as an empirical paired ablation with existing evidence.** Inspected source-pinned V47 public Notebook, original Gate-B Actions artifacts `41398543856/13125872633` (2,699 official OOF predictions), original Gate-A artifact `41333631967/13094213203` (one aggregate JSON, **no per-row external predictions**), frozen DCT-v3 `14,506`-row external split (locked **for v3, not independently proven heldout against V47**), and authenticated completed Kaggle score `58008463 / 0.666`. None supplies both row-aligned **and provably disjoint** per-component labels/predictions for both compared models. V47 code creates `core.npz`, `complementary.npy`, `pre-balance.npz` at inference, but the inspected public source artifact has none of these on an independent labeled holdout. **Do not compare fitted V47 final checkpoints on official 2,699 train rows as if they were unbiased OOF. Do not use Public LB as an ablation loop.**

**Exact implementation mechanism PROVEN on synthetic logits, NOT accuracy:** added `scripts/audit_v47_family_gate_cpu.py` and `.github/workflows/ntl-v47-family-gate-source-pinned-cpu-audit.yml`; [Actions **42021098034**](https://github.com/kolihhan/name-kaggle/actions/runs/42021098034) **SUCCESS**, source SHA locked `be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d`, exact family-gate function AST executed on 384 synthetic rows for all five gates. Family log-mass drift at most `4.44e-16`, outside-family logit drift 0, all 64 high-confidence controls unchanged. Important counterexample: top-1 `gpt-5.5` -> `claude-fable-5` is possible even as family mass stays constant; gate is **not top-1 family invariant**. Artifact `13411512310` contains JSON aggregate proof, never official data.

Full evidence and stop/reopen criteria: [`artifacts/research/ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md`](artifacts/research/ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md). **No training, no GPU inference, no new Kaggle submission.** Need genuine source/prompt independence proof + aligned predictions before any real paired effect-size claim; current strongest valid scored public version stays the externally authored reproduced 0.666; independent team's own research route remains 0.520.


## V47 vs own LoRA24 mechanism investigation — 2026-11-05

**COMPLETED: exact-source static comparative audit.** Report: [`artifacts/research/ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md`](artifacts/research/ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md). Official score `58008463 / COMPLETE / 0.666` was verified at 18:51 MYT by [official run **42014195583**](https://github.com/kolihhan/name-kaggle/actions/runs/42014195583), vs independent project 0.520, +0.146 absolute. The external source is public Lightning V47; **attribute it to the original author**.

Read exact published v47 Notebook SHA `be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d` from temporary public Apache-2 source artifact [42016381979](https://github.com/kolihhan/name-kaggle/actions/runs/42016381979); 11 local static-source consistency assertions PASS. Concrete differences from our 0.520 single Qwen3-8B LoRA24 + plain head: V47 has six diverse neural backbone/member streams, three fitted sparse core readouts (including strong SVC4), confidence-gated family-sibling adjustments, inferred test-group centered/Gaussian features, complementary large ensemble, optional Atlas, class balancing, and typography evidence. **Important correction: full hidden-route GSQ Qwen3.8-27B probability share is only 5% (initial config 7% overwritten to 5% in cell 45)**. Pre-postprocessing core/complement/Atlas/GSQ shares depend on Atlas gate; details in report. Thus the +0.146 cannot honestly be attributed to 27B alone.

**Evidence limitation:** this is a comparison of two entirely different systems on the Kaggle public 40% leaderboard, not a controlled ablation or independent private transfer. V47 has material extra data provenance and source-scale differences; exact train/validation overlap and causal component gains are **unknown**. No GPU inference, retraining, extra Kaggle competition submission, or tuning was performed in this audit. Recommended next scientific check is a frozen, paired component comparison using genuinely disjoint source-bank labeled rows **only if** already-existing component outputs and provenance support it; otherwise report NOT TESTABLE.


## Verified Kaggle V47 public score — 2026-11-05 18:51 MYT

**OFFICIAL SCORING SUCCESS.** Fresh authenticated Kaggle competition API detail returned `Submission Ref: 58008463`, `Status: COMPLETE`, and **`Public Score: 0.666`** at `2026-11-05T10:51:53Z` (18:51 MYT), read-only GitHub Actions [42014195583](https://github.com/kolihhan/name-kaggle/actions/runs/42014195583). The authenticated submissions listing independently matches the exact ref and `SubmissionStatus.COMPLETE`, `publicScore: "0.666"`, newest ref among 10 records. **Own best Public score is now 0.666**, vs previous own best 0.520 (**+0.146 absolute; +28.08% relative**).

This was the original already-published public Lightning V47 Notebook `lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble/47`, source SHA256 `be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d`, directly submitted to Kaggle code competition via `kaggle competitions submit -k ... -v 47` in [41963828666](https://github.com/kolihhan/name-kaggle/actions/runs/41963828666). **No training or Kaggle GPU notebook push by us.** Do not claim this is an independently invented model or the score was achieved by prior experimental repo candidate; attribution remains with the original public Notebook author. Kaggle official public-scoring is proven; private score remains blank/unavailable. **No need for another duplicate submit, GPU reset wait, or retry.** The previous PENDING snapshot is superseded.


## PIC official DIRECT submission — 2026-11-05 11:48 MYT

**Confirmed new competition submission ID: `58008463`**, created `2026-11-05 03:48:15.950000 UTC` from exact pre-existing published upstream Notebook `lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble/47`. The source SHA-256 `be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d` was checked before submit. The official Kaggle `kaggle competitions submit -k ... -v 47 -f submission.parquet` API accepted the submission. Authenticated read-only [run 41963909784](https://github.com/kolihhan/name-kaggle/actions/runs/41963909784) shows **10 total official submissions**, latest `58008463 / SubmissionStatus.PENDING / Public Score empty` and detailed `kaggle competitions submission 58008463` confirms `PENDING`. The exact submission execution is [GitHub Actions 41963828666](https://github.com/kolihhan/name-kaggle/actions/runs/41963828666), which polls official score. **Own scored best remains 0.520 until an actual new nonblank score appears.** No score comparison yet; published upstream 0.666 is NOT ours.

**Important corrected mental model:** Direct code-competition submission of an already-published Kaggle Notebook version **does not require personal weekly GPU quota**; only a new author-side Kaggle GPU Notebook push/commit or our self-imposed private inference preflight does. Kaggle performs code competition hidden rerun on separate evaluation compute. The prior GPU preflight was unnecessary for this user's direct submission goal. Therefore the wrong planned November 7 GPU-gated cron workflow on `main` has been **disabled** in commit `851702ec209afd4c497f93d3078bcf9d8a65edf9`; no GPU reset waiting needed. We did NOT train any model, push GPU Notebook, or burn personal GPU allocation in direct submit. Do not submit this V47 again while `58008463` exists; if it ends with blank score inspect `kaggle competitions submission 58008463` and do not retry unchanged source.


## PIC execution checkpoint — 2026-11-05 MYT: GPU hard block, one-shot continuation armed

**Formal competition result:** No new accepted competition submission. Authenticated Kaggle [snapshot Actions 41961137677](https://github.com/kolihhan/name-kaggle/actions/runs/41961137677) still has **9 total** submissions, newest `57985641` with `SubmissionStatus.COMPLETE` and both score fields empty. Best actually scored own submission remains **0.520**; delta from proposed V47 is **unknown**, not +0.146.

**Current Kaggle GPU quota (authoritative):** `GPU used 31.79h | remaining 0.00h | total 30.00h | refreshAt 2026-11-07T00:00:00+00:00` (**2026-11-07 08:00 Asia/Kuala_Lumpur**). [Quota log 41961137677](https://github.com/kolihhan/name-kaggle/actions/runs/41961137677). Do **not** burn more pushes before quota restoration.

**Exact V47** remains `lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble/47` with pinned notebook SHA-256 `be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d`. Original upstream-reported 0.666 is **not** this team's score. Full-inference proof is still **missing**; no official submit.

**Defect fixed, regression tested:** [Actions 41956709567](https://github.com/kolihhan/name-kaggle/actions/runs/41956709567) observed Kaggle CLI return zero while printing `Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.`. The previous script incorrectly continued against an old `KernelWorkerStatus.COMPLETE` and failed `Missing proof or parquet: proof=0 parquet=0`. `scripts/ntl_v47_full_preflight_submit_once.py` now requires a new `Kernel version N successfully pushed` acknowledgement, fails closed on quota/403 and validates predicted labels against the 18 actual official labels rather than an assumed hardcoded set. [Actions 41960701231](https://github.com/kolihhan/name-kaggle/actions/runs/41960701231): five zero-network guard tests PASS. No training executed.

**Stale runs canceled:** Pending `41956662951` and stuck older `41955673124` both report `cancelled`, via read-only/no-Kaggle GitHub concurrency stopper workflows `41960971704` and `41961054354`. This prevents old scripts from attempting duplicate pushes when quota refreshes.

**One-time continuation configured on default main branch:** [`.github/workflows/ntl-v47-one-shot-quota-refresh-20261107.yml`](https://github.com/kolihhan/name-kaggle/blob/main/.github/workflows/ntl-v47-one-shot-quota-refresh-20261107.yml), added at commit `966e06dc0b77a1ca9f9a382c62467f33e7cb6606`, is scheduled for **2026-11-07 00:17 UTC / 08:17 MYT**. It checks the exact 2026 date, live Kaggle GPU quota (requires >=4h), sparse-checks-out the tested script pinned at `f05ebd6d3852663bd903d2e4ee6b592fd2c02f8b`, runs no-network tests, and only then attempts genuine private GPU inference and sample `submission.parquet` validation. **Only if the gate passes**, that script calls the real Kaggle competition submit API **once**, deduplicates against My Submissions, and polls for nonblank Public Score or reports the raw failure. The annual cron entry is inert on other years because of the exact-date guard; it is not evidence that a future execution or score succeeded. Do not trigger any parallel attempts and do not claim success in advance. ChatGPT scheduled-task limit was full (5/5), so no separate ChatGPT automation was created.


## PIC score attempt — 2026-11-05 10:29 MYT (no new Public score)

**Actual Kaggle competition submission: NOT CREATED.** Most recent authenticated `My Submissions` snapshot [Actions 41863313129](https://github.com/kolihhan/name-kaggle/actions/runs/41863313129) contains exactly 9 records, latest `57985641` (`COMPLETE`, both score fields blank), previous `57983359` (`COMPLETE`, both blank; UI reports `Notebook Threw Exception`). Best scored own result remains `0.520` (`57908683` and its duplicate retries). Difference for a new candidate is **unknown** until Kaggle actually scores a new competition submission.

**Pinned public candidate:** Lightning `lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble/47`, SHA-256 `be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d`. Source audit [41953332095](https://github.com/kolihhan/name-kaggle/actions/runs/41953332095) verified 37 code cells, disabled internet, configured GPU, attached public input families, no explicit backward training, and exactly one **3-row known-fixture preview shortcut**. Published author Public `0.666 V47` is not our score. Hidden full inference is materially different from Cosmorin V1/V3 (Qwen3.8-27B and many additional model components).

**Full-sample gate NOT passed:** Strict no-training private preflight and optional once-only formal submission are implemented in `scripts/ntl_v47_full_preflight_submit_once.py` and `.github/workflows/ntl-lightning-v47-full-preflight-submit-once.yml`. Actual attempts [41953754311](https://github.com/kolihhan/name-kaggle/actions/runs/41953754311) and [41953948794](https://github.com/kolihhan/name-kaggle/actions/runs/41953948794) were rejected by the Kaggle `SaveKernel` API with **`403 Client Error: Forbidden`**; first also contained a title/slug warning. Fixing the mismatch did not remove 403. As the full-sample GPU inference did not execute, the guarded official `kaggle competitions submit` command was never reached. Do not classify this as a scored result, training success, Kaggle quota failure, or a hidden rerun traceback; root cause currently unknown.

**Access checks:** [41954191222](https://github.com/kolihhan/name-kaggle/actions/runs/41954191222) verified readable metadata for all 16 V47 datasets and all eight referenced notebooks; [41954755867](https://github.com/kolihhan/name-kaggle/actions/runs/41954755867) verified all five Qwen model-version file listings. Read access does not prove attachment authorization. An empty no-data private CPU canary **successfully pushed version 1** in [41954918109](https://github.com/kolihhan/name-kaggle/actions/runs/41954918109), proving general Kaggle private-kernel save access. [Source-family bisect 41955074478](https://github.com/kolihhan/name-kaggle/actions/runs/41955074478) tests whether specific V47 attachment families trigger the 403; it performs **no competition submission**. Do not retry identical V47 full-preflight push without new permission/packaging evidence.

**Alternative assessed:** `lightningv08/0-595-two-stage-lora-gaussian-olmo2/14` source SHA `1f33d5866b68531e5eef49a20e9ac586e5a20b0aa52eb9d7e9fd491797968c3b`. Published 0.595 version requires supervised Gaussian fit on official train at runtime, so it is **not** a strictly prefit-only no-training fallback; not submitted. The original public writer's score cannot be credited to us.

## Stage-1 LoRA24 heldout GPU execution — 2026-11-04 23:37 MYT

**KAGGLE PRIVATE KERNEL PUSH ACCEPTED; MODEL INFERENCE STILL RUNNING, NO RESULT YET.** One-shot Actions run [41859340827](https://github.com/kolihhan/name-kaggle/actions/runs/41859340827), on branch `exp/dct-temporal-fingerprint-v3`, commit `d57552602f0a3e93990128bb58b6b396b3e73467`, completed static tests and passed the Kaggle private kernel push step; the follow-on aggregate-only output retrieval is actively polling Kaggle. No competition submission or training.

Root cause of previous execution failure: [41858850656](https://github.com/kolihhan/name-kaggle/actions/runs/41858850656) reached GPU job but died **on GitHub runner before Kaggle push** with `ModuleNotFoundError: No module named 'numpy'`. The standalone inference bundle builder transitively imported NumPy through an unrelated full-training submission builder. Fix `69c5edaa9614302d725d3260551d2567d1ee129d` isolates the offline wheel list from training imports; regression test in `tests/test_build_lora24_sibling_holdout_kernel.py` builds with `python -S` (stdlib only). Corrected one-shot runner is active. The earlier failure **did not consume GPU quota**.

Scientific interpretation remains strictly limited: Stage-1 A1 is **not** the final Public-0.520 Stage-2 model; original Gate-A heldout sources were consumed during model selection. The upcoming diagnostics are descriptive within-family confusion, not a fresh blind generalization test and not a new leaderboard candidate. Do not retune on the result, auto-retry, train or submit anything to Kaggle competition. Later result must be verified from the GitHub Actions aggregate artifact rather than inferred from a successful push.

## Stage-1 LoRA24 sibling-holdout inference implementation — 2026-11-04

**IMPLEMENTED AND STATIC VERIFIED; GPU INFERENCE NOT EXECUTED.** User's "go" authorized the scoped source-heldout inference-only validation, not a Kaggle competition submission or training.

- Source: `scripts/audit_lora24_sibling_holdout.py`, `scripts/lora24_sibling_holdout_inference.py`, `scripts/build_lora24_sibling_holdout_kernel.py`. Synthetic tests: `tests/test_lora24_sibling_holdout.py` and `tests/test_build_lora24_sibling_holdout_kernel.py`.
- GitHub Actions workflow: [41857723140](https://github.com/kolihhan/name-kaggle/actions/runs/41857723140) **SUCCESS**, `6 passed`, generated offline Kaggle source compiled, and metadata/security preflight `STATIC_SOURCE_HOLDOUT_INFERENCE_GATE_PASS`. GPU job was **SKIPPED by design** on a push; only `workflow_dispatch` can start it. This is NOT empirical GPU inference or a source-transfer confusion result.
- Frozen external data: original SHA-pinned `nishantdahal/ntl-handwriting-blend/pool_v4.parquet`; existing Gate-A source split `abdullah/semenoff/zakir`, 3,324 eligible rows / 13 eligible labels. Exact locked Stage-1 A1 checkpoint `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint` v1, manifest `c9ba8f77...`; Qwen3-8B Base, 24 layers. No official competition input sources mounted, private Kernel, internet off, aggregate outputs only uploaded to GitHub.
- Crucial scientific limit: Stage-1 A1 is **not** the final Stage-2 model behind Public `0.520`. The inference diagnosis can describe sibling-family errors across already-consumed Stage-1 sources but cannot independently establish the final 0.520 model's hidden transfer. The original Gate-A sources were previously used for model decision and are not a newly untouched holdout.
- Since Kaggle weekly GPU quota was previously exhausted and no fresh positive GPU eligibility evidence exists, **do not trigger repeated quota-rejected pushes**. New workflow is prepared for **one manual private inference run** when a valid GPU execution window is confirmed. No new competition submission or training authorized.
- Scientific result remains **UNRUN / UNKNOWN**. Do not claim 57.32% replicated, any new Public score, or any performance gain.

## CURRENT DECISION — ACTIVE NTL RESEARCH, COSMORIN LOOP CLOSED (2026-11-04)

**CORRECTION TO MY PREVIOUS MISINTERPRETATION:** The user said to stop repeatedly hitting the same failed 0.622 Notebook route, **not** to abandon Name That LLM or switch to Gemma 4. The unjustified stop directive added to `main/AGENTS.md` has been reverted in commit `c11233ac847bab8416d110583ed3640b3f535f31`. This is a Name That LLM-only research continuation.

- **Retire only the repeated external Cosmorin 0.622 V1/V3 submit/probe route.** The historical workflow, experiments, results, and failed submission IDs remain preserved below. Do not retry the identical frozen source or consume more Kaggle attempts to diagnose it.
- Current verified best **own** Public LB remains **0.520** from `two-stage-lora24-v1`; its honest OOF plain BA was `0.546761` and the original Stage-1 source-transfer gate passed.
- **New evidence from original frozen Gate-B OOF predictions:** 1,223 plain errors, 701 (57.32%) inside the same model/provider family; centered head rescued 353 but broke 350. This suggests sibling-model discrimination as a research direction, **not** that centered or label-specific correction should be promoted. Dataset/label-level metrics are previously consumed OOF evidence, not a fresh transfer set.
- **Read-only source-evidence check completed:** original held-out Gate-A artifact `13094213203` contains aggregate A0/A1 scores only, **no per-row predictions/embeddings**; A1 checkpoint artifact `13094069155` is not a held-out predictions export. Therefore sibling-confusion transfer is **UNVERIFIED** and the no-training audit stops here. Next independently useful step would require a separately specified inference-only held-out prediction export, not a repeat submission or model training.
- Keep no-self-training and no-unjustified-submission boundaries. A materially different representation or inference candidate must pass a frozen independent transfer check before any competition submission; no Pub-LB tuning.
- Detailed evidence and fact/inference/unknown separation: [`artifacts/research/ntl-sibling-confusion-pivot-2026-11-04.md`](artifacts/research/ntl-sibling-confusion-pivot-2026-11-04.md).

## Latest read-only score check — 2026-11-04 21:45 MYT

- Submission **57985641**: authenticated Kaggle competition submissions API returns `SubmissionStatus.COMPLETE`, but **Public Score and Private Score are both blank**. Evidence: successful read-only GitHub Actions [41841013584](https://github.com/kolihhan/name-kaggle/actions/runs/41841013584), polled `2026-11-04 13:45:37 UTC`.
- **No valid new score; prior best remains `0.520`.** This result is not evidence of successful scoring. Submission `57983359` had the same API symptom and the user's Kaggle UI showed `Notebook Threw Exception`.
- **Actual `57985641` detailed failure category is UNCONFIRMED**: the Kaggle submissions listing is insufficient to distinguish `Notebook Threw Exception` from another invalid-run outcome. Since this retry uses the same V1 inference source as the failed entry, repeat exception is plausible but not verified.
- No model retraining or additional submission occurred in this check. **Do not re-submit identical source again**; inspect exact Kaggle submission details or perform targeted isolated GPU runtime debugging if available.

Updated: 2026-11-04 MYT

## Authorized one-more Kaggle submission — 2026-11-04 21:09 MYT

**NEW SUBMISSION ACCEPTED, PENDING SCORING, NO TRAINING.** User explicitly requested one additional competition submission after the four CPU diagnostics. Preserved the exact published prefit pipeline; no model weights or competition data committed to GitHub, no self-training, no GitHub-side GPU Notebook push.

- First controlled alternative: Cosmorin V3 (`cosmorin/name-that-llm-public620-reproduction/3`) has the same executable V1 code SHA `1595f543caf314fc766a79dca313961af4d73f2a5cbea1ec2b4724aea5c4289b` but different Kaggle Docker image. A direct competition submission attempt via [Actions 41835333341](https://github.com/kolihhan/name-kaggle/actions/runs/41835333341) was **rejected by Kaggle API with HTTP 400 Bad Request**; **no submission record / no score** for V3. Specific error reason not exposed.
- Final direct V1 retry (unchanged exact notebook source, original publisher and assets): [Actions 41835523954](https://github.com/kolihhan/name-kaggle/actions/runs/41835523954) **SUCCESS**. Kaggle CLI printed **`Submission ref: 57985641`** and **3 submissions remaining today**. Initial authoritative list shows `57985641, SubmissionStatus.PENDING, publicScore blank`, timestamp `2026-11-04 13:09:32.673000 UTC`. Submission description: `Cosmorin V1 repeat after CPU diagnostics 57983359 no training`.
- **IMPORTANT:** This V1 code is identical to the failed submission `57983359`. CPU preflight PASS did not fix the unknown hidden-run exception. Do not claim a valid score until Kaggle's submission detail confirms no `Notebook Threw Exception` and the actual `publicScore` is nonblank. CLI `SubmissionStatus.COMPLETE` with blank score was previously a failed rerun.
- One authorized additional submission has been created. **Do not submit again automatically**. Next action: read-only check `57985641` score *and* detailed failure state when available. Until then the best verified valid Public score remains `0.520`.

## Kaggle #57983359 postmortem — four private CPU diagnostics completed (2026-11-04 MYT)

**STATUS: PRE-MODEL + FULLSIZE CPU PASS; GPU / HIDDEN RERUN ROOT CAUSE UNKNOWN.** No training, no new official competition submission, no base-weight downloads to GitHub, and no changes to frozen ML/validation contracts. These probes executed privately on Kaggle CPU through GitHub Actions. They cannot reproduce actual hidden samples or two 14B GPU inference passes.

| Layer | Run | Observed result | Limits |
|---|---|---|---|
| 1. Kaggle environment/input hashes | [41828297015](https://github.com/kolihhan/name-kaggle/actions/runs/41828297015) — success | Exact `numpy=2.1.3`, `scipy=1.16.3`, `scikit-learn=1.6.1`, `joblib=1.6.0` matched SVC cache manifest; `train.csv`, `pool_v4.parquet`, `svc_prefit.joblib` SHA-256: **PASS**; all 5 adapter/encoder small JSON config files mounted once and readable | These are CPU image checks; hidden scoring GPU image and asset state may differ |
| 2. SVC on invented passages + safetensors tensor headers | [41828879403](https://github.com/kolihhan/name-kaggle/actions/runs/41828879403) — success | SHA-pinned SVC cache loaded; four views produced finite shape `3x18`; Qwen3 adapter: 189 matching A/B keys; Qwen2.5: 224 matching A/B keys; both have valid rank/pairs, plain+centered heads; DeBERTa classification head header valid | Safetensors headers only; no neural tensors loaded or model forward |
| 3. Full-size CPU algorithm path | [41829562049](https://github.com/kolihhan/name-kaggle/actions/runs/41829562049) — success | 4,000 fabricated passages; full prefitted SVC + TFIDF all-pairs average-linkage + 300-round Sinkhorn finite; SVC 2.51 s, cluster 1.11 s, Sinkhorn 0.18 s; total 9.41 s, maxRSS ~547.8 MiB | Synthetic clusters produced **0 centered-twin rows**; true hidden text, neural ensemble, and GPU inference not covered; these measurements are NOT full runtime estimates |
| 4. Exact two pinned base Qwen model mounts/config/tokenizer | [41830046480](https://github.com/kolihhan/name-kaggle/actions/runs/41830046480) — success | Both `qwen3` / `qwen2` configs unique and matched required depth/hidden size; keep/layer indices/eos valid; both tokenizers worked on invented text. Formula in scored source estimates truncated Qwen3 **18.06 GiB** and Qwen2.5 **17.86 GiB** fp16 model weights, each processed sequentially | No model-weight loading/forward, LoRA merge, DeBERTa forward, or GPU memory/hidden test sizing |

### Diagnosis and decision

- **FACT:** The official submission `57983359` failed with Kaggle UI detail `Notebook Threw Exception`; the CLI's `COMPLETE` only meant processing finished. No score; the team's valid Public best remains `0.520`.
- **FACT:** Current Kaggle private CPU environment passes exact cache package versions/hash checks, SVC prediction, two adapter tensor-header/heads checks, 4,000-row synthetic cluster/Sinkhorn, and base config/tokenizer mounting.
- **INFERENCE:** The highest-value remaining investigation targets **GPU-only model loading/sharding, LoRA merging, long-input batches, DeBERTa forward, or different hidden-rerun environment**. Static 18 GiB/trunk estimates imply two-GPU sharding and substantial additional activation/adapter memory, but **GPU OOM is not proven**.
- **UNKNOWN:** Hidden-exception line/trace, runtime memory peak, whether hidden scoring uses the same image and mounts as these CPU tests, whether long hidden passages trigger uncaught single-row CUDA OOM, and actual Private/Public transfer.
- **STOP:** No more blind Kaggle submissions or training. No GPU quota available at previous push (`Maximum weekly GPU quota of 30.00 hours reached`). When two-GPU inference capacity becomes available, first run a private no-training sample/long-passages GPU diagnostic with stage markers and OOM handling, not another competition scoring submission. Do not claim 0.622 replicated or change evaluation/scoring rules to suppress an unknown error.

## IMPORTANT CORRECTION — submission 57983359 FAILED (2026-11-04 MYT)

**SUBMISSION FAILURE VERIFIED FROM USER-SHARED KAGGLE SUBMISSION DETAILS: `Notebook Threw Exception`. DO NOT INTERPRET KAGGLE CLI `SubmissionStatus.COMPLETE` AS PASS.** The authenticated submissions list reported `COMPLETE` with blank Public/Private scores (read-only Actions `41824958338`, including second attempt). The Kaggle UI's detailed evaluation outcome for this exact submission is **Notebook Threw Exception**. Consequently **0.622 WAS NOT REPRODUCED**. Existing own best scored Public remains **0.520**; no valid new score.

- Submitted notebook: `cosmorin/name-that-llm-public620-reproduction` version 1; Kaggle user-facing link supplied: `https://www.kaggle.com/code/cosmorin/name-that-llm-public620-reproduction?scriptVersionId=361356190` ("Nishant V1 exact cache smoke600"). The CLI accepted our direct source-version submission `57983359`, but the subsequent hidden-data rerun threw an exception. The user provided the definitive UI outcome. The historic upstream `0.622` score is not this team's result.
- Kaggle's official [code competition debugging documentation](https://www.kaggle.com/code-competition-debugging) explicitly classifies `Notebook Threw Exception` as an unhandled error on the hidden-dataset rerun; Kaggle intentionally does not provide detailed private-run traceback to avoid hidden-data leakage. This failure *does not prove* GPU quota, a particular dependency failure, an OOM, or code incompatibility. GPU quota blocked a **separate notebook-push** attempt; it was not the observed error class for this already accepted direct submission.
- Static source check: scored V1 Python code SHA-256 `1595f543caf314fc766a79dca313961af4d73f2a5cbea1ec2b4724aea5c4289b`, AST syntax valid. Source hard-asserts hashes of `train.csv`, `pool_v4.parquet`, `svc_prefit.joblib`, 18-label metadata, and **exact** `numpy/scipy/scikit-learn/joblib` versions from the cached manifest. Any difference in the hidden-rerun environment or assets throws an exception; this is a plausible failure family, **not a verified root cause**.
- Hidden-size execution path calls `TfidfVectorizer.fit_transform` on all test passages, creates an all-pairs dense distance matrix, runs SciPy hierarchical clustering, sequentially loads two 14B LoRA trunks and DeBERTa, and (at >=50 test rows) runs 300 iterations of Sinkhorn not exercised on the three-row public sample. All this code is in the scored V1 source; no complete hidden rerun trace is available.
- Local **synthetic no-training CPU check** exercised just the same SciPy pairwise-clustering and 300-step Sinkhorn math on 4,000 fabricated passages: completed with finite probabilities, ~128 MB pairwise distance matrix, ~1.47 s in this local CPU environment. It did **not** exercise Kaggle's GPU, pretrained models, hidden passages, asset mounts or runtime image; this success does **not** validate submission readiness.
- Next valid diagnostic is an independent no-training asset/environment compatibility and longer synthetic-passages stress test in an approved runner (when quota permits), not repeated blind submissions. Consider strict version/hash assert drift, ambiguous mount paths, 14B GPU placement/OOM, and hidden-only large-test branches as distinct hypotheses. Do not silently remove scientific integrity checks to force a run. **No retry or new leaderboard submission until a material cause is addressed and independently verified.**
- The previous "Kaggle score check" entry below is preserved as historical evidence of the CLI's limited response; its interpretation as successful scoring is explicitly superseded by this correction.

## Kaggle score check (2026-11-04 19:56 MYT)

- Submission 57983359: `SubmissionStatus.COMPLETE`; Public Score and Private Score are both blank in the authenticated Kaggle API.
- Evidence: GitHub Actions run 41824958338, read-only successful status check.
- Prior best Public Score remains 0.520. No new submit or training in this check.

## Existing public V1 direct Kaggle submission — 2026-11-04 MYT

**ACCEPTED BY KAGGLE API; SCORING PENDING; NO USER TRAINING OR NEW NOTEBOOK PUSH.** After the user clarified that they wanted a *competition submission*, not self-training, submitted the public Cosmorin already-saved V1 directly via `kaggle competitions submit name-that-llm -k cosmorin/name-that-llm-public620-reproduction -v 1 -f submission.parquet`. The official code competition's server will re-run the submitted Notebook against hidden data for scoring.

- **New submission ref: `57983359`**, created `2026-11-04 11:45:46 UTC`; GitHub Actions [41823529904](https://github.com/kolihhan/name-kaggle/actions/runs/41823529904) **SUCCESS**; Kaggle CLI printed `Submission ref: 57983359`.
- Verified the public scored V1 source SHA `e9819e87b1b60231388aff0db1d0ef312b707c6c6b9911a054603333f6bb067b`, code attribution in submission description `Direct public Cosmorin V1 prefit reproduction with attribution`, and no duplicate among 7 prior submissions.
- The next immediate submissions list showed `57983359 / SubmissionStatus.PENDING / publicScore blank`; **0.622 is the upstream author's reported Public score, NOT our scored result**.
- No repo-side Kaggle GPU kernel push, model training, or local inference occurred. Kaggle will run the submitted public inference Notebook on hidden test during scoring; this is execution/inference, **not additional model training by us**.
- **Do not re-submit duplicates.** Next action is read-only check of submission `57983359` for `COMPLETE / ERROR` and only then compare its actual Public score to existing own `0.520`. If it fails, examine the official failure class before any retry.
- Exact one-shot GitHub Actions workflow: `.github/workflows/submit-existing-public-v1-direct.yml`.

## Kaggle 0.622 retry outcome — 2026-11-04 MYT

**CONFIRMED GPU QUOTA BLOCK; NO NEW KAGGLE SUBMISSION.** The user explicitly requested a fresh retry. GitHub Actions run [41823017835](https://github.com/kolihhan/name-kaggle/actions/runs/41823017835) failed at `kaggle kernels push` with Kaggle's explicit response: `Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.`. The account's maximum is **30.00 hours per week**; remaining quota at this attempt was insufficient for the GPU notebook.

- This retry corrected a genuine `kernel-metadata.json` title/slug mismatch, put original upstream attribution in a Markdown cell rather than executable Python, and set the documented `machine_shape: NvidiaTeslaT4` for T4 x2. Static preflight verified that executable code hash still matches the published scored V1 source, both dataset/model source lists and the output contract remain unchanged, and no new model training is introduced. Relevant commits: `44e738cdffd1fe206502a54943c70cc36a2de6a9`, `dbe1a5d54b38df19cae51c1a855e3cc60c015819`, `98eb2fc4d8f575ae090879df80e073ce44ad7ce5`.
- Earlier run [41822974665](https://github.com/kolihhan/name-kaggle/actions/runs/41822974665) failed only on a transient static-test newline-join bug; final run `41823017835` passed the corrected source SHA/preflight gate. The actual Kaggle push was rejected due to the **confirmed quota**.
- The existing Kaggle submissions remained **7**, including five redundant previously completed v1 rows (Public `0.520`); **there is no 0.622 submission or new Public score** from this retry.
- **DO NOT RETRY before GPU quota renews.** Do not mislabel this as a model/Notebook inference failure, silently change hardware, or attempt another training approach. Use the frozen Kaggle prefit inference-only bundle after eligibility returns; verify quota and existing submission duplicates first.

## User-requested Kaggle prefit submission attempt — 2026-11-04 MYT

**NO NEW SUBMISSION CREATED. KAGGLE KERNEL PUSH BLOCKED.** User authorized submitting a Kaggle Notebook to run on Kaggle rather than self-training. Chosen path was frozen published Cosmorin 0.622 V1 inference-only code, **not** a new training job.

- Added `scripts/build_public_0622_inference_kernel.py`, verifying upstream V1 raw notebook SHA `e9819e87b1b60231388aff0db1d0ef312b707c6c6b9911a054603333f6bb067b` and executable code SHA `1595f543caf314fc766a79dca313961af4d73f2a5cbea1ec2b4724aea5c4289b`, preserving author attribution. It does not include new neural model training.
- Added `.github/workflows/submit-public-0622-inference.yml`, a once-only Kaggle submission guard: existing submissions, exact source, public sample execution, parquet validation, and only then final Kaggle submission. It fails closed if Kaggle refuses a GPU notebook.
- First GitHub Actions run [41821527795](https://github.com/kolihhan/name-kaggle/actions/runs/41821527795) and second diagnostic run [41821744868](https://github.com/kolihhan/name-kaggle/actions/runs/41821744868) both failed at **`kaggle kernels push`**, with Kaggle CLI reporting `400 Client Error: Bad Request`. The response did not expose a confirmed detailed cause in sanitized job logs. Earlier Kaggle GPU quota probe on this date had found `Maximum weekly GPU quota of 30.00 hours reached`, so GPU quota remains a plausible but **unconfirmed** explanation for this particular HTTP 400.
- Verified **7 existing official submissions** (Phase7, Phase15, 5 duplicate v1 submissions), latest v1 Public **0.520**; no new submission, score, model training, inference or Kaggle kernel version from these attempts.
- **STOP:** Do not auto-retry 400 pushes, change Kaggle metadata or resource settings blindly, bypass the GPU restriction, or claim 0.622 for our account. Preserve pinned builder/workflow for a future targeted diagnosis or quota reset. Any later attempt must verify the specific 400 cause and current exact dataset versions before submitting. Keep user preference: avoid self-training.

## 0.622 version-pinned preflight — 2026-11-04 MYT

**STATIC PASS, END-TO-END RUNTIME UNKNOWN — NO TRAINING OR SUBMISSION.** Read the scored Kaggle Notebook V1 and current V3 directly using the public Kaggle API: executable code SHA-256 is identical (`1595f543caf314fc766a79dca313961af4d73f2a5cbea1ec2b4724aea5c4289b`), but Kaggle Docker image digests differ. The existing SVC cache's public manifest matches its code-pinned cache hash and source version; both LoRA adapter configs, three model metas and Kaggle dataset license labels were checked using **small JSON files only**. Dataset metadata says `CC-BY-SA-4.0` for both hosted datasets; HF Qwen model cards say `apache-2.0`. Current dataset version numbers and full weight-byte checksums remain **UNKNOWN**.

The full ~4,000-row hidden test triggers pairwise clustering and 300-round Sinkhorn absent from the three-row public sample, plus sequential two-14B+DeBERTa inference. The Notebook requires pinned NumPy/SciPy/scikit-learn/joblib versions, GPU T4×2 and <=9h; these full-run requirements are not benchmarked. **Do not promote 0.622 to a new Kaggle submission without the independent environment, integrity and runtime gates.** Evidence: [`artifacts/research/public-submission-reuse-audit-2026-11-04.md`](artifacts/research/public-submission-reuse-audit-2026-11-04.md), GitHub Actions `41818385187` (source V1/V3), `41818435010` and `41818716399` (public metadata/small manifests). No model weights, official data or private credentials were downloaded into the public GitHub artifacts.

## Public-submission source audit — 2026-11-04 MYT

**DONE: READ-ONLY AUDIT; NO TRAINING, INFERENCE OR SUBMISSION.** Report: [`artifacts/research/public-submission-reuse-audit-2026-11-04.md`](artifacts/research/public-submission-reuse-audit-2026-11-04.md). Verified public-notebook source pull: Actions `41810333036`; verified authenticated dataset metadata enumeration: Actions `41809970435`.

**Prioritize reproducibility of Cosmorin's historic 0.622 prefit SVC + 2x14B LoRA + DeBERTa inference pipeline** (2 dataset sources, 2 base models) using exact published versions and rights checks. No independent score gain or hidden runtime is established. The newer Cosmorin 0.636 code-only guide references 32 fixed inputs, uses CPU SVC fitting, and reports 2002s for 3-row sample, not hidden execution. The current Lightning 0.666 source uses a hardcoded 3-row preview (21-second UI is not real ensemble runtime); its scored version V47 precedes newer source edits. **DO NOT retrain, resume v3, tune on Public LB, or submit without renewed explicit authorization.** Public dataset filenames readable via authenticated API do not verify weights' bytes, licensing, or final runtime.

## User-directed strategy change — 2026-11-04 MYT

**STATUS: SELF-TRAINING PAUSED; READ-ONLY SUBMISSION REVIEW.** The user explicitly requested that the project stop pursuing self-trained v3/full 8B/27B experiments for now and look at actual Kaggle submission results instead.

- Do not launch/relaunch Colab, Kaggle GPU, self-hosted, or CPU training jobs; do not add another model submission merely to compare Public scores.
- Existing Colab v3 resume-equivalence run [41794900419](https://github.com/kolihhan/name-kaggle/actions/runs/41794900419) ended `cancelled` at 2026-11-04T08:44:34Z; **no CUDA-resume PASS or v3 source-transfer result** was obtained.
- Existing verified submissions: Phase 7 Public `0.369` (#57838845); Phase 15 Public `0.357` (#57861337); two-stage LoRA v1 Public `0.520` (#57908683). Private LB unavailable. The v1 baseline remains unchanged.
- Kaggle's live competition overview states that Public LB uses about 40% of the test, with about 60% reserved for final Private LB; Public scores are not a model-selection oracle.
- Public Kaggle notebook pages currently display stronger third-party methods (for example, `0.622`, `0.642`, `0.666`), but the availability, licensing, full checkpoint/input accessibility, clean reproduction, and independence of those methods are **not yet verified**. Their displayed notebook interactive runtime is not necessarily the hidden re-run runtime.
- **Next task:** read-only audit of published high-scoring Kaggle notebooks and existing competition submission records. Prioritize reproducible, permissible inference-only reuse of public pretrained checkpoints plus fixed, non-tuned ensemble design; evaluate runtime/data/source integrity before any future submission decision.
- Existing frozen v3 contract and past negative evidence are retained for reproducibility, not treated as an obligation to keep training. Reopening GPU training requires a fresh explicit user request.

## Competition anchor

- Submitted champion: `two-stage-lora24-v1` — Public LB `0.520`, submission `57908683`.
- v1 Gate-B primary plain OOF BA: `0.5467611235396471`.
- Previous Phase-7 Public LB: `0.369`.
- Do not rerun/retrain or resubmit identical v1.
- Public LB is external evidence only, not a tuning loop.

## Research doctrine

`RESEARCH_DOCTRINE.md` is binding on this branch.

Every major restriction must record `WHY / CURRENT EVIDENCE / PIVOT TRIGGER / STATUS`. Do not let a temporary restriction become a permanent method-class ban. Frozen contracts remain binding for already-running candidates.

## Rejected v2

`prompt-invariant-contrastive-v2` is permanently rejected under its frozen gate.

- v1 BA: `0.41027732627916647`
- v2 BA: `0.4129626992051072`
- delta: `+0.002685372925940732`
- 5,000 prompt bootstrap 95% CI: `[-0.0006323911599452988, +0.0069285212171021765]`
- decision: FAIL because the bootstrap lower bound was not positive.

Do not rescue v2 by tuning contrastive weight/temperature, changing its source holdout, or running official OOF.

## Active scientific candidate: DCT temporal fingerprint v3

Branch: `exp/dct-temporal-fingerprint-v3`.

Frozen scientific change only:

- control: exact v1 Stage-1 A1 masked-mean pooling;
- treatment: fixed two-mode length-normalized DCT-II `[DC0(mean), AC1]` at the same pooled layers;
- DCT modes fixed `[0,1]`;
- all other Stage-1 data/model/LoRA/optimizer/seed/epoch settings remain v1;
- no contrastive-v2, attention pooling, top-k, blend, router, second backbone, or Public-LB tuning.

Frozen contract: `artifacts/candidates/dct-temporal-fingerprint-v3/CONTRACT.md`.

Frozen transfer split:

- sources: `alperen`, `hailey`, `willrice`
- heldout rows: `14,506`
- training rows after prompt-bank leakage removal: `14,502`
- eligible labels: `17`
- prompt-id overlap: `0`
- prompt-key overlap: `0`
- model scores used for selection: `false`

Verified implementation:

- static CUDA-oriented suite `41569836280`: 19 passed;
- self-hosted executor suite `41580519705`: 24 passed.

## Authoritative CUDA v3 execution — awaiting usable GPU, not scored

- Self-hosted CUDA run `41580819667` was **CANCELLED** after remaining queued without a matching GPU host. It produced no paired v1/v3 training or score.
- Frozen source, model, evaluation contracts and holdouts remain unchanged.
- V3 source-transfer decision remains **UNRUN / UNKNOWN**, not FAIL.
- No official OOF, Public LB retuning, or v3 Kaggle submission has been authorized.

### Available execution routes (verified 2026-11-04)

- **Google Colab Free T4:** User-observed Tesla T4, **14.56 GiB** addressable, torch `2.11.0+cu130`. The frozen executor's `>=15 GiB` preflight rejects it; that is a conservative infrastructure gate, not a demonstrated OOM.
- `notebooks/dct_v3_colab_t4_probe.ipynb`: pinned synthetic T4 LoRA + DCT forward/backward feasibility probe; **EXECUTED SUCCESSFULLY** via GitHub Actions run `41772827966` (attempt 2), 14.563 GiB device memory, 11.68 GiB peak reserved, 4.08-second synthetic update. No official or heldout rows scored; NOT v3 transfer evidence.
- `notebooks/dct_v3_colab_gpu.ipynb`: pinned full paired-transfer launcher for sufficiently large CUDA GPU. **Full frozen v1-v3 source-transfer run still UNRUN**; the separate two-update real-data training pilot `41778479162` is not that experiment.
- **Kaggle T4**: private, no-data quota-only probe `41767348246` re-ran from GitHub Actions. Push failed at `2026-11-04T03:28:46Z`: `Maximum weekly GPU quota of 30.00 hours reached`. GPU kernel did not launch. The updated workflow now fails closed instead of polling a missing kernel.
- **Hugging Face Jobs**: connected Free account's L4 attempt returned `402 Payment Required`; not a usable no-cost GPU route.
- The agent-local runtime has no CUDA GPU, but the connected GitHub Actions workflow **can operate Colab T4 through authenticated official CLI**: synthetic run `41772827966` and real-data pilot `41778479162` both verified; quota and full-run lifetime remain unguaranteed.

GPU *single-batch* LoRA feasibility is **verified**; full-epoch viability and frozen paired v1-v3 source-transfer remain **UNKNOWN/UNRUN**. Do not claim a v3 scientific transfer score without explicit paired-training logs and heldout evaluation evidence.

## CPU shadow — stopped, separate non-scientific lineage

- The previously checkpointed CPU BF16 DCT lineage successfully preserved state through **microbatch #4** (run `41636327118`); no optimizer step had been reached (accumulation=6).
- BF16 microbatch #5 (run `41644695149`) timed out after **330 minutes**, shell exit `124`. Previous checkpoint artifact was preserved, but #5 did **not** produce verified training progress.
- The CPU continuation trigger was removed; latest CPU workflow `41673536800` was `success` only because the training chunk was **skipped**.
- No sustained CPU training. Do not convert CPU shadow partial gradients or loss to scientific transfer PASS/FAIL.
- Previous FP16/BF16 diagnostic summaries elsewhere in this branch are historical and do not authorize further CPU chunks.

## Parallel gradient restriction

Do not split one six-triplet accumulation group across independent runners yet. LoRA dropout + head dropout make naive parallel reduction trajectory-changing unless RNG equivalence is explicitly proven.

Pivot trigger: a dedicated equivalence test under a frozen per-microbatch RNG scheme.

## Validation-budget firewall after v3

Independent branch: `research/v1-mechanism-audit-2026-11-03`.

Frozen firewall run `41590073872`, artifact `13209726705`, digest `sha256:7de14265ac8490511b134a11832ab427d1e15a75077376b3d4e305a58ec2f190`.

Firewall sources:

`abhinav, dophuchung, kaplinskiy, keshav, prvsiyan, prvsiyan_v1, seyit, zbigniew`

Accounting:

- rows: 15,206;
- eligible rows: 14,013;
- eligible labels: 17/18;
- missing eligible label: `claude-fable-5`;
- leakage-clean development rows: 10,359;
- prompt-id/key overlap: 0.

Do not reselect firewall sources to recover the eighteenth label. The firewall is not retrofitted into v3.

## Fallback research — not active v4

If v3 fails, current strongest fallback hypothesis is centered-to-plain privileged-information distillation.

Mechanism audit:

- centered rescues 353 plain errors;
- centered breaks 350 plain-correct rows;
- centered is complementary, not universally stronger;
- same-prompt centering is privileged context unavailable at test inference.

Research primitive TDD:

- RED `41591144258`;
- GREEN `41591267999`.

This is not v4 yet and has consumed no firewall prediction, official OOF, or new Public-LB observation.

## v3 PASS path

Only authoritative exact CUDA transfer PASS may authorize official OOF.

Then:

1. preserve the exact passing v3 Stage-1 checkpoint lineage;
2. adapt the v1 Gate-B worker only for frozen DC0+AC1 representation;
3. keep v1 folds, Stage-2 epochs/LRs/optimizer/seed/class weighting/triplet isolation unchanged;
4. run five-fold official OOF once;
5. require plain BA > `0.5467611235396471`, at least 3/5 fold wins, positive 5,000-group-bootstrap lower bound, and class-recall safety veto;
6. only then final fit + at most one submission.

CPU shadow results cannot authorize this gate.

## Current execution policy

- Primary blocker: availability of an actual GPU executor with enough tested memory and runtime to preserve the **exact** frozen 24-layer LoRA v1-v3 contract.
- Do not retrigger CPU continuation; previous #5 hit a 330-minute timeout.
- Kaggle GPU is quota-blocked as of latest `41767348246`; do not launch full kernel while this remains true.
- Synthetic T4 memory probe is only a hardware diagnostic; it is **not** v3 transfer evidence.
- Official Google Colab CLI `0.7.5` bridge was added at `.github/workflows/v3-colab-t4-probe.yml` under explicit user authorization, using a single bounded synthetic T4 probe, sparse checkout, no official rows, and forced Colab cleanup.
- First validation run `41772721301` failed at workflow parsing (job-level runner context); fixed in commit `20435de64b63e56f8d5957574c754d78b5a57ccf`.
- Initial `41772827966` attempt was blocked by missing user OAuth; the user configured `COLAB_ADC_JSON` and **the same run, attempt 2, SUCCESSFULLY allocated Colab T4**, finished one synthetic forward/backward/optimizer, peaked at 11.68 GiB reserved, and returned `synthetic_single_triplet_step_seconds=4.08`. No scoring or submission.
- New separate training pilot: `scripts/dct_v3_colab_t4_train_pilot.py`, invoked via `.github/workflows/v3-colab-t4-real-train.yml`. It loads the immutable external Phase14 safe subset artifact `13067907516`, verifies the frozen 3-source holdout split, trains **only on 31 real external-source training rows** in 12 prompt-local triplet microbatches, and performs **2 genuine AdamW optimizer updates** on FP16 Qwen3-8B (24 layers) + DCT [0,1] + LoRA r16/norms/heads.
- First full-data pilot attempt `41777780491`: remote training COMPLETED 2 optimizer steps but Colab CLI default execution timeout prevented checkpoint recovery. This is a tooling timeout, not training failure.
- **Verified successful pilot `41778479162`**: completed 2/2 optimizer updates, `head_parameter_delta_l2=0.0255487971`, `loss_mean=5.7616079` (train-only; not predictive quality), peak reserved 11.043 GiB, no official data, no heldout predictions, no Public LB. CLI was given `--timeout 1200`, and both `real_train_result.json` and `real_train_checkpoint.zip` were downloaded, validated, and retained as Actions artifact `13297683118` (~114MB compressed). Remote Colab T4 session cleanly stopped.
- This pilot checkpoint contains adapter + norm/head weights only; **it is NOT an exact-resumable full epoch/AdamW state**, NOT the frozen paired v1-v3 transfer, and NOT authority for official OOF or submission. Do not overinterpret train-loss changes or repeatedly rerun the pilot.
- The Colab CLI/Google user OAuth secret now works. Protect `COLAB_ADC_JSON`, never expose it in logs/history; the consent-screen Testing setting may affect credential lifetime.
- Never weaken the frozen source lock, architecture, DCT modes, training steps, split, seed, or evaluation gates to fit a smaller GPU and then call the result v3.
- Do not consume official OOF or Public LB before exact paired source-transfer PASS.

## PIC checkpoint/research execution review — 2026-11-04 MYT

### Evidence and decision
- **VERIFIED:** Real external-data T4 pilot `41778479162` completed 2 true optimizer updates and preserved trainable-only adapter/heads. It tested training mechanics, **not** source-transfer or predictive quality. Full v1-v3 paired source-transfer remains `UNRUN / UNKNOWN`.
- **VERIFIED:** Added `scripts/dct_v3_cuda_resume.py` (atomic optimizer-boundary save/load for trainable parameters, optimizer, scheduler, GradScaler, Python/NumPy/torch CPU and all CUDA RNG states, strict checkpoint fingerprint/cursor validation, fail-closed mid-accumulation); `tests/test_dct_v3_cuda_resume.py` proves bitwise CPU interruption/resume equivalence with dropout and AdamW and checks wrong lineage/unsafe cursor rejection.
- **VERIFIED:** Run `41792568893` — `V3 Static Tests` — completed **27 passed**, including resume primitives, and compiled frozen v3 worker/kernel. This test suite does **not** verify CUDA RNG restoration or a real resumed 24-layer Qwen run.
- **UNKNOWN:** GPU resume equivalence across separate Colab sessions, Colab free-T4 quota/lifetime for full runs, genuine source-transfer advantage for DCT, or resulting Kaggle performance.
- **REJECTED:** Treating 31 train rows, 2 updates, training loss or a successful CI as evidence that v3 beats v1. The saved pilot adapter is **not** an exact-resumable v3 Stage-1 checkpoint.

### Current PIC priority (smallest discriminating next work)
1. Wire the already-tested optimizer-boundary checkpoint primitive into the **existing frozen v1 and materialized v3 workers**, preserving all training semantics, prompt triplet order and every frozen configuration field, not building a parallel experimental training recipe.
2. Freeze a verifiable resume fingerprint binding the model revision + truncated shard manifest, source-code revision, raw public pool SHA, Phase14 subset/split-lock digest, label order, v1/v3 condition, DCT modes, triplet order, training spec, PyTorch/transformers/PEFT runtime and exact optimizer-step count. Reject foreign/incompatible checkpoints.
3. Test uninterrupted vs interrupted-and-resumed training on the **same** small fixed real-data microbatches on Colab T4; compare trainable parameters, optimizer/scheduler/scaler cursor, per-microbatch loss and RNG state without looking at heldout predictions. CPU resume 27-test CI does not substitute for this.
4. After CUDA equivalence PASS, execute both full **one-epoch** Stage-1 conditions using the exact frozen 14,502 training rows, recover checkpoints after bounded runs, and score the predeclared three-source 14,506-row holdout **once** only when both conditions finish.
5. Apply frozen paired-transfer gate as-is: pooled BA increase, positive 5,000-prompt bootstrap lower CI, positive mean source delta, at most one negative source and no per-label recall safety veto. **FAIL -> reject v3 without retuning**; **PASS -> official five-fold OOF**, then only if that gate passes proceed to final fit/submission per contract.

### Resource and research guardrails
- Do not relaunch the 2-step pilot or the 330-minute CPU shadow. Reuse the successful T4 feasibility and existing CPU-resume implementation as infrastructure evidence.
- Colab free T4 is **not** an unlimited GPU; stop cleanly and preserve optimizer-boundary checkpoints before session limits. Do not open several competing GPU sessions. No additional Kaggle GPU run while weekly quota status is unverified.
- No Public-LB tuning, no adaptive holdout source reselection, no v3 mode/layer/loss sweeps. Source-transfer is a frozen external-data diagnostic, not a guarantee of hidden-Kaggle generalization.
- Research question: whether an added temporal AC1 component captures generator identity that survives source shift, versus noise/corpus-format artifacts. This mechanism is **hypothesis**, not yet evidence. If transfer fails, write per-source/label error analysis as allowed by the frozen result report, and only then reconsider the model family or centered-to-plain distillation under a *new* predeclared contract.

- **Pre-result mechanism confound (2026-11-04):** DCT `[DC0, AC1]` also doubles the Stage-1 classifier input width and consequently increases head parameter count versus v1 mean-only pooling. A future paired-transfer PASS would establish improvement by the *frozen composite treatment*, not isolate AC1's generator-fingerprint contribution from added head capacity. Keep the v3 contract unchanged. Any capacity-matched placebo/ablation must be separately preregistered and judged with independent evidence, not used to rescue/reselect v3 against its already-locked holdout.
- **Version pin audit:** The live Colab optimizer-resume proof pins `d9bf4f0f88d396d32348e94033c07759c244ec6d`; Git blob SHAs of its shared Stage-1 worker, CUDA resume helper, v3 materializer, and frozen split builder match those on this branch as of `10a6919a18641a885d4876ab07fbbc835dfa108d`. Later commits changed proof/CI/tests/docs, not those four worker inputs. A PASS would still be *only* resume-equivalence evidence, never source-transfer evidence.

- Do not declare CUDA resume or paired transfer complete just because a state-serialization unit test passed.

## PIC implementation ledger — 2026-11-04 (resume worker integration)

- **IMPLEMENTED:** Opt-in `optimizer_boundary_path` / `checkpoint_every_steps` / `stop_after_optimizer_steps` parameters on the **shared frozen v1 A1 CUDA worker**, and the exact generated v3 worker inherits them. Defaults remain disabled; no science configuration, source selection, head pooling, DCT coefficient or evaluation threshold was changed.
- **IMPLEMENTED:** Checkpoints carry strict runtime/worker source/safe-parquet/split index/triplet order/label/model revision/manifest/training-spec fingerprint, trainable weights, AdamW, scheduler, GradScaler, CPU/CUDA RNG, optimizer cursor and training counters. Clean optimizer boundary is enforced. Pausing cannot read heldout predictions.
- **VERIFIED CPU/STATIC:** GitHub Actions `41794531008` 30 passed and follow-up `41795292904` **31 passed** including new worker default/resume contract, interrupted-vs-uninterrupted CPU dropout/AdamW equality and training counter restoration; frozen worker materialization/kernel compilation passed.
- **RUNNING/NOT YET VERIFIED:** GitHub Actions `41794900419`, a single Colab T4 invocation comparing fixed real external-data frozen-v3-worker uninterrupted two optimizer steps against interrupted one step + resume one step. No official competition data or source-holdout prediction is used; result must be checked before labeling CUDA resume PASS.
- **NOT DONE:** CUDA bitwise equivalence, full v1/v3 one-epoch training, frozen paired source-transfer and official OOF/submission. Do not infer model quality from optimizer state or training-loss outputs.

## Next action

Real-data T4 LoRA training feasibility and checkpoint artifact are now **VERIFIED** by run `41778479162`. Stop further bounded pilot reruns: there is no value in repeating two optimizer steps. The open scientific question is still the original *frozen paired v1 vs v3 full source-transfer* experiment. Before any costly full attempt, build an exact-resumable CUDA training/checkpoint mechanism that preserves optimizer/scheduler/RNG/grad-accumulation state and survives Colab session limits; keep evaluation firewall and split unchanged. GPU availability, remaining free quota, full-run duration, and the effect of DCT on heldout BA remain **UNKNOWN**.

Visibility policy (user reaffirmed 2026-11-04): Keep using the PUBLIC repository and GitHub Actions / Colab workflows; do not impose a mandatory switch back to PRIVATE after routine training. This does **not** prove Git history was sanitized: official competition data exposure in reachable history was previously recorded, and cleanup is unverified. Do not introduce new official competition files or secrets; do not rewrite history without separate authorization. See `PUBLIC_VISIBILITY.md`.
