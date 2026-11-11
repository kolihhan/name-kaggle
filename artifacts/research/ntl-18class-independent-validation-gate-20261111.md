# Name That LLM — Independent 18-Class Validation Readiness Gate

**Decision date:** 2026-11-11 MYT. **Result: BLOCKED — no currently established clean, unused, complete 18-class independent benchmark.**

This is a **read-only review of existing repository provenance and past executions**, not a new dataset search with new unseen rows, not an inferred result on uninspected files, and not a new experiment. It is intended to prevent **adaptive test-set selection, source contamination and score-shopping**.

## What must be evaluated

Competing systems must predict the **same 18 exact generator labels** using only the text permitted at competition inference. Balanced accuracy is the unweighted mean of per-true-label recalls, not the overall accuracy. An evaluation without truth support for some labels is explicitly a **conditional subset diagnostic**, not 18-way balanced accuracy.

**A source is not clean just because it is recent or is absent from the champion's direct Kaggle dataset mounts.** We must separately check whether texts, prompts, upstream sources or held-out model outputs influenced any prefit model, readout, calibration, rule or model selection in the incumbent and new candidate.

## Previously inspected candidate evidence

| Dataset/evidence | Source size/label support actually established | Prior use/identity issues | Admissible purpose | Untouched full-18 transfer? |
| --- | --- | --- | --- | --- |
| Official competition train / historical grouped OOF | 2,699 labeled official rows, 18 true classes | Reused across many adaptive experiment phases and shadow splits; training/OOF data | Historical grouped OOF, error diagnostics | **NO** |
| `nishantdahal/ntl-handwriting-blend` `pool_v4.parquet` | 75,783 public rows, 18 labels in source; own Stage1 A1 used a selected 30,133 rows | Own pretraining/heldout-source Gate A conducted there; no independence from incumbent prefit data proven | Frozen past model evidence, provenance analysis | **NO** |
| `crtproduction/name-that-llm-open-pool` | 212,623 rows, 18 labels, 60 nominal upstream source IDs | Source-owner evidence contains official-host-conditioned generation and prompt reconstruction in candidates for CDA-L24; v1 study declared INVALID; nonuniform licensing/protocol risk. **Not all individual rows separately assessed** | Source filtering/provenance investigation only | **NOT VERIFIED** |
| `toanplam/ntl-fresh-generation-pilot-nov08` | 107 rows, **12/18**, nine authored prompts | Already scored 0.679 champion on the 347-row diagnostic; served weights/endpoint revision not verified | Spent conditional 12-label diagnostic | **NO** |
| `nimithashreea/ntl-proxy-candidate-generations-20261110` | 240 rows, **12/18**, 20 Seyit-derived prompts | Already scored and reviewed; reused known prompt bank; no proven prefit disjointness | Spent conditional 12-label diagnostic | **NO** |
| `willrice/name-that-llm-openrouter-harvest` | Source data declared 14/18 labels. 80 controlled Kimi high/off *pairs* (160 answers) from distinct `setting_trials.jsonl` | Kimi-only reasoning condition diagnostic already read; Seyit prompt overlaps; generation protocol and request version confounding. The wider source has not been certified safe | Spent Kimi protocol diagnostic and future metadata/provenance only | **NO** |
| `olegzholobov/name-that-llm-response-corpus` | 13,642 wide records; strict 4,446 source-attributed response cells audited, useful GLM/GPT sibling pairs | Upstream SWEeper-Bench agentic traces; model/harness/effort/target-length not controlled; prior Phase8 use | Existing channel-confounding diagnosis | **NOT VERIFIED** |
| `ScaleAI/MA-Bench` | 1,560 long-form records; Luna/Sol only 10 distinct matched scenarios per condition | Lengths thousands of words, incomplete request/version metadata, very different domain | Negative provenance/coverage control | **NO** |
| `maniceet/name-that-llm-openrouter-openai-generations` | Claimed interesting model coverage; authenticated access **403** in previous audits | File-level license/rows/endpoint not independently inspected | Do not bypass access | **INACCESSIBLE** |
| Official Kaggle Public leaderboard | Own attributed public-source reproduction `58210835` **0.679**, prior `58022340/58045266` **0.668** | Hidden public slice already used to select a champion; no individual labels/error rows available | Competition outcome evidence, not tuning target | **NO** |

Evidence references:
[Phase 19 historical OOF](../phase19/RESULT.md);
[LoRA24 source checkpoint](../candidates/two-stage-lora24-v1/RESULT.md);
[CDA-L24 v1 invalidated data provenance](../candidates/conditional-domain-adversarial-lora24-v1/CONTRACT.md);
[Fresh 12-label survey](ntl-fresh-12-label-cohort-triage-20261111.md);
[347-row result](ntl-0679-transfer-diagnostic/RESULT_20261111.md);
[Kimi 80 result](ntl-kimi-protocol80/RESULT.md);
[STABLEPAIR-18 data gate](stablepair18-ma-bench-data-gate-20261107.md);
[Generator versus channel audit](generator-vs-channel-identification-decision-20261106.md);
[Official 0.679 Kaggle evidence](../candidates/public-ozertuu-0683-guarded-repro/OFFICIAL_SCORE_20261111.md).

**Correction to a false binary:** There *are* 18-label external corpora; what is missing is **audited, independent, unused 18-label transfer evidence with comparable collection protocols**. We do not have grounds to call all 18-class source rows permanently unusable, but none audited here is eligible to be represented as *already qualified*.

## Admission gate for a NEW independent benchmark

Do not execute or score a new model until the benchmark has a pre-registered source/version manifest satisfying:

1. **Source rights:** original dataset owner/license and permitted use documented; normal authorized access; never push official Competition Data into public Git.
2. **Actual label support:** all 18 exact competition labels have labeled observations, counted per model and provider; no pseudo-labeled unknowns in validation. Report absence explicitly.
3. **Unit of independence:** unique original prompt/request IDs and upstream family; repeated generation samples or multiple settings of the same prompt are **correlated**, not new independent prompts.
4. **Prompt/model/source disjointness:** prove no identical normalized responses, near duplicates, shared original prompt templates, or source-derived examples were in relevant candidate training/calibration. Exact hash screening is necessary but **not sufficient** to establish semantic independence.
5. **Provider/version/protocol identity:** requested AND actually served model ID, date/version when available, original prompt/system wrapper, target length, reasoning/temperature/top-p/seed policy. Explicitly mark unobserved fields. A provider slug matching is not proof of identical weight snapshot.
6. **Coverage and balanced protocol:** ideally independent source/batch variation with same requests/length targets across candidate models. If model and generation channel are coupled, mark the benchmark as *observational* and do not claim intrinsic attribution or protocol invariance.
7. **Holdout governance:** set aside a test partition before architecture/model selection and ensure the next model's developer/agent has not inspected its outputs, per-label errors or aggregate score. No post-result slice- or threshold-based tuning. Once viewed for any candidate selection, mark **SPENT**.
8. **Sufficient uncertainty resolution:** predeclare sample size and precision/power based on **independent prompt clusters** and a plausible paired-error/disagreement rate. Small datasets may support a diagnostic but cannot reliably establish a +0.021 absolute BA improvement over 0.679. Do not invent a fixed 50/100-prompt threshold as universal proof of precision.
9. **Same-row control:** evaluate a **hash-pinned 0.679 champion** and a fixed candidate on precisely the same admissible rows with all 18 outputs. Report 18-label macro recall, per-label recalls, rescue/regression with prompt/source clustering, computational cost, and negative controls.
10. **Promotion:** a new inference rule/architecture and parameters are frozen *before* any unseen benchmark result, with an explicit fail case. One authorized competition submission only **after** the existing candidate and transfer gates pass; Public BA cannot be fed back into parameter search.

These criteria are **prospective scientific requirements**, not a claim that Kaggle's published scoring rules require our entire research standard. Model fitting remains prohibited by the user's current `NO NEW TRAINING` boundary.

## CURRENT DECISION (fail closed, not indefinite research theater)

- **18-class independent validation: BLOCKED / not verified.** No source from this existing evidence list may be promoted by name alone. No 0.700+ claim is supported.
- **Source scouting next:** one bounded **metadata/provenance-only** source intake (not an unbounded GPU experiment). Candidate source must supply exact prompt / model / request protocol provenance and data license. Publish a small pass/fail matrix, not new scripts or Actions by default.
- If no credible source qualifies, **STOP the 0.700+ proof claim** and state the concrete missing evidence; do not manufacture fresh holdout status via resplitting 2,699 official examples, combining spent 12-label corpora, using Public LB or using a proxy-only 80-pair p-value.
- Separately, propose a representation-level change only with an independent mechanism/negative control and a feasible frozen test; do not reactivate closed A1 blend, Kimi heuristic or length-order shortcuts as untuned "new methods".


## Authenticated nine-source recheck — 2026-11-11

**Evidence:** Reused existing read-only [GitHub Actions audit #42993607587](https://github.com/kolihhan/name-kaggle/actions/runs/42993607587), job `130432132503`, **SUCCESS** (Kaggle authenticated datasets/files, public README text and public metadata only). It inspected **9 explicitly selected source refs** against the scored champion's **20 direct Kaggle dataset dependencies**, and did **not** run the model, fit a classifier, access official train/test passages, or make a submission. One new outcome is source *ineligibility* for clean validation, not a new score.

| Candidate | Verified file-card / README findings at 2026-11-11 | Directly attached to 0.679 champion? | Verdict for **independent full-18 validation** |
| --- | --- | --- | --- |
| `prvsiyan/prvsiyans-extended-name-that-llm-dataset` | Source README declares **13,500 texts, 810 prompts, 18 labels**; `new` split **300 prompts × 18 labels = 5,400 rows**, 4,158 passing source quality screening; **300 complete 18-model prompt blocks**, with 231 passing an additional quality funnel. 18-class `new2` and `train_regen` use reconstructed host-style or community-reconstructed train prompts. Per-model target lengths sampled using host competition train word-count distribution; multiple routes and model-specific defaults. | **YES** | **REJECT as fresh**: a current champion direct dependency; source/model/target-length confounding. Counts reported from the publisher's README, not independently recomputed from 13,500 raw passages. |
| `nishantdahal/ntl-handwriting-blend` | `pool_v4.parquet` visible; existing own A1 research documents 75,783 rows / 18 labels, with 30,133 previously fitted Stage1 rows | **YES** | **REJECT**: reused for modeling and direct champion assets |
| `seyitkaangunes/ntl-extra-api-passages` | **4,998** texts, **5 Claude labels**, **1,000 prompts**, clear first-party API provenance; requested lengths drawn per model using competition-train distribution | **YES** | **REJECT**: only Claude labels, champion direct input |
| `lxdlxdlxd/ntl-synth-generations-v1` | **4,758 passages / 13 exact labels / 340 generated prompts**, real first-party or official-provider route with `requested_model,returned_model,settings,target_words,user_message,timestamp`; publisher explicitly says **all five GPT labels absent**. Prompt design was conditioned on community-reconstructed host prompts, and some settings differ among models | **NO** | **PARTIAL, useful provenance lead only**; 13/18 not full 18 and indirect exposure through merged source pools not ruled out |
| `willrice/name-that-llm-openrouter-harvest` | **10,772 cleaned / 14 labels**, served IDs and per-call protocol available; same **Seyit** prompt-bank upstream. Publisher used **competition-trained style classifier** for per-setting fidelity selection; Kimi 80 high/off data already scored by champion | **NO** | **REJECT**: 14/18, source-selection bias, spent Kimi diagnostics |
| `nimithashreea/ntl-proxy-candidate-generations-20261110` | 240 / 12, 20 reused Seyit Prompt IDs, existing exact Kimi-independent diagnostic run | **NO** | **REJECT**: 12/18, spent |
| `toanplam/ntl-fresh-generation-pilot-nov08` | 107 / 12, nine authored prompts, proxy served ID unavailable, prior diagnostic | **NO** | **REJECT**: 12/18, spent |
| `maniceet/name-that-llm-openrouter-openai-generations` | Public Kaggle page claims **7,753 / 10 labels**, rich per-row route provenance, including exact request when retained; current authenticated `datasets files` **rc=1**, same as previous inaccessibility | **NO** | **UNAVAILABLE through authenticated Kaggle CLI**, and not 18-label even if access changes. No bypass attempted |
| `crtproduction/name-that-llm-open-pool` | Earlier version was independently counted in repository as **212,623 rows / 18 labels / 60 source IDs**; public mirrors mention later **254,561**, and live page title suggests yet another updated clean pool count. In this authenticated run, `datasets files -v` **rc=0 but zero parseable file entries**, `datasets metadata` rc=0 but no expected parsed JSON fields; cannot verify latest actual row counts/files/version. The pool merges upstream sources including host/reconstructed-prompt-conditioned datasets | **NO** | **NOT VERIFIED** for current version or independence. Explicitly **do not interpret empty parsed list as proof that dataset is empty**, and **do not promote an aggregator just because it advertises all 18 labels** |

**Interpretation of the new evidence:** the best *well-documented single-provider acquisition* lead found here is `lxdlxdlxd` (13/18, no direct champion link), but it is insufficient on its own and may be indirectly included in the public pool/pretuned sources. The most complete **18-class prompt panel** found here is `prvsiyan`, but it is a direct champion dependency; its extensive provenance does not make it unseen. The updated `crtproduction` source is **unresolved** because the current file API/metadata parse did not establish version and per-source membership; the old host-conditioned warning still applies.

**Next stop decision (bounded):** This survey is concluded: **0/9 candidates admitted** as a currently unused independently qualified full-18 benchmark. Do **not** retry these nine as repeated source-count optimizations. The next *new* admissible source would need **documented genuinely new prompt families, actual served model IDs, per-class 18-label support and source lineage excluding all model-fitting inputs**, and a reserved holdout shielded from candidate development. Building a synthetic "18-class" composite from `lxdlxdlxd` + another GPT-only bank has unverified mixed-protocol/source confounds and **is not** a clean shortcut.

**Feasibility reality:** a pristine full-18 benchmark may remain infeasible because of model availability and provenance. This does **not** prevent all engineering. Limited 12/13/14-label datasets can be described transparently as exploratory transfer diagnostics; the Kaggle hidden rerun remains the only currently authenticated official 18-label evaluation, and if a future candidate is compelling its one-shot official submission should be preauthorized as *exploratory*, not advertised as statistical proof of out-of-distribution generalization. Do not hold all future competition engineering hostage to a possibly unavailable perfect data source, but do not re-label Public-LB tuning as research either.

**Resource footprint:** 1 reused read-only metadata workflow job, **zero new workflow files**, no external GPU, no new training, no competition submissions.
