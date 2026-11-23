# Claude blind review — Stage B: actual evidence and code provenance
OPEN **ONLY AFTER** Round A answer has been written and frozen. Do NOT retroactively rewrite Round A.

Primary repo: https://github.com/kolihhan/name-kaggle
**Use branch** review/ntl-claude-blind-third-round-20261123, not stale default main.
This branch is a handoff snapshot descended from exp/ntl-0715-stage-trace-frozen-20261123. Some contemporaneous experiments live in **other** branches; the links below identify them.

## Review discipline

1. Verify numeric and technical claims against actual code, terminal GitHub Actions logs, publisher source metadata, and original Kaggle receipts where accessible. A successful workflow does **not** prove competitive model quality. A notebook title is **not** a score.
2. You have permission to read public GitHub code/artifact summaries, not to publish or export original competition data or user API secrets. Some historic public Git history was not certified safe; do not amplify it.
3. Distinguish: (a) official scored Public BA, (b) genuine matching OOF, (c) domain-specific external panel, (d) publisher self-attestation, (e) static code / synthetic test, (f) speculation.
4. Review source and data lineage before treating a new source as independent. Same-row reference is compulsory for an improvement claim; unrelated OOF and Public are not comparable estimands.
5. Do not learn the project's **own proposed next-step verdict** from NOW.md or old Claude review files until you write your Stage B diagnosis. NOW.md is an append-heavy log and includes superseded scores and unresolved "next steps".

## Baseline facts: verify rather than accept

- Task: 18 exact generator classes, unweighted per-class Balanced Accuracy. Official TRAIN: 2,699 rows, 150 prompt groups. Hidden test about 4k; preview is only 3 rows; prompt IDs not provided at hidden inference. Official 9h code runtime.
- Latest **account-authenticated Public incumbent** reported: 0.715, submission #58680531, externally attributed copy of Estopadilla v5 (not independent original model research). See authenticated owner 27-entry score ledger [Actions #45137623140](https://github.com/kolihhan/name-kaggle/actions/runs/45137623140), [scored code lineage](../../research/ntl-0715-highscore-decision-lineage-20261123.md), and exact scored Notebook SHA256:
  202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8
- Scored model relies on an external/pretrained multi-stage stack: nine model mounts, 26 dataset sources, 10 kernel sources. Exact source and module evidence: [Actions #45136005059](https://github.com/kolihhan/name-kaggle/actions/runs/45136005059), [source census #45142007503](https://github.com/kolihhan/name-kaggle/actions/runs/45142007503). "9 models attached" does not prove nine statistically independent effective classifiers.
- 25/25 selected public notebooks in a Kaggle requested scoreDescending catalog sample share GLM likelihood, stylometry, residual and soft grouping code; 24/25 have same asset manifest as own 0.715, 17/25 share nine core module bodies. That selected public sample is genealogically dependent, not 25 independent replications; numeric scores were not provided by the CLI catalog.
- The top 0.719 in some notebook **titles** is not verified as an owner-scored 0.719 result. Do not import title scores as ground truth.

## Historic experiments; do not merge metrics across cohorts

| Evidence | Actual metric | Source & methodological limit |
| --- | --- | --- |
| Phase 7 vs Phase 15 | Phase 7 OOF BA .370052; Public .369. Phase 15 OOF BA .405322; Public .357. | [Phase19 shadow result](../../../artifacts/phase19/RESULT.md). Three additional prompt-group assignments retain OOF uplift yet Public remains worse. Old TRAIN has now been repeatedly mined. |
| Own learned LoRA24 experiment | Gate A external-only .394243 vs .465713 across selected 13-label external set; later **OOF .54676 vs own Public .520** for a distinct scored recipe. | [LoRA24 Gate A](../../../artifacts/candidates/two-stage-lora24-v1/RESULT.md), score history in README; multiple lineage and distribution confounds. Do not compare .54676 directly with .715 Public. |
| Actual frozen full-18 external test on older 0.691 | 89/216 = .412037 BA, 12 examples per exact class; family correct 153/216, within-family sibling errors 64, cross-family 63. | [Real test and domain caveats](../../../artifacts/research/2026-11-16-pro-external18-actual0412-domain-bottleneck.md), [paired oracle audit](../../research/ntl-real-family-vs-sibling-216-20261123.md), [Actions #43804320931](https://github.com/kolihhan/name-kaggle/actions/runs/43804320931). **ALL 216 from sensitive_topics**, reused; 0.691 is not 0.715; no causal overfit diagnosis. |
| Legitimate precomputed GLM-5.3 vs Flash likelihood | Mean genuine loglikelihood binary BA .893333 on historical 300 sibling-labelled rows; 53-feature strong logistic .943333. | [GLM original evidence](../../../artifacts/research/ntl-glm-frozen-likelihood-20261114/RESULT.md). Do not compare binary 300 performance directly to Public 18-way BA. |
| ATLAS 2026 clipped/token readout A0 | 274/300 = .913333 vs global53 283/300 = .943333. | [Real A0](../../research/ntl-atlas-a0-real-result-20261123.md), [Actions #45152458986](https://github.com/kolihhan/name-kaggle/actions/runs/45152458986). This implements only token readout, **not** source-conditioned trained ATLAS. |
| ATLAS fixed A1 marginal feature | global53 .943333 (283/300), global53 + one ATLAS margin .946667 (284/300); +2 rescues, −1 regression; grouped-bootstrap 95% delta interval [−.006667,+.013417]. | [A1 actual report](../../research/ntl-atlas-a1-incremental-result-20261123.md), [Actions #45174502026](https://github.com/kolihhan/name-kaggle/actions/runs/45174502026). Same spent GLM 150 prompt groups; signal highly correlated with existing paired means. |
| Different 13-feature nested token fusion | .923333, under global53 .943333, same historical binary cohort. | [Separate branch result](https://github.com/kolihhan/name-kaggle/blob/exp/ntl-global-token-fusion-nestedcv-20261123/docs/research/ntl-global53-token13-fusion-result-20261123.md), [Actions #45169232586](https://github.com/kolihhan/name-kaggle/actions/runs/45169232586). |
| 18-way original token-order Markov proxy | .189292 conditional bigram versus .244474 class unigram and .367089 strong sparse same-row OOF; context-shuffled variant .153753. | [Generator-transition study](../../research/ntl-generative-transition-v1-result-20261123.md), [Actions #45106442160](https://github.com/kolihhan/name-kaggle/actions/runs/45106442160). Tests this weak observed-token proxy, not full learned source-conditioned likelihood. |
| Frozen DistilGPT2 context-head proxy | .182655 vs .202262 unigram vs .367089 same-row sparse reference. | [GCTH study](../../research/ntl-gcth-v1-result-20261123.md), [Actions #45123925475](https://github.com/kolihhan/name-kaggle/actions/runs/45123925475). Limited contexts/projection, cannot reject all pretrained causal representations. |
| Qwen3-1.7B short contrastive trial | CE .078889 vs CE+InfoNCE .071667 on 1,800 source-disjoint external rows, 62 rescues/75 regressions. | [Historical observer](../../observer/2026-11-21-independent-review.md), [Actions #44683814370](https://github.com/kolihhan/name-kaggle/actions/runs/44683814370). Weak trial, **not** proof contrastive learning broadly fails. |
| 0.715 + weak Hungarian final postprocess | authenticated official Public fell from .715 to .594 in a previously submitted change. | [Source lineage report](../../research/ntl-0715-highscore-decision-lineage-20261123.md). Specific intervention plus competition rerun; not a theorem all structured decoding fails. |

## Current external-source evidence (actually counted; NOT an independently accepted 18-way holdout)

- [Actions #45186312734](https://github.com/kolihhan/name-kaggle/actions/runs/45186312734); [underlying result](../../research/ntl-adversarial-evidence-real-result-20261123.md):
  - ankiteshtiwari/ntl-exact-corpus-v3: 17,009 rows, 9/18 labels, no independent served-ID column;
  - ankiteshtiwari/ntl-blocks-v1: 1,800 rows, 2/18;
  - beroli/ntl-benchmark-c1-generated-panels: 360 rows, 12/18, 30 distinct prompt IDs;
  - beroli/ntl-benchmark-c2-generated-panels: 470 rows, 12/18, 50 distinct prompt IDs.
- C1/C2 publisher self-declares served_matches=true for all 830 records, but independently authenticated original provider receipts are absent. Last-segment literal model ID match only 530/830 due to potentially distinct aliases. These two sources lack GLM, Kimi and Qwen, plus three other specific labels. They have 0 shared exact prompt IDs/response texts; source identity/protocol and prefit overlap are not yet proven independent.
- Official TRAIN 2,699/150 and Oleg216 and older external subsets are **spent**; don't create novel data by resplitting them.
- There exist larger 18-label external pools (e.g., 75,783; 212,623) but size alone does not resolve training overlap, actual model identity, data rights, source bias or clean holdout design. [Original validation audit](../../../artifacts/research/ntl-18class-independent-validation-gate-20261111.md).
- Do not treat this absence of qualified full-18 external data as a blanket prohibition on any mechanism-oriented experiment. Likewise don't claim 18-way transfer on a 12-class panel.

## Strong champion code/execution state

- Exact 0.715 Notebook code source SHA pinned above. It performs pre-GLM strong inference and saves /kaggle/temp/final-logits.npz, then GLM v1_stage may change /kaggle/working/submission.parquet (original source has 32 code cells).
- [CPU-only trace source proof #45188580008](https://github.com/kolihhan/name-kaggle/actions/runs/45188580008) succeeded: SHA and all 32 original cells preserved, two diagnostic sidecars added (one snapshot before original cell29, one aggregates after final original cell), **but this derivative was never pushed to Kaggle or run on actual labelled examples**. It produces no measured gain and its sidecar may change runtime overhead. [Detailed report](../../research/ntl-frozen0715-stage-trace-result-20261123.md).
- Public Kaggle kernel preview saved output is only 3 rows (or not saved), **not the hidden rerun's full final ~4k per-row predictions**. This means there is no authentic post-GLM 0.715 same-row external BA, error matrix or source-conditional rescue/regression evidence yet.
- Nine-hour global budget; GLM stage configured to operate until 8.8 hours. Presumed "free instrumentation" is not measured on full-size real inference; time has opportunity cost.

## Operational evidence to examine adversarially

- On the linked current snapshot repository, there are approximately **444 GitHub workflow YAML files**, **210 scripts**, **46 docs files**, **197 artifact files** and **89 tests files**. Git file counts are **NOT** independent experiment counts or runtime costs. Large workflow/script proliferation is worth investigating as an organizational failure mode.
- The default main has far fewer files and stale chronology; do NOT silently review main instead of the specified branch.
- Repo RULES.md (owner newer than EVALUATION_POLICY addenda) allows bounded prompt-disjoint mechanism research without an immaculate full18 heldout. Previous periods explicitly allowed lawful Colab training for NTL, but the most recent practical scope has not authorized new paid API, unapproved Kaggle submit or blind new GPU burn. Review restrictions as changeable decisions with cost and pivot triggers.
- Some docs still call 0.691, 0.702 or 0.683 "current" (dated snapshots); DO NOT present them as current verified best. Audit whether the research process fails from inconsistent canonical state.

## Your Stage B job: don't just summarize this dossier

A. Open at least **one actual scorer implementation**, one scientific mechanism script, their exact terminal execution logs, the evaluation policy, source provenance audit, and one negative result. Cite links, paths and individual numeric evidence in your independent response. Label evidence not verified if you could not access it.
B. Mark the project claims as VERIFIED / NOT VERIFIED / INCONCLUSIVE, distinguishing correctness, source quality, and competitiveness.
C. Determine whether the central bottleneck is representation, source/protocol shift, model identity ambiguity, resource allocation, validation mismatch, error concentration, or experimental process. Demand falsifiers instead of opinions.
D. Confront the possible wrong next action: rerunning current 0.715 on **spent narrow Oleg216** against prior 0.691. Would it change a scientific or submission decision enough to justify a near-9h scorer? What would it actually estimate? Could a cheap new mechanism falsifier deliver more information?
E. Choose **one** next experiment (or a principled stop) with exact base and candidate, preexisting data/lineage, frozen split and labels, resource/time maximum, primary metric and negative control, rescue/regression by exact class, expected value of information, explicit kill gate. Do not prescribe a third party API or training/official submission as though already authorized.
F. State a plausible radically different approach if all incrementally patched source-conditioned ensemble directions fail; quantify what evidence would justify changing course.

**Write ROUND_B_ANSWER.md as a new answer, without modifying ROUND_A_ANSWER.md. STOP.** You are not permitted to train, run official Kaggle scoring, modify champion, or publish sensitive texts as part of this review.
