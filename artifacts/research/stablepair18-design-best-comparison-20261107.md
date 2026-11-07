# STABLEPAIR-18 — source-stable conditional sibling evidence vs current Best

**Date:** 2026-11-07 MYT  
**Work type:** research and architecture design, **NOT an evaluated, trained or submitted candidate**.  
**Status:** design complete; **DATA IDENTIFICATION GATE OPEN / NOT SATISFIED**. No new GPU, training, Kaggle notebook, scoring, or leaderboard iteration.

## 0. Verdict

Do **not** build an unconditional new 18-way classifier or replace an already-strong ensemble with a small token statistic. The next research bet is **conditional sibling evidence that survives a change of independently recorded generation source/protocol** and adds **paired residual correct predictions** to a strong 18-way anchor. This is *not* a claim that pairwise classification, conditional probability correction, source-balanced risk, or contrastive objectives are original in isolation.

**First blocker:** current public sibling corpora audited in this repository do not provide adequately crossed **exact model × identical prompt × identical request/target-length × multiple recorded protocols**, with independent source holdout. Thus **protocol-robust identity is not currently identifiable** in these files. Source labels alone are weak environment proxies, not proof of matched protocol. Do not manufacture “counterfactuals” by rewriting text. Research can still pursue an explicitly **exploratory source-transfer** experiment, but that cannot establish causal protocol invariance or authorize a competition submission.

## 1. What exactly is the Best?

All public numbers below are **authenticated own-account Kaggle submissions**, *different* models evaluated on the same competition Public Leaderboard. They are **NOT paired OOF** scores, and the private metrics are not observed. Evidence: [submission history 42385772029](https://github.com/kolihhan/name-kaggle/actions/runs/42385772029), [Short-TRACE verdict](../candidates/short-trace-8b-direct-submit-v0/RESULT.md), [V47 audit](ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md).

| System | Kaggle submission | Public BA | Attribution | What we can infer |
|---|---|---:|---|---|
| Ozertuu Dual-SVC V3/Apex | 58022340 / 58045266 | **0.668** | **Best verified reproduction** of another author's public system | Full-pipeline benchmark only |
| Lightning V47 | 58008463 | **0.666** | Another author's notebook reproduced | Full-pipeline benchmark only |
| V47 length-40 window | 58078151 | **0.663** | Our frozen variant on a public baseline | Changed variant did **not** beat 0.666 |
| LoRA24 own two-stage | 57911893 | **0.520** | Independently developed model route | We have archived row-aligned own OOF diagnostics |
| Short-TRACE 8B standalone | 58093864 | **0.093** | Our frozen new representation experiment | Reject standalone; true/shuffled ablation never ran |

We should distinguish:
- **Competitive Best = 0.668 reproduced public system**, not proof our contribution.
- **Own-model baseline = 0.520**, not a fair proxy for the 0.668 model's current error modes.
- **New STABLEPAIR-18 score = UNKNOWN.** A design cannot be reported as beating either score.

**The bar:** a meaningful new Public BA target can be provisionally **0.688 (+0.020 absolute over 0.668)**, *not a prediction*; the exact score comparison is appropriate only **after** source-transfer + same-row base-versus-treatment validation passes and one frozen Code Competition submission is authorized. One Public number alone is not statistical proof of generalization.

## 2. Why generic hierarchy, larger evaluator and ATLAS copy are poor bets

Evidence from archived **LoRA24 2,699-row grouped OOF** (reused/spent, not a fresh validation set): plain exact-model BA 0.546761; wrong-sibling 0.259771, wrong-provider 0.193468 macro; 701/1,223 wrong rows were siblings. Provider correct: Gemini **0.975**, GPT **0.880**, Claude **0.8409**, GLM only **0.420**. [Exact archived analysis](independent-conditional-attribution-diagnosis-20261106.md). This means a hard first-stage family decision cannot generally rescue GLM; a sibling specialist may help GPT/Gemini but needs proof it corrects *new* errors.

Existing V47 already includes six model streams, trained sparse features, family-conditional specialists, inferred prompt groups, and an uncertainty-gated ATLAS-like likelihood expert. [Exact source audit](ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md). A copy of “classifier + likelihood + family gate” is not novel.

Other failures constrain the search:
- Prompt-residual LDA −0.03579 mean heldout-source BA, worse on 11/13 sources [result](../candidates/prompt-residual-style-v1/RESULT.md).
- Prototype prompt-invariant contrastive v2 +0.002685, interval crossed zero [prior diagnosis](independent-conditional-attribution-diagnosis-20261106.md).
- SURFACE-only < LEXICAL on both previously checked external banks; FULL lexical+surface did not win consistently [source-transfer test](source-transfer-views-20261106/RESULT.md).
- Contextual wording-choice position had +0.000746 vs Phase7 and failed gates [result](contextual-choice-cpu-v1/RESULT.md).
- Short-TRACE 8B T-only 0.093; no true-vs-shuffled paired treatment test was completed [result](../candidates/short-trace-8b-direct-submit-v0/RESULT.md).
- Length ordering Sol>Luna in SWEeper 195/196, but only 20/47 on a different source with requested target words confounded. **Fixed sibling length direction FALSIFIED as transferable shortcut** [cross-source falsification](generator-vs-channel-identification-decision-20261106.md#9-frozen-cross-source-falsification-of-sibling-length-identity--2026-11-07).

Literature scope:
- [ATLAS / MAABench (Ko et al., 2026)](https://arxiv.org/abs/2610.19583): source-conditioned token likelihood improves *model-family* attribution under version/domain shift, not a demonstration of exact-version sibling separation as 18 distinct classes.
- [OpenTuringBench (La Cava & Tagarelli, EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1354/): contrastive authorship representation under several shifts, not directly our 18 frontier exact versions.
- [Profiler (Guo et al., EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1265/): context-loss surrogate logits; simple logit fingerprinting is prior art, and our Short-TRACE result does not establish it as an effective complement.

## 3. The one central hypothesis

> **H1:** In a matched prompt and recorded comparable generation channel, pairwise model-specific residual evidence has more stable *sign* across sources/channels than a pooled 18-way classification decision, and corrects errors the strong anchor gets wrong.
>
> **H0:** Pairwise margins mostly learn source, target length, topical or serving artifacts; on held-out protocols they do not improve over an equally powerful pointwise sibling readout, or they break more strong-anchor correct predictions than they rescue.

This simultaneously needs a *mechanism* test and a *complementarity* test. A standalone 3-way accuracy win is insufficient. **Exact hidden model endpoint generation protocol remains unknown**; even successful source-shift tests cannot prove causality without a crossed design.

## 4. STABLEPAIR-18 architecture (working name, not novelty claim)

Only train one genuinely different scoring component, NOT multiple new large models:

\`\`\`text
                             Input response x (single short paragraph)
                                                |
                  +-----------------------------+----------------------+
                  |                                                    |
        Frozen full 18-label anchor B                             Source-stable
         (0.520 own for honest paired                  pairwise sibling margin head Q
          analysis; 0.668 public system                     (train on matched requests,
          only if row-aligned outputs exist)                 train source-balanced)
                  |                                                    |
           provider probability P_B(F|x)                 q_F(m|x): within-family
           + conditional P_B(m|F,x)                      source-stable conditional
                  +-----------------------------+----------------------+
                                                |
                    **soft provider-mass-preserving conditional correction**
                        P_new(m|x) = P_B(F|x) *
                          [ (1-alpha) P_B(m|F,x) + alpha q_F(m|x) ]
                                                |
                                       18-class prediction
\`\`\`

**alpha = 0.25 fixed before scoring** for the *first operational design*, zero is anchor control. No gate learned on OOF failures or Public LB. This preserves **family probability sums**, not necessarily the final argmax provider (a sibling change can change which individual class has top probability). Report cross-provider regressions explicitly. Families unsupported by credible matched pairs receive \`q_F = P_B(m|F,x)\` and remain unchanged. Family-singleton classes are unchanged.

### The research component: matched-prompt pairwise margin

Start with the real **GPT-5.6 Luna / Sol / Terra trio** only; external published banks have overlapping labels, unlike the misleading claim that all 18 are equally represented. Expand Gemini/Claude only after documented matched-protocol support. No source-specific inference gate.

1. Representation \`z(x)\`: first CPU feasibility uses **frozen lexical/style features** with fit limited to training source/prompt partitions, not low-dimensional rank histograms. A later *separate* contract may replace this with frozen contextual LoRA embeddings only if the CPU objective survives; do not pre-commit 8B/14B use for this design.
2. For sibling pair a/b, fit an antisymmetric margin \`r_ab(x) = s_a(x)-s_b(x)\` using pointwise balanced logistic loss **plus** same-prompt, same-recorded-protocol pairwise ranking loss:
   \`L_pair = softplus(-(r_ab(x_a,p,c) - r_ab(x_b,p,c)))\`.
   Use fixed equal source weight *among eligible source cohorts with both models*, and fixed equal weighting of pointwise and pairwise losses in the first pilot. No source ID as a test-time feature.
3. Combine scores from all pairs into within-family \`q_F\` (normalized per family). Train-fold-only feature fitting, and no labels for inferred test prompt groups. Never use prompt IDs or generation settings at inference.
4. This objective by itself is **not guaranteed invariant**. Its empirical claim depends entirely on transferred heldout source/protocol performance against the same pointwise scorer. “Counterfactual” is reserved for genuinely crossed requests, not arbitrary text edits.

### Mechanism controls (all predefined, not leaderboard variants)

- **B (anchor):** frozen original 18-way model, no correction.
- **C0:** identical \`z(x)\`, same family/label training data, pointwise sibling head (no paired loss), same fixed \`alpha=0.25\`.
- **C1:** same head/capacity + correctly matched prompt/request pair ranking (the treatment).
- **Cshuf:** same head/capacity/loss but **shuffle prompt partner within each eligible source and label pair**, preserving per-class/source label frequencies; breaks *prompt matching only*.
- **Cnuisance:** length/format-only scorer (negative/source-artifact comparator), never used as a deployed model.
- **Data audit:** measure source predictability and target-length correlation conditional on class; if one source is effectively one class/protocol, do **not** infer causal invariance.

To prove the *pairwise mechanism*: \`C1 > C0\` and \`C1 > Cshuf\` on source-disjoint material. To show useful engineering: \`B+C1 > B\` on same exactly aligned rows, with rescues/regressions, without collapsing provider accuracy. Full provider mass preservation by itself is not proof of this.

## 5. Data-first feasibility gate — the crucial missing prerequisite

We previously checked:
- 834-row external proxy corpus: *zero* exact sibling comparisons with equal requested target words on the same prompt/setting; requested target length itself depends on model label.
- Oleg/SWEeper responses: hundreds of within-source/same-prompt sibling pairs, but all from software-repair agent interactions with incomplete per-passage protocol/endpoint trace, not a matched short-standalone-paragraph design.
- Another plausible external source returned normal authenticated Kaggle 403, no data obtained.
- Two previously examined transfer corpora share **only GPT-5.6 Luna/Sol/Terra** labels, but their protocol and version parity are not fully established, and both have been reused as exploratory development evidence.

**Gate 0 is NOT SATISFIED:** before claiming protocol robustness or promoting a new candidate, obtain an *independent* train/transfer cohort with same prompt, model pair, fixed length/request, endpoint/version, time and generation settings, with multiple source/protocol conditions; document differing unsupported API sampling parameters. A concrete first-study scale is **at least 100 distinct prompts** per tested sibling pair and condition, held out at the level of prompt and entire source/protocol condition. This count is a preregistered research target, *not a statistical adequacy proof*. Prompt overlap and all fitted-source overlap with the strong anchor must be auditable.

If this is unavailable, outcome = **NOT IDENTIFIABLE / BLOCKED**, not “contrastive failed” and not “scored worse than Best”. Existing mismatched source corpora may support *descriptive/exploratory CPU evaluation only*, with no claim of causal protocol invariance or 18-way Public BA improvement.

## 6. What is a genuinely fair Best comparison?

| Comparison | Exact measure | Status / caveat |
|---|---|---|
| Published Best 0.668 vs new framework | Public BA **on Kaggle** | New score unknown; may run only after offline promotion gate, one frozen submission |
| Original 0.520 vs B+C1 | **same 2,699 official rows**, same prompt-grouped five folds, no fold contamination; per-row rescues and per-class/provider errors | Archive is reused, thus descriptive/secondary, not fresh transfer |
| 0.668 full pipeline vs B+C1 | **same independent source/test rows**, same eligible 18 labels, source/prompt/anchor-training disjoint, paired probabilities/logits | **NOT CURRENTLY AVAILABLE**: no audited row-aligned fresh V47 predictions, incomplete external training-provenance overlap checks |
| C1 vs C0 vs Cshuf | **same heldout source, same sibling labels**, pairwise macro recall and prompt-bootstrap interval | Not executed; 3-class BA must never be passed off as 18-class BA |
| Provider protection | Exact model 18-class BA, provider prediction macro, source-weighted BA, rescues vs regressions | Needs complete 18-label heldout; source with only GPT trio cannot evaluate GLM/Kimi regression |

**Hard stop before official competition submit:** (i) source/protocol gate satisfied or a clearly downgraded noncausal objective independently validated; (ii) C1 beats both C0 and Cshuf by >=0.03 **absolute sibling macro BA** in at least two truly independent source/protocol conditions, and prompt-bootstrap lower bound for C1-Cshuf is >0; (iii) B+C1 improves same-row whole-system 18-label BA over B by >=0.01 absolute without increased provider-misclassification macro by >0.01, with uncertainty interval supporting a positive whole-system gain; (iv) strong 0.668 anchor can be evaluated honestly under verified source/provenance isolation; (v) Kaggle code-competition full inference preflight within 9 hours and one frozen submission. The +0.02 above 0.668 is a **subsequent competition goal**, not a result predicted by this plan.

Failure at any stage is a scientifically meaningful stop, not permission to tune a sibling gate on Public LB.

## 7. Decision / resource and research value compared to Best

**STABLEPAIR-18 can potentially be a low-cost add-on** if a fixed CPU text feature head suffices; compared with the V47/Ozertuu already-trained ensemble, that new head would be small. The **end-to-end inference budget would still include the existing heavy Best pipeline** and potentially require extraction of its intermediate scores. Therefore claiming it is "cheaper than Best" before implementation would be false.

Unlike V47's generic family gate, the **specific falsifiable question** is whether model-pair margins learned from *matched prompts under matched request conditions* retain their sign/utility across source/protocol changes. Its novelty is a *hypothesis and controlled evaluation*, **not** the family gate itself. A source-disjoint, aligned residual rescue demonstration is required for any contribution claim.

**Immediate next feasible work without GPU:** a **CPU-only source-support/data-provenance audit** for the GPT-5.6 trio, identifying exact usable matched prompt/protocol rows from *already allowed metadata* and whether fresh untouched source support exists. Do **not** repeat old inventories or reopen unavailable Kaggle dataset 403. If no new comparable cohort exists, close/park this route and put resources into acquiring well-controlled data rather than submitting another engineered variant. If Gate 0 passes, freeze an executable C0/C1/Cshuf pilot *before fitting*, with original data and exact source identifiers held privately and only aggregate public results.

**This deliverable is a researched design and comparison, not evidence of a stronger scored model.**
