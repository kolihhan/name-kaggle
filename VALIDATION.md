# VALIDATION — Name That LLM (2026-11-16)
Canonical governance: [EVALUATION_POLICY.md](EVALUATION_POLICY.md), [NOW.md](NOW.md). This is a prospective correlation ledger; **do not pair scores from different candidates or evaluation distributions as comparable CV**.

## Historical official pairs (original grouped OOF; repeatedly inspected and SPENT)
| Candidate | Prompt-grouped 5fold OOF BA | Kaggle Public BA | Official score ref | Interpretation |
| --- | ---: | ---: | --- | --- |
| historical Phase7 | 0.370052 | 0.369 | 57838845 | old adaptive reference |
| historical Phase15 | 0.405322 | 0.357 | see EVALUATION_POLICY | OOF improvement did not transfer |
| attributed public reproduction (Nov 13) | N/A: pretrained/no comparable honest OOF | 0.683 | 58295802 | real Kaggle score; not matched CV |
| current frozen champion Lightning V2 | N/A: pretrained/no comparable honest OOF | **0.691** | **58375806**; authenticated [Action 43743764085](https://github.com/kolihhan/name-kaggle/actions/runs/43743764085) | current actual Public best; original author's title not score |

**Correlation status:** historically the two Phase scores move opposite directions; n=2 pairs, multiple adaptive comparisons, no honest correlation estimate. Current champion cannot be assigned fabricated OOF. Source of historic Phase pairs: frozen [EVALUATION_POLICY](EVALUATION_POLICY.md).

## Existing independent-looking evidence — NOT qualifying confirmation
- Oleg external sensitive-topic only **89/216 = 0.412037** (exact 18 classes, 12 per class); row-stratified bootstrap interval **[0.3704, 0.4537]** from [authenticated Action 43804320931](https://github.com/kolihhan/name-kaggle/actions/runs/43804320931). Not an estimate of Kaggle Public/Private BA; topic, source and possible prefit exposure confounded. **SPENT**.
- Prospective non-sensitive Oleg pool: 427 eligible texts, 10/18 class coverage, 75 distinct prompts; only 4 fully same-prompt 10-class groups ([Action 43808061989](https://github.com/kolihhan/name-kaggle/actions/runs/43808061989)). **INELIGIBLE** as independent full 18-class confirmation. Not a model loss.
- 2,699 official train labels and permanent folds were adaptively used; another salt is not a new holdout.

## Prospective paired evaluation contract — UNRUN
H0: an architecturally different method does not improve 18-class BA versus frozen 0.691 champion under source/prompt-disjoint transfer. H1: positive transfer improvement.

1. Freeze source IDs, actual served model version, license, prompt/request lineage and all 18 labels **before model inference**. Split by upstream source/prompt/response lineage, never by individual rows; create a truly untouched confirmation partition. Overlap with any incumbent prefit training data must be assessed; UNKNOWN overlap blocks a clean independence claim.
2. Same exact rows for champion and candidate; fixed model snapshots, prompt preprocessing, seeds and inference budgets; store predictions in private artifact (not public raw competition text). Report per-class recall, paired rescues/regressions, clustered source/prompt bootstrap percentile 95% CI and multiple seed variation; qualitative error analysis follows frozen decision.
3. **Sample gate proposal, NOT YET USER-CONFIRMED:** at least 30 independent source/prompt clusters per true label, minimum 540 fully labeled responses, representation across multiple real domains/sources; this is an operational floor, not proof of power. Recalculate the required n via pilot *paired discordance* and desired detectable effect before approving promotion. E.g., BA 0.70 with 30 independent items/label has naive binomial 95% half-width around 0.038; prompt/source correlation widens it. A +0.009 true effect would require substantially more evidence.
4. **Provisional promotion test (must be frozen before any new run):** candidate's clustered paired 95% CI lower bound >0, no catastrophic class regression, legitimate provenance, and budget compliance; otherwise status **INSUFFICIENT EVIDENCE**, not global rejection. Holdout not recycled to optimize.
5. Before any official submission write both (local grouped CV or `N/A + reason`, confirmed-source transfer BA with CI, Public LB result) and why candidate selected. Selection rules by independent transfer robustness vs complementary error coverage; never by best observed Public.

## 2026-11-16 — prospective sensitivity check (CPU arithmetic; NOT an experiment result)
**Hypothesis audited:** proposed `30 independent examples per label` (= 540 total across 18 labels) is sufficient to **detect a tiny +0.009 BA gain** of the kind separating 0.691 from 0.700. **Result: not sufficient under ordinary paired-variance assumptions**.

For balanced equal-class paired binary correctness differences `D in {-1,0,1}`, hypothetical discordance rate `q=P(D!=0)`, target true paired BA gain `delta=0.009`, one two-sided alpha 0.05 and nominal 80% power, approximate optimistic normal-theory independent-response count:
`N ≈ (z_.975+z_.80)^2 * (q-delta^2)/delta^2` (z sum ≈ 2.8016). No data was sampled and q is **assumed**, not measured on a fresh holdout.

| Assumed paired discordance q | Nominal independent observations N | Per class (18 equal classes) | Detectable delta with N=540 (nominal) |
| ---: | ---: | ---: | ---: |
| 0.10 | 9,683 | 538 | ~0.038 |
| 0.20 | 19,373 | 1,077 | ~0.054 |
| 0.30 | 29,063 | 1,615 | ~0.066 |

**IMPORTANT:** Reused prompts, provider/source groups and class imbalances make effective N lower; cluster-aware uncertainty and pilot discordance may require **more** than table values. This is a sensitivity illustration, *not a verified requirement to collect 20,000 rows or a power guarantee*. The 540-row proposed floor may detect **larger** interventions, but it cannot be used to certify a marginal +0.009 as statistically robust. Do not re-split the 2,699 known training rows, and do not use Public LB to estimate q or tune the sample size post hoc.

**Decision:** keep the minimum 540 as an *exploratory floor, not advancement evidence*, and prioritize clearly orthogonal candidates expected to produce an effect above plausible detectable delta. For any proposed promotion, size the new source before model evaluation from a fresh blinded pilot, paired disagreements and group correlation; freeze n/CI/decision before scoring.

No new experiment run, GPU inference, or official submission created by this file.
