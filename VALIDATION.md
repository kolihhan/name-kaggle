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

No new experiment run, GPU inference, or official submission created by this file.
