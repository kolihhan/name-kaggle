# Claude blind review — Stage C: after independent Round B is frozen

Do NOT open this until you have completed and saved both ROUND_A_ANSWER.md and ROUND_B_ANSWER.md. This stage is **adversarial calibration of your own independent review**, not permission to rewrite earlier answers or choose the most pleasing narrative.

## Earlier blind reviews — historical, not current authority

Read these original dated review prompts and the separate notes about errors and contradictory evidence:
- [2026-11-19 Claude Round 1](../NTL_CLAUDE_BLIND_ROUND1_20261119.md)
- [2026-11-19 Claude Round 2](../NTL_CLAUDE_BLIND_ROUND2_20261119.md)
- [F2 pre-existing falsification evidence](../NTL_CLAUDE_ROUND2_F2_FALSIFICATION_ADDENDUM_20261119.md)
- [Claude Round 2 factual corrections](../NTL_CLAUDE_ROUND2_CORRECTIONS_20261119.md)
- [Independent observer 2026-11-21](../../observer/2026-11-21-independent-review.md)
- [Blind phase A original 2026-11-18](../../../blind_review_A.md)

Those prior reviews addressed a different incumbent (~0.702 or older), not today's verified 0.715, and are not evidence the latest scorer is improved by the suggested methods. Some are **prompts and critiques**, not Claude's actual model answers; do not invent a Claude response if it is not recorded.

## Controlled adversarial checks

1. **Old reviewer error:** The earlier claim that 23 typography contradictions were "TRAIN rows" was false. They were 23/216 from a spent Oleg external panel, not 23/2699 train; no reliable Public BA gain ceiling follows. Examine whether any new calculation you made similarly used the wrong cohort/denominator.
2. **Previously attractive Hungarian structure:** 149/150 training prompt groups are essentially 18×1, but hidden prompt IDs and multiplicity are unknown. Third-party participant reported grouped OOF .374→.481 but Public .353→.350. Own later hard group assignment caused .715→.594. This does **not** refute every possible prompt-aware model. What new non-posthoc prediction distinguishes your proposed idea?
3. **Frozen embedding negative:** A source-disjoint 18-class study found MiniLM-L6-v2 + linear head .107407 vs char + linear head .213704 on matched rows. Old pretrained-geometry LoRA vs dense numbers used other cohorts. A failed frozen MiniLM does **not** refute strong pretrained multi-layer source-disjoint contrastive learning, but it lowers optimism for easy embedding wins.
4. **Learned contrastive negative:** Qwen3-1.7B short trial CE .078889 vs CE+InfoNCE .071667 (spent 1,800 external rows, 62 rescues/75 regressions). Is the issue objective, capacity/training, source identity, evaluation domain, or all of them? Do not generalize to all SCL without controlled data.
5. **New readout ATLAS A1:** fixed +one-feature 53→54 gives only +1/300 over .943333 binary, interval crosses zero, strong redundancy. Does this justify retiring all ATLAS source-conditioned learning? **No**. But what distinct, budget-aware falsifier is worth trying if you reopen it?
6. **Operational red flag:** 444 workflows, 210 scripts, many action green runs; zero recently verified independent 18-way challenger improvement over 0.715. Would you delete/simplify any loop? Which *existing* output should be reused instead of adding another script?
7. **Data false dichotomy:** an untouched perfect full-18 external cohort is desirable for a generalization claim, not logically necessary for a constrained scoped negative test on subset classes. But a subset score cannot be promoted as full18 BA and cannot diagnose missing families.
8. **Cost trap:** Rerunning the exact 0.715 model on the spent Oleg216 cohort may cost nearly the 9-hour inference budget and would only produce a changed-model comparison on a narrow already-analyzed domain, not independent evidence of incremental research novelty. Does its information justify the cost? Answer from first principles, not because the previous assistant suggested it.

## Final task: deliver a standalone critic's decision

Write ROUND_C_FINAL_REVIEW.md; do NOT overwrite Round A/B. Maximum 1,300 words.

- **What changed from your truly blind A to evidence-backed B to old-review-calibrated C?** Show 3 deltas; keep original answers unchanged.
- **What actually worked**, where, and what *did not work*? Separate terminal official Public score, source-only code validation, historical grouped OOF, small binary diagnostic, external source stress test, and authentic independently validated improvement.
- **Top 3 actual mistakes in this team's *research management***, not only model code. Include source locations, falsifiable evidence, cost/risk, and how to avoid repeating them.
- **Top 2 reasons your own recommendation may be wrong**. Do not manufacture agreement with the owner or prior Claude.
- **One chosen next research action with a scientific decision value**, cost ceiling, exact unmodified control, data/split, metric, false-positive risk, minimal acceptance gate and stop condition. If no experiment has positive value of information under current resources, state STOP with a trigger to reopen.
- **Do we need (a) better representation, (b) better/source-matched data, (c) a different inference decision architecture, or (d) a different research operating process?** Rank and justify rather than automatically choosing one.
- Explicitly state **GO / PARALLEL / PIVOT / STOP**, confidence level, and what single observation would most strongly overturn this choice.
- No Kaggle official submission, paid generation, GPU run or pretrained weight change is authorized by this review.

If the original evidence is insufficient, respond INCONCLUSIVE for the claim. A review can be valuable even with no algorithmic winner.
