# NTL research-methodology correction and evidence map — 2026-11-27

## Correction
Do not equate failures of a prompt classifier with failures of reference-conditioned modeling. Actual 14B anonymous letter-choice run 45998578119: 2/36 BA=.05556, permutation .05625, panel agreement 0. Actual 14B Yes/No run 46003869781: within-family rank 16/32, cross-family 17/36, all Yes-minus-No logits negative, panel ranks unstable. Neither implements OSST, neither trains a verifier, neither evaluates new incremental evidence versus v36, and both reuse two spent query prompt clusters. Retire these exact prompts, not the general hypothesis.

## Evidence and non-equivalent methods
- OSST: https://arxiv.org/html/2510.13302v3 section 3 conditions on neutral and original reference text, and neutral target, then scores actual target token log likelihood. Not letter/Yes-No probability. Human authorship benchmarks are not exact LLM version identification.
- ATLAS: https://arxiv.org/abs/2610.19583, source-conditioned learned LM likelihood, not zero-shot style judgement. Reported family-level transfer is not evidence of exact 18-class competition gain.
- Prior actual GLM source-likelihood result: artifacts/research/ntl-glm-frozen-likelihood-20261114/RESULT.md: matched ranking 146/150 on spent train, fixed binary BA .8933; old .683-lineage external 252 rows +5 net (6 rescues/1 regression), p=.125, observational. The .718 lineage already uses a GLM likelihood stage; do not relabel this as a novel hypothesis or assume old gain adds to .718.
- Current lineage external panel docs/research/ntl-fable252-v2-actual-252-results-20261125.md and ensemble-decomposition decision: 151/252, 51 cross-family + 50 within-family errors, all spent and unverified publisher labels. Stage cache audit 45944364110: two 75-row scores missing row IDs; no aligned complete stage margins. Prior GPU, historical OOF, and CPU classifier runs are not independent validation of the current champion.
- Stage0 45997901312: 40659 metadata-screened candidates of 238750; 16/18 crossed labels; Kimi/Qwen fail and five GPT classes lack literal served-label matches. No rights or source independence established.

## Freeze the research question
Can independently qualified evidence for exact generator identity improve the actual .718 lineage on the same rows, beyond its existing signals and within the hidden runtime budget? No further ad hoc prompting experiments, small encoder replacements, threshold sweeps, blind submit, or retry loops.

## Ordered work (not yet completed)
1. Recover the exact scored executable and asset manifests, enumerate each stage's information inputs and outputs. Determine whether stage scores/IDs and known cache provenance can be recovered without inference. Do not reconstruct ID order by guess. If unavailable, write an ID-keyed trace contract for next authorized inference (core, per-expert class logits, family/sibling likelihood, final predictions, model and source hashes).
2. Separate existing v36 errors into observed stage transitions versus unobserved mechanisms; make an exact-model versus family confound matrix. Preserve same-row alignment and label-blind candidate selection. Do not tune against spent cohorts.
3. Resolve full-18 source eligibility, upstream rights and served identity. Until then use spent train only for implementation/mechanism debugging, not promotion. Stop if no feasible cohort.
4. Only if available resources and comparative evidence justify it, compare actual token-level OSST (neutralization, teacher forcing, reference-vs-no-reference) and learned source-conditioned likelihood / verifier against the same existing-stage baseline. Choose method once data and runtime feasibility known. If OSST cannot fit budget, do not run a cheap prompt surrogate and call it a replication. True reference versus label-shuffled references, equal-length control, independent prompt/source directions, same-row rescues/regressions, confidence intervals by prompt cluster. Full18 BA >= baseline+.02 with paired CI above zero, measured runtime, then one official candidate. Training/GPU require separate authorization.
5. If CPU-only data and traces cannot answer the comparison, record the blocker, retain .718 and defer GPU science. No new speculative experiment in this correction.

The objective remains .760; no paper or historical diagnostic establishes that score. Do not assume any independent source or candidate is already admitted.
