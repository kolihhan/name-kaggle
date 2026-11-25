# Lightning v52 exact 0.724 mechanism — 2026-11-25

Authenticated Kaggle public best 0.724 is Version 52, scriptVersionId 373716012. Exact historical source SHA256 74972fd38a769bf685105c5d4980ca1ff4a5921a37c4cab6b8a14bfdb1427f53. LumiBee v1 0.722, Ranjeet Sibling Odds v1 0.721. Evidence: [score](https://github.com/kolihhan/name-kaggle/actions/runs/45573015308), [exact version source](https://github.com/kolihhan/name-kaggle/actions/runs/45573128983), [routing and visible logs](https://github.com/kolihhan/name-kaggle/actions/runs/45573677383).

## Exact findings
- v52 vs LumiBee v1: 14 emitted modules identical after normalizing paths/whitespace. Only two common modules differ: gsq_stage.py removes 5400-second subprocess timeout; infer.py removes timeout argument. Additional V1 pipeline files are separated/embedded rather than a wholly new classifier. Same sibling odds and residual modules, same broader GLM protection, same 0.25 PB115 hybrid parameter.
- v52 sets gsq_share=.035, LumiBee .03. **This is a probability blend weight, NOT the fraction of test rows routed to GSQ.** infer.py passes all test IDs/texts into GSQ and computes P=(1-share)*P+share*Q27. Start deadline remains; removing timeout does not guarantee a stage completes before the global Kaggle 9h deadline.
- v53 changes share back to .03 and introduces a 75/25 PB115/legacy likelihood probability blend via logaddexp. Do not attribute v53 changes to v52's scored 0.724.
- Original Evgen36 lineage 0.718 also has .03 GSQ share, 5400s timeout, no PB115, narrower Fable-only guard.
- Visible Kaggle logs are not hidden scoring logs, and show no authenticated actual PB115 execution or GSQ timeout events. Therefore cannot assign the .002 difference to share, timeout, stage coverage, or other orchestration.

## New mathematical constraint
If P and Q27 are normalized per-row probability vectors and all other stages are held fixed, raising share from .03 to .035 changes any pairwise GSQ-stage probability margin by at most .01 (0.005 times the range [-2,2]). A margin >.01 after the old blend cannot flip directly at this stage due solely to the share increase. Subsequent nonlinear stages can still change decisions. This is a mathematical bound, not measured accuracy.

## Decision
Do not claim a new expert or a new attribution representation. The .006 public gap relative to the owner's 0.718 is real external reference only and is multi-variable; 0.724 still .036 short of goal. A raw GSQ weight sweep is low information and risks LB overfit. The alternative research mechanism is confidence-conditioned use of existing expert disagreement, evaluated using actual aligned P and Q27 plus final predictions; no such cache is verified for the exact champion, so no accuracy claim or official submission. Preserve 0.718. No new training or GPU.
