# NTL Apex forensic phase B — fixed typography rule contradictions (2026-11-19)

Frozen before the *new* analysis. This supplements, not rewrites, [Apex Residual diagnostic preregistration](../apex-forensics-20261119/PREREG.md). The prior CPU experiment finished but disclosed that `pngy_residual.refine` is **family-mass preserving** and its direct evidence argmax across families was an invalid substitute for the true Apex decision stage.

## Competing explanations
H_format_robust: Stylometric format markers correspond to model identities sufficiently reliably that the Apex postprocess fixed penalty is rarely applied to the correct known class on heldout generated responses.
H_format_fragile: The hard-coded marker/label exclusions are not reliable across sources or prompts and can penalize a true label despite genuine model identity, especially on nonstandard-text domains.

## Real available evidence
- Same previously spent frozen 18-class Oleg216 cohort: 216 responses, 12 per class, one prompt group per row, same file SHA and selectors from previous prereg; comparator hard predictions from previous **0.691** Apex predecessor on *the same* 216. The current 0.702 hidden competition logits and truth are not available. No fresh independent evaluation claim.
- Frozen `kolihhan/ntl-0702-apex-pinned-nov18` code SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`, extract exact `%%writefile pngy_residual.py` and source block `%%writefile apex_postprocess.py`. Do not run full model, train a substitute, rewrite source or add external data.
- Source's fixed last-stage format rules, in order:
  1. Text contains `" — "` and proposed label is **not** one of `claude-opus-4-6,claude-opus-5,glm-5.3,glm-5.3-flash`.
  2. Text contains `"’"` and proposed label starts with `claude-` or `glm-`.
  3. Text contains `"—"` and proposed label starts with `gemini-`.
  Original code multiplies the implicated classes' probabilities by **0.15** and re-normalizes *only if the current proposed class is forbidden*. This is a **soft penalty**, never a deterministic forced flip.
- Measure, for each fixed rule: number of responses exhibiting marker, number with true class forbidden by this rule, and number where the prior frozen strong prediction was **correct and that rule would trigger**. Union and by true class/family. The quantity is an **exposure / risk ceiling**, not measured rescues/regressions or 0.702 BA.
- CPU mathematical invariant check on `pngy_residual.refine`: use seeded 32×18 synthetic logits and strictly positive style evidence, `strength=0.10`; verify per-family `logsumexp` remains unchanged to numerical tolerance across four fitted families and ungrouped classes are unchanged. No source text, labels or row values printed.
- No model training in this phase B, no Kaggle GPU or submission, no new data; only aggregate counts. `n=216` with 12/class and sensitive-topic source shift can estimate label-rule contradictions on this cohort but cannot adjudicate performance on general hidden-test prompts.

## Decision
If ANY true-label violation occurs, reject interpretations of the marker mapping as a guaranteed model identity rule. If >0 frozen-strong correct predictions are exposed, mark format rules as a real potential damage source but **not observed flip**. If zero, don't infer a gain from removing it. No 0.702 stage promotion or new threshold based on this test. Given no 0.702 pre-Apex logits, score-facing C2 veto remains **unverified**, and must not be submitted on this evidence alone. Any subsequent test must reproduce the actual complete `refine → prompt_pair_adjustment → GLM keep → typography` pipeline, not aggregate family-agnostic evidence.
