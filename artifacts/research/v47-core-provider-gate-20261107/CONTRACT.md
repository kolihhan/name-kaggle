# V47 core provider gate

**Frozen 2026-11-07 MYT, before evaluating the candidate.**

## Hypothesis

Inferred-group votes can improve sibling selection and can also move the core winner to the wrong provider. Keep their full score vector on provider agreement and use the no-group core score vector on disagreement. This is a conditional readout, distinct from the closed proposal to remove all group votes based on mixed-prompt prevalence.

On the previously consumed LoRA24 Gate-B archive (artifact 13125872633), the user-reported fixed replay improved balanced accuracy from 0.54676 to 0.55233 over 2,699 rows (207 rescues, 192 regressions). It used known prompt IDs and its prompt-bootstrap interval crossed zero. It motivates testing with inferred groups. [Prior archive audit](../independent-conditional-attribution-diagnosis-20261106.md)

## Fixed candidate

- Baseline is Lightning V47, Kaggle script version ID 365000313, official submission 58008463, Public 0.666. Joined code-cell SHA-256 is `c50abf03f5d5472dc2404f62d000df659001e2b73def475c17b45d581c4199dc`. [Source and training lineage](../ntl-0520-vs-lightning-v47-0666-mechanism-audit-2026-11-05.md)
- Reuse the exact V47 source snapshots and all existing checkpoints, CPU fit states, fixed weights, labels, inferred grouper and downstream ensemble. No new training data or retraining. Upstream already-fit assets include five-fold and full official-data fits and external training sources. The source audit records the available provenance and its gaps.
- In `core_readout.py`, compute the original sum and the sum excluding keys ending in `_twin` or `_gauss`. Divide both by the original `max(1, active)`, with `active` counted over the complete vote dictionary across the batch. Use the original top-label order on ties. The provider families are Claude, Gemini, GLM, GPT, Kimi and Qwen. Select the entire original score row when top-label providers agree, otherwise select the entire excluded-vote score row. The existing ensemble receives this through `core.npz:logits`.
- Freeze the rule, weights, thresholds and decision before inspection. The three-row preview is bypassed only in the execution context of a smoke test, with no alternate candidate source.

## Evidence gates and stop rule

1. Check the exact code diff, same-provider and cross-provider routing, denominator and downstream score handoff. Run V47 inference without the preview and validate the produced `submission.parquet` against the official schema, ID order, row count and valid labels. Format-only and synthetic checks do not satisfy the real-inference gate.
2. For promotion, preserve the permanent prompt-grouped five folds. Every member, head and fitted upstream state must exclude the validation prompts of its fold. Final fitted V47 assets and LoRA24's OOF rows cannot establish candidate OOF. Lack of fold-safe V47 assets or sufficiently verified provenance blocks promotion under `EVALUATION_POLICY.md`.
3. The transfer check is one paired V47-versus-gate comparison on an independently labeled, exact-version, multi-prompt corpus with all 18 labels and source and prompt overlap ruled out against V47's fitted training sources. Infer groups from text. Use balanced accuracy, group coverage, provider agreement and cross-provider rescue/regression counts, within-provider sibling changes and per-class recall. The cohort must be fixed before candidate inference. If no cohort can be verified, the check is blocked.
4. Only after preflight and all promotion gates pass, submit one pinned version. Compare its official Public score to 0.666. **At least 0.671** earns follow-up, **at most 0.661** closes the variant, and a score in between is inconclusive. An execution error or missing score gives no scientific decision.
