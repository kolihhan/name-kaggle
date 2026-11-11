# STABLEPAIR-18 — executable source/protocol capacity gate

**2026-11-12 (MYT).** Status: **auditor implemented and locally tested; real training/evaluation remains BLOCKED by source evidence**.

This is the execution follow-through to [PPO vs task-aligned decision](ppo-vs-task-aligned-objective-decision-20261112.md) and the existing [STABLEPAIR-18 pre-registered design](stablepair18-design-best-comparison-20261107.md). It avoids re-training another marginal contrastive head on consumed sources.

## Delivered

- [`scripts/stablepair_protocol_gate.py`](../../scripts/stablepair_protocol_gate.py): fail-closed CSV metadata check of matched sibling pairs, identical prompt fingerprint, model-agnostic request fingerprint, generation-settings fingerprint and requested target length; source lineage and upstream prompt origin independence; single served-model/revision consistency; license and prior-source-exposure fields.
- [`tests/test_stablepair_protocol_gate.py`](../../tests/test_stablepair_protocol_gate.py): 7 synthetic tests for eligible metadata capacity, shared prompt origin, reused exact prompts, mismatched requested lengths, unverified exposure, invalid hashes, duplicate inflation and invalid threshold. **7/7 passed locally**, Python pytest, 2026-11-12. No real dataset was run in this checkpoint.
- The gate **only checks metadata capacity**; even a `CAPACITY_ONLY_NOT_VALIDATED` result is NOT independent scientific validation, source-owner permission proof, 18-label benchmarking or authorization to train.

## Input schema (metadata-only CSV)

Columns required: `source_lineage,prompt_origin,prompt_hash,model,request_hash,settings_hash,target_words,served_model,served_revision,rights_verified,prefit_exposure`.

- Hashes are lowercase SHA-256 fingerprints of the original prompt, canonical model-agnostic request template, and canonical generation settings, respectively; no raw prompt, response, secret or request body is read or emitted.
- `source_lineage`: **underlying corpus/origin** identifier, not a derivative Kaggle filename; `prompt_origin` is the earliest known upstream prompt bank.
- `served_model` + `served_revision`: explicit served endpoint provenance; a guessed version or requested alias alone is not verification.
- `rights_verified=true` and `prefit_exposure=none` are strict metadata assertions, **not independently verified facts**. Unknown/missing flags are rejected. Original-source licensing and all incumbent component exposure still need external audit.
- Default capacity threshold: at least **100 unique matched prompts in each of two distinct source lineages**, with disjoint prompt hashes and upstream prompt origins, per specified sibling pair. This is the existing STABLEPAIR research target, **not a universal statistical adequacy guarantee**.

Run with privately acquired, rights-audited metadata only:

```bash
python scripts/stablepair_protocol_gate.py /private/path/source_metadata.csv --output /private/path/gate.json
python -m pytest -q tests/test_stablepair_protocol_gate.py
```

No public Git file should include raw official competition data, private model weights, external copyrighted responses, API credentials or generation prompts.

## Current evidence / action decision

- [Nine-source intake](ntl-18class-independent-validation-gate-20261111.md): **0/9 qualified** as a new unused independently validated full-18 model/protocol-comparable holdout; not proof that no such source exists.
- [347-row diagnostic](ntl-0679-transfer-diagnostic/RESULT_20261111.md): 71/87 errors are same-provider siblings on a **spent 12-label** cohort. Do not retune a sibling gate from those errors.
- [Earlier source feasibility](stablepair18-design-best-comparison-20261107.md): GPT-trio matching can be demonstrated in some pools, but true exact matching of served version/request/settings and independently unused source conditions is not established.
- The new gate has **not been run against a newly independently certified real corpus**, so **no observed pair passed** and no claim of new BA or scientific transfer is supported.
- **Training and new Kaggle submission remain BLOCKED**. Next concrete dependency is original-source, API-request, serving-version and upstream prompt-lineage metadata from at least two independent producer cohorts. Do not synthesize verification flags to obtain a pass.

## Frozen future mechanism test (only after a source audit)

1. Control C0: fixed representation, class-balanced pointwise sibling CE/ranking baseline.
2. C1: same capacity and train budget, add correctly matched same-prompt within-family contrastive/ranking term.
3. C-shuffled: preserve labels and source proportions but destroy true prompt matching.
4. Precommitted source-disjoint prompt holdout; paired same-row balanced recall and rescues/regressions, label/provider degradation and prompt-cluster uncertainty. Score incumbent on the same truly reserved rows if legally possible.
5. No automatic leaderboard or extra Notebook submission. Only independently validated positive residuals can motivate a frozen, one-shot official experiment.

**This commit is research infrastructure, not model quality.**