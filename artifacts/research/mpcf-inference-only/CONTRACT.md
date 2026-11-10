# MPCF — one-shot no-training real-data feasibility, precommitted before execution

**Scope:** Test the previously implemented Matched-Prompt Contrastive Fingerprinting primitive on public licensed third-party short responses. This is an exploratory **same-source**, prompt-held-out, **binary sibling-pair** study, NOT a new 18-way benchmark, not independent source transfer, and not a Kaggle submission.

## Why this is materially different

Compare a query against answer **differences of two known generators responding to the same reference prompt**. The reference pair centroid is used to retrieve related *other* prompts; its A-minus-B residual votes for A or B. No test-group Hungarian matching and no prompt_id at query inference. Intended mechanism is cancellation of prompt-shared content, not merely another fitted classifier or blend.

## Frozen data and eligibility (no result-driven selection)

Public source: `olegzholobov/name-that-llm-response-corpus`, specifically `zlab-princeton/SWEeper-Bench-traces`; files `external_short_wide.parquet`, `external_cell_metadata.parquet`, `external_cell_provenance.parquet`, `external_response_metadata.parquet`.

Exactly three predeclared binary pairs: `gpt-5.6-luna / gpt-5.6-sol`, `gpt-5.6-sol / gpt-5.6-terra`, `glm-5.3 / glm-5.3-flash`. Other pairs have no comparable same-source support in the preceding actual public-data audit (#42362668801). Require both texts for the same exact prompt and the same source; `view=short`, `prompt_exact=true`, eligible and text-only suitable if columns exist, and exclude the audited hazard flags. Length: **40–250 whitespace-separated words per answer**. Deduplicate normalized prompt string, then partition by SHA256 ordering of unique normalized prompt strings into 70% reference, 30% heldout; reject repeated response text across partitions.

Prior audit reported **196/199/200** same-source matched prompt keys before this stricter short-text length and overlap filtering, **not** guaranteed sample sizes for this test. If <16 reference or <10 heldout unique prompts in any pair, record **INSUFFICIENT**, no change of length limits or selection rules to rescue.

## Fixed inference representation, comparisons and boundaries

Use `sklearn.feature_extraction.text.HashingVectorizer` with **char n-gram 3–5**, `n_features=4096`, `norm=l2`, `alternate_sign=False`, lowercase; transform-only, **never fit**. Use current fixed MPCF source with **top_k=16**, cosine retrieval by A/B pair midpoint, median normalized pair residual evidence. Query texts are held-out, so no paired query answers are required.

Binary baselines on exactly the same heldout items:
1. **Matched** reference pair residuals (hypothesis).
2. **Prompt-shuffled** B reference partner (single deterministic seed `20261110`), preserving all reference examples and model labels but destroying paired-prompt correspondence.
3. **Unpaired** A and B class-mean prototypes (deterministic reference lookup; no weights learned).
4. **Length-only** reference-median nearest class (shortcut check).

Score binary balanced accuracy (mean sensitivity for both labels; zero evidence counts 0.5). All values are `binary same-source heldout`, never 18-class official balanced accuracy. Report sample counts, source/hash, and matched minus both controls. Do not upload individual response texts, official Kaggle data, predictions, identifiers, or checkpoints to public Git.

## Frozen decision

**Local feasibility positive only if every eligible pair** has matched BA ≥0.55 and matched-minus-shuffled ≥0.05 and matched-minus-prototype ≥0.05, **AND** no length-only shortcut matches or beats MPCF. This is intentionally demanding, not a prediction. Even if positive, the scientific result is **EXPLORATORY**, because the source is SWEeper software repair, protocol/version/decoding settings are not fully evidenced, and no independent different-source transfer or aligned strong champion outputs exist. Promotion to competitive work separately requires documented source-disjoint short-text cohort, strong-anchor same-row results and a frozen full-stack test showing ≥0.02 improvement without provider collapse. Otherwise **REJECT/INSUFFICIENT/NOT YET TRANSFER VALIDATED**.

No GPU, no neural or supervised model fitting/calibration, no Kaggle Code Competition notebook, no official submission, no Public LB-based selection, no subsequent within-study parameter tuning. If control shows no benefit, close **this particular frozen hash-vector MPCF feasibility**, not every possible pretrained encoder or reference pairing algorithm.

## Execution

Extend and reuse existing `.github/workflows/ntl-olegzholobov-pair-audit.yml` rather than creating another workflow. New runner `scripts/run_matched_panel_pilot.py` should reuse `scripts/matched_panel_evidence.py` and print aggregate JSON only. Run focused synthetic unit checks and exactly one public-data CPU comparison. Report actual result without overclaiming.
