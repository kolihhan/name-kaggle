# Paired sibling signal audit — 114 descriptive tests (2026-11-12 MYT)

**Executed an original aggregate-only audit on the existing 2,699-row official training corpus.** These are descriptions of *already used* prompt/model answers, not fresh heldout validation, classifier accuracy, a new candidate, a Kaggle submission, or confirmation of external transfer.

## Reproducibility

- Real input: existing official 2,699 rows, 150 distinct prompts, 18 exact target labels. Git blob SHA `d892e62d0f9e13ce1619ceb9da349a1e6092320e`. No source passages, exact prompts, row IDs or hidden test labels are present in this report.
- Implementation committed at [`scripts/audit_official_sibling_signals.py`](../../scripts/audit_official_sibling_signals.py), with [six local synthetic tests](../../tests/test_audit_official_sibling_signals.py), **6/6 PASS**. Real-data 114-test statistics were calculated from the GitHub blob with a semantically equivalent read-only paired-sign implementation; no raw data stored in research artifacts.
- Pair selection: six sibling comparisons from existing frozen diagnostic concern (GPT Luna/Sol and Sol/Terra, Gemini 3.7/3.8 and 3.6/3.8, Claude Opus 4.6/4.8, GLM 5.3/5.3 Flash).
- Feature selection: 19 predefined simple numeric/binary properties (whitespace token counts and average length, >=9-letter word percentage, punctuation rates, curly Unicode quotes, em dash, question mark, four connective terms, conclusion phrase, newlines, Markdown, digit).
- For each shared prompt, subtract `feature(A) − feature(B)`. Count positive, negative and tied prompt comparisons. Two-sided exact binomial sign test on non-ties; Benjamini–Hochberg FDR across 6×19=**114** tests. This is **exploratory/descriptive**: feature list, pair list and official rows had already been inspected during development. Nominal p/q values must not be reinterpreted as prospective independent evidence.

## Aggregate outcomes

All six below meet the descriptive q<0.05 criterion, with at least 35 non-ties:

| Pair (A vs B) | Feature | A > B | B > A | Ties | Median A−B | BH q |
|---|---|---:|---:|---:|---:|---:|
| Gemini 3.6 vs 3.8 Flash | mean whitespace-token length | 38 | **112** | 0 | −0.183 chars | 0.000000132 |
| Claude Opus 4.6 vs 4.8 | commas per 100 whitespace tokens | 39 | **111** | 0 | −1.455 | 0.000000193 |
| Claude Opus 4.6 vs 4.8 | words >=9 letters per 100 alphabetic words | **105** | 45 | 0 | +2.120 | 0.0000402 |
| GPT 5.6 Luna vs Sol | words >=9 letters per 100 alphabetic words | **104** | 46 | 0 | +1.474 | 0.0000704 |
| Claude Opus 4.6 vs 4.8 | mean whitespace-token length | **100** | 50 | 0 | +0.155 chars | 0.00124 |
| Claude Opus 4.6 vs 4.8 | periods per 100 whitespace tokens | 55 | **95** | 0 | −0.485 | 0.0260 |

**No descriptive q<0.05 effect for GPT Sol vs Terra, Gemini 3.7 vs 3.8, or GLM 5.3 vs GLM 5.3 Flash across this narrow 19-feature probe.** Non-significance is not equivalence or absence of model fingerprints.

For total whitespace word count itself, prior [paired-length negative control](ntl-official-sibling-length-negative-control-20261112.md) found Luna vs Sol **74/75/1** and Sol vs Terra **74/75/1** (A/B/ties); no reliable universal ordering. This is consistent with earlier refutation of cross-source transferred length-order heuristics.

## Adversarial interpretation

- Some fine-grained lexical and punctuation regularities exist *within the same prompts*. Those may help learn a residual representation **if and only if** they remain stable on independently acquired, source/protocol-comparable text.
- The strongest incumbent already contains lexical/style specialists. **A feature difference is not necessarily new complementary information beyond the 0.679 system.** Without same-row incumbent logits/predictions and unseen source rows we cannot compute rescued-vs-regressed or improvement probability.
- Prompts and generation protocols might still influence text length, lexicon or punctuation; requested target lengths and served model revisions are not recorded in official train. Do not assert causality or generator intrinsic fingerprint.
- Specific discovered features were selected from already-used data. Do not implement fixed threshold rules, source-conditioned gates or weights from the table. Do not select competition submissions from this table.
- Prior Phase15 official/outer mismatch and domain-separability [counterexample](../domain-separability/RESULT.md) weakens simplistic source-style explanations. Signal identification and validation mismatch are both open hypotheses.

## Decision (research PIC)

1. **No PPO** for single-step 18-label classification. Direct supervised CE/ranking already supplies known-label signal.
2. **No immediate 19-feature rule patch**; current Best is a multi-representation attributed public reproduction, not an unsupervised blank slate.
3. Reopen *representation-level sibling learning* only after genuine source/protocol provenance and independent control C0 vs matched-prompt C1 vs shuffled Cshuf on **same-row externally held out responses**, per existing [STABLEPAIR contract](stablepair18-design-best-comparison-20261107.md).
4. Use this audit to **falsify overly general claims**, not to tune. If independent data is unavailable, report blocked scientifically and consider clearly labeled exploration only where user authorizes a bounded test and no hidden-label tuning.
5. Existing pending competition refs MGT `58233227`, WeMM `58244483`, Lightning `58167751` require a **new authenticated Kaggle ledger** before any numerical update. A read-only one-shot status workflow has been committed separately. The last already verified champion is `58210835` Public BA **0.679**.

**Execution cost:** no GPU, no new supervised model, no Kaggle competition submission. This result narrows the mechanism search; it is not a leaderboard breakthrough.
