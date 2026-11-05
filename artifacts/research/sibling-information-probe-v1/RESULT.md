# Sibling Information Probe v1 — Execution Result (2026-11-05 MYT)

## STATUS: ENGINEERING PASS / SCIENTIFIC DATA BLOCKED

**No real model-accuracy result**. This is neither a scientific PASS nor FAIL.

Research branch: research/sibling-information-probe-20261105

### Implemented

- Pre-result contract: CONTRACT.md, with source/prompt integrity, exact representation, primary balanced-accuracy metric, fixed conditional-head comparison, negative control, per-source gates, CI and safety veto.
- Reusable CPU evaluator: scripts/run_sibling_information_probe.py. Fits flat 18-way and fixed family-conditional LogisticRegression on identical already-extracted frozen LoRA embeddings; maintains each family's total predicted probability mass. Includes a within-family shuffled-label negative control, prompt-group bootstrap and source-aggregated metrics. Reports same-family and cross-family errors and explicitly counts top-1 family switches.
- Synthetic-only tests: tests/test_sibling_information_probe.py. Verify all 18 labels/six families, family-mass invariants, provenance/source/prompt admission, checksum / no-pickle policy, complete evaluator run and the counterexample that preserving family mass need not preserve top-1 family.
- Dedicated data-safe GitHub Actions: .github/workflows/ntl-sibling-probe-cpu-tests.yml. Sparse-checkout asserts official data absent, uses CPU only, no Kaggle tokens, no training, no submissions.
- Candidate source lead audit in SOURCE_ADMISSION.md.

### Verified runtime evidence

GitHub Actions **42059420744**, commit **20eb7b2098b87d7e66f5171b348d77c0427959d4**: SUCCESS. **6 synthetic tests passed**, code compiled. Prior pipeline **42058829399**: 5 tests passed before counterexample test was added. The 15 pytest warnings are from sklearn/SciPy L-BFGS deprecation, not observed test failures.

Note: this is synthetic-only execution. It supplies no estimate of OOF/Public/Private transfer.

### Scientific blocker

No source-hashed, private **frozen LoRA embedding** matrices from two genuinely new independent, audited external sources exist in verified accessible evidence. Previously used Gate-A aggregates do not contain row-level heldout embeddings and are already spent. Newly discovered public pool aggregates may be re-packaged previously used examples; source name alone is not a reliable holdout. Some API aliases (notably Google 3.7 Flash after 2026-10-08) can route to a different served model.

Therefore: do not run the evaluator with reused official folds or pretend synthetic data measure real improvement. Before expensive GPU inference, verify candidate raw corpus provenance and prompt/source overlap, then freeze exact source and feature hashes.

### Next scientific action

1. Audit the underlying 18-label source lineages from olegzholobov or the 255k pool (not whole merged aggregates), including served-model metadata and overlaps.
2. Select and lock two genuinely unseen source lineages covering >=3 multi-sibling families, using the minimum 20-per-label test threshold.
3. Extract fixed Stage-1 A1 Qwen3-8B LoRA24 embeddings once to a private source, then run the predeclared CPU evaluation; issue PASS/FAIL without threshold tuning.

**No new model training. No formal Kaggle competition submission. No official data or private embeddings added to public GitHub.**
