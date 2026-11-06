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

## Executed data-source audit addendum — 2026-11-06 MYT

**Status: REAL SOURCE / OVERLAP VERIFICATION COMPLETED; SCIENTIFIC DATA GATE STILL BLOCKED.** Previously synthetic test CI remains 6/6 PASS. This update does not represent a new model score.

- Retrieved public external dataset file inventories and primary README generation protocols in successful GitHub Actions 42063920056, 42065056290, 42065237791, and 42065527178. maniceet dataset returned a genuine resource-specific 403.
- Verified current crtproduction 212,623-row pool, 60 upstream catalog entries (not the stale 255k listing). Full source-catalog and two-Parquet audits succeeded in Actions 42064170317 and 42064722943. Exactly **61,681 of its 212,623 normalized response texts** overlap the pre-existing frozen Stage-1 75,783-row public pool. This is *text* overlap, not a source/prompt independence proof.
- Independently downloaded the original public source files (Actions 42065925337). alessandrolair original: 3,866 rows / 12 labels / 3 multi-label families, SHA256 e98b1924269c30116f50623052f7fc4ea6148f7575d2fe483c9bd899b4ea2e21, zero exact normalized Stage-1 overlap; per-row Kaggle Benchmarks proxy slug but no independently verified served backend response. elliotthankinson pool_v8j_add original: 8,347 rows / 7 labels, SHA256 e8aeafcd75b9832c297324da360051a48fec3b58d1d8cfe251e03c2558f40185, zero exact normalized Stage-1 response or exact prompt_key overlap; **only two qualifying sibling families** (GLM and GPT), not three. Other batches require separate provenance audit before 3-family admission.
- trnhip original README explicitly admits served-model ID was not recorded; aswathvanan reuses other public prompt banks. Neither qualifies as an independent full strict holdout merely because of zero exact response overlap.
- Potential independent source pair: alessandrolair plus correctly audited sub-batches of elliotthankinson, with unknown cross-source prompt equivalence, API served-model validation, and Stage-1 *training-source* independence at prompt-bank level. No final two-source lock exists.
- **No Stage-1 A1 LoRA embedding extraction, readout fitting, statistical assessment, additional GPU training, or Kaggle competition submission happened.** Real BA delta remains UNKNOWN. The readout evaluator must not be run on fake or spent holdout material.

Canonical details / hashes / source evidence: SOURCE_ADMISSION.md. The next necessary scientific dependency is a defensible pair of fully source/prompt-disjoint, serving-ID-traceable corpora and private frozen checkpoint embeddings; only then run the previously fixed paired experiment. Do not loosen the preregistered three-family/source conditions just to obtain a result.

## 2026-11-06 follow-up: real-data source-prompt join

GitHub Actions 42151858490 SUCCESS: source-prompt linkage detected 522 reused prompt identities, contaminating 621 of 2,162 zeminglii candidate rows relative to the **full** Stage-1 public pool; 1,541 rows remain without detected overlap, not certified independent. Model response metadata 42151491857 verified model_seen for 2,162/2,162 with exact expected model label as substring. The frozen actual 30,133-row A1 training subset's prompt overlap is not separately determined. **No scientific readout PASS/FAIL, no actual model inference, no training or submission.** See SOURCE_ADMISSION.md for detailed counts and caveats.

## Exact A1 training split follow-up — 2026-11-06

GitHub Actions [42152215480](https://github.com/kolihhan/name-kaggle/actions/runs/42152215480) **SUCCESS**. Exact frozen A1 30,133 train rows reconstructed; 18,406 could be mapped to open-pool prompt_canon, 11,727 could not. **125 zeminglii rows confirmed to reuse A1 training prompts**; 621 confirmed against broad 75,783-row source pool. Thus the candidate cannot be admitted unchanged, and the remaining rows lack full prompt-independence proof. Scientific scoring remains blocked; no GPU, LoRA inference, model fitting, or competition submission.
