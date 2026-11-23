# Provider-shift paired gate — preregistered research contract (2026-11-24)

Hypothesis: a candidate that appears to improve on official TRAIN may lose its advantage on genuinely independent provider/version generation protocols. Test on the **same real rows** with strong incumbent and candidate predictions; evaluate 18-class balanced accuracy, per-class recall, paired rescues/regressions, and source-stratified results.

## Inputs and exclusions
- CSV (private, never commit): `prompt_id,label,source,baseline_pred,candidate_pred`. Source must describe verified provider, served model version, sampling and generation protocol; names alone are not proof.
- Strong scorer must be executed on the exact same rows as candidate. No synthetic examples, Kaggle test labels, 3-row previews, pseudo-labels, or previously exhausted 12-class panels.
- GroupKFold by prompt for any fitted component; no leakage across prompt groups. Script is read-only diagnostic, not a fitted model.
- Minimum: complete 18 classes overall, at least two truly independent sources. Each source lacking full 18 coverage is explicitly marked incomplete and cannot support per-source BA.
- Compare against 0.715 external V5 only when identical executable scorer is available. Do not compare unrelated local BA to official Public BA.

## Precommitted decision
Advance only if the complete 18-class same-row candidate BA exceeds baseline, net paired rescues are positive, worst-class recall does not deteriorate materially, and improvement persists in at least two independently verified source/version panels. Report uncertainty using prompt-cluster bootstrap before promotion. If any requirement fails or provenance is unclear: no submission; document which gate failed. One-shot submission under doctrine §11 is a separate explicit preregistered bet and must not be inferred from this gate.

## Cost/stop
CPU-only script; no training or GPU; stop after one pass per frozen candidate/panel. This contract does **not** authorize new Kaggle submission. Current champion and existing notebooks remain untouched.

## Source-derived motivation
SemEval 2024 Task 8 AISPACE first-place work used confusion-oriented classifier analysis, but its six sources and large training sample do not transfer directly to NTL 18 classes / 2699 rows. DAIGT winning data protocols highlight source diversity, but its binary target differs. This gate operationalizes the **protocol-shift test**, not their architecture or score. The previous NTL SemEval transfer test found only 2/5 stable cross-source fingerprints; do not reuse those panels as fresh OOD.
