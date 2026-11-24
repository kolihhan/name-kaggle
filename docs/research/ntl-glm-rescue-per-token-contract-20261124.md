# NTL one-shot candidate: Expected GLM rescue per estimated token

Date: 2026-11-24. Owner explicitly requested direct execution, no scheduler, and one technically justified exploratory competition submission if preflight passes.

## Frozen sources and hypothesis
- Account reference: externally reproduced Public BA 0.715, #58680531. Preserve it. No claim of our own original model or independent private score.
- Parent source: public `dennis6886/glm-user-prefix-guarded-fable-return`, actual fetched code SHA256 `f36faa76126819e799b9a70b79b76c4c21f3fc7498e9e60d5106a8c654eceaa9`, verified [exact source contract #45315527607](https://github.com/kolihhan/name-kaggle/actions/runs/45315527607). It is a newer source than the Nov 23 audited revision; its own official score is unknown. Attribution belongs to Dennis and upstream authors.
- Source evidence: original prefix_variant_stage.py computes mass = P(GLM-5.3 or Flash), sorts descending mass, then estimates token rows and truncates to the cumulative budget. It does not rank by avoidable error or compute cost. Existing two-pass GLM likelihood, prefix, protected Fable return, shrink, max cover, token budget, and original safety guards remain intact.
- Hypothesis: a specialist should prioritize potentially incorrect GLM-family predictions, not confident GLM predictions that would already be correct. At the same cost/deadline, ranking by optimistic GLM rescue opportunity per estimated token may increase useful coverage. This is an inference-allocation experiment, not a new trained backbone or a proven accuracy improvement.

## Single change
Given base probabilities p_i = softmax(z_i), G = the original select_classes, c_i = original estimated token rows, and b_i = argmax(p_i):

    r_i = sum_{j in G} p_ij - 1[b_i in G] * p_i,b_i
    priority_i = max(0, r_i) / max(1, c_i)

r_i is the idealized probability mass of true GLM-family labels that a perfect specialist could rescue from the base top-1 decision, assuming no regressions. It is NOT a calibrated expected gain from the actual GLM member. Use deterministic stable ordering (descending priority, ties descending original mass, then original row order).
- Do not modify any likelihood feature, GLM model, weights, calibration, blending, input truncation, cover fraction, timeout, scheduler, or downstream postprocessing.
- Parent source is not the exact scored 0.715 source; comparison to account 0.715 is a competition utility test, not clean causal attribution of any score delta to ranking.
- Known risk: the new order can discard cases where likelihood correction would help despite high confidence, or over-prioritize short difficult passages. The proxy assumes idealized specialist and base probabilities may be miscalibrated. No honest same-row full18 labeled parent/candidate test available; this is explicitly an exploratory one-shot bet, not scientific promotion.

## Operational gate and stop
1. SHA-pinned public parent; exactly one modified existing stage cell; exact source anchors, compile, deterministic isolated numerical tests; all original other cells/assets, privacy, GPU and deadline unchanged.
2. Unique private destination `kolihhan/ntl-glm-rescue-per-token-20261124`; no repeated push. Preview must COMPLETE, valid submission.parquet schema, 18 legal model labels, nonempty unique IDs. Preview can be a 3-row fast path and is NOT full hidden inference or BA validation.
3. Verify actual owned 0.715 ledger receipt and current submission quota; do not submit if active official submission exists or fewer than two slots are currently available. Use exactly one code submission, exact private Notebook version 1, unique ledger description `ntl-glm-rescue-per-token-once-20261124`. No duplicate or automatic retries after an uncertain POST.
4. Hard time cap is the parent's original 8.8 h V1 deadline and existing budget margins. No claim of measured hidden runtime. Run only within existing free Kaggle GPU allocation. Bounded one-shot Actions polling only; no scheduler.
5. Record official receipt and score when available. If error, timeout, or <=0.715, retire this frozen variant without Public-LB retuning. If >0.715, report the score and provenance, not independent mechanism proof; preserve original winner.
6. If any gate fails or preview remains queued at bounded polling limit, stop and report blocker; do not manufacture a score, auto-repush, spend paid resources, or run duplicate jobs.

No competition texts, credentials, row-level predictions or third-party notebook source persisted in GitHub. Only our patch builder and aggregate tests.
