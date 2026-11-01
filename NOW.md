# NOW

Updated: 2026-11-01

- **Current competition champion:** Phase 7 gated specialist.
- **Local OOF balanced accuracy:** `0.3700521998508576`.
- **Current Public LB:** `0.369`.
- **Submission ref:** `57838845`.
- **Most important rejected experiment:** Phase 15 terms-clean public sparse — local OOF `0.4053218990802883`, Public `0.357`, submission `57861337`.
- **Validation diagnosis:** Phase 19 reproduced a substantial Phase15 − Phase7 gain on all three frozen alternate prompt-group assignments (mean delta `+0.0398434`), so one permanent fold mapping is not sufficient to explain the collapse. Those three shadow assignments are now spent diagnostic folds, not unseen future holdouts.
- **Transfer diagnosis:** Phase 18 found Phase 15 external text materially closer to official responses than the Phase 7 control; the model-controlled domain audit then found Phase 15 external text *less* separable from official text than Phase 7 (lexical domain BA `0.5811` vs `0.7553`). The simple “obvious external-source fingerprint” explanation is therefore rejected. The leading inference is official-distribution overmatching / hidden-distribution mismatch.
- **Evaluation policy:** `EVALUATION_POLICY.md` is frozen for future candidates. Permanent grouped OOF remains required for comparability but is not a sole promotion oracle; new candidates require a predeclared transfer/robustness check beyond re-splitting the same official rows, and Public LB cannot be used as a tuning loop.
- **Closed candidate — prompt-residual-style-v1:** Gate A failed before any official OOF. Across 13 eligible held-out external sources, raw frozen ModernBERT mean BA was `0.4034268654` versus `0.3676358221` after the prompt-residual LDA projection; mean delta `-0.0357910434`, with the projection worse on `11 / 13` sources. `official_oof_authorized=false`; no submission. Exact evidence: `artifacts/candidates/prompt-residual-style-v1/RESULT.md`.
- **Interpretation of that failure:** the specific prompt-centering + supervised LDA mechanism removes transferable attribution signal more often than it isolates it. This does not reject all neural/style representations; it closes only this frozen mechanism.
- **Most important unresolved question:** the exact hidden-distribution dimension causing the Phase 15 transfer failure remains unknown; Public LB is only a partial hidden-test anchor.
- **Next authorized work:** return to brainstorming and freeze a **genuinely different** complementary candidate before implementation. Do not run official OOF for `prompt-residual-style-v1`, retune its projection/source rules, rescue favorable sources, retune Phase 15, reuse Phase 19 shadows as unseen holdouts, mint new splits for reassurance, or make another Kaggle submission yet.
