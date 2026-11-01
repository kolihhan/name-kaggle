# NOW

Updated: 2026-11-01

- **Current competition champion:** Phase 7 gated specialist.
- **Local OOF balanced accuracy:** `0.3700521998508576`.
- **Current Public LB:** `0.369`.
- **Submission ref:** `57838845`.
- **Most important rejected experiment:** Phase 15 terms-clean public sparse — local OOF `0.4053218990802883`, Public `0.357`, submission `57861337`.
- **Validation diagnosis:** Phase 19 reproduced a substantial Phase15 − Phase7 gain on all three frozen alternate prompt-group assignments (mean delta `+0.0398434`), so one permanent fold mapping is not sufficient to explain the collapse. Those three shadow assignments are now spent diagnostic folds, not unseen future holdouts.
- **Transfer diagnosis:** Phase 18 found Phase 15 external text materially closer to official responses than the Phase 7 control; the model-controlled domain audit then found Phase 15 external text *less* separable from official text than Phase 7 (lexical domain BA `0.5811` vs `0.7553`). The simple “obvious external-source fingerprint” explanation is therefore rejected. The leading inference is official-distribution overmatching / hidden-distribution mismatch.
- **Most important unresolved question:** the exact hidden-distribution dimension causing the transfer failure remains unknown; Public LB is only a partial hidden-test anchor.
- **Next authorized work:** freeze the future candidate-evaluation policy first, then brainstorm one genuinely different modeling direction. Do not open another modeling phase, retune Phase 15, reuse Phase 19 shadows as unseen holdouts, or make another Kaggle submission yet.
