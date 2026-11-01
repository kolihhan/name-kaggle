# NOW

Updated: 2026-11-01

- **Current competition champion:** Phase 7 gated specialist.
- **Local OOF balanced accuracy:** `0.3700521998508576`.
- **Current Public LB:** `0.369`.
- **Submission ref:** `57838845`.
- **Most important rejected experiment:** Phase 15 terms-clean public sparse — local OOF `0.4053218990802883`, Public `0.357`, submission `57861337`. Phase 19 confirmed its local advantage over Phase 7 on all three frozen shadow prompt-group assignments, so one-split luck is not a sufficient explanation for the transfer failure.
- **Most important unresolved question:** which external-corpus / hidden-distribution differences make Phase 15 generalize across official held-out prompt groups but fail on the Public hidden distribution?
- **Next authorized experiment:** audit-only domain-separability diagnostic: official responses vs frozen Phase 15 external text, with Phase 7 external text as the known-transfer control and model identity controlled. No model retuning and no Kaggle submission.
