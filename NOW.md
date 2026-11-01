# NOW

Updated: 2026-11-02

- **Current competition champion:** Phase 7 gated specialist.
- **Local OOF balanced accuracy:** `0.3700521998508576`.
- **Current Public LB:** `0.369`.
- **Submission ref:** `57838845`.
- **Most important rejected experiment:** Phase 15 terms-clean public sparse — local OOF `0.4053218990802883`, Public `0.357`, submission `57861337`.
- **Validation diagnosis:** Phase 19 reproduced a substantial Phase15 − Phase7 gain on all three frozen alternate prompt-group assignments (mean delta `+0.0398434`), so one permanent fold mapping is not sufficient to explain the collapse. Those three shadow assignments are now spent diagnostic folds, not unseen future holdouts.
- **Transfer diagnosis:** Phase 18 found Phase 15 external text materially closer to official responses than the Phase 7 control; the model-controlled domain audit then found Phase 15 external text *less* separable from official text than Phase 7 (lexical domain BA `0.5811` vs `0.7553`). The simple “obvious external-source fingerprint” explanation is therefore rejected. The leading inference is official-distribution overmatching / hidden-distribution mismatch.
- **Evaluation policy:** `EVALUATION_POLICY.md` is frozen for future candidates. Permanent grouped OOF remains required for comparability but is not a sole promotion oracle; new candidates require a predeclared transfer/robustness check beyond re-splitting the same official rows, and Public LB cannot be used as a tuning loop.
- **Closed candidate — prompt-residual-style-v1:** Gate A failed before any official OOF. Across 13 eligible held-out external sources, raw frozen ModernBERT mean BA was `0.4034268654` versus `0.3676358221` after the prompt-residual LDA projection; mean delta `-0.0357910434`, with the projection worse on `11 / 13` sources. `official_oof_authorized=false`; no submission. Exact evidence: `artifacts/candidates/prompt-residual-style-v1/RESULT.md`.
- **Public-method audit:** current strong public Kaggle work supports a materially different mechanism: Qwen3 LoRA24 with multi-depth attribution features, same-prompt triplet training, and separate plain/centered heads. Public final Stage-2 checkpoints are **not valid for honest repo OOF** because final training sees all official rows. Stable audit: `artifacts/research/public-two-stage-lora-audit-2026-11-01.md`.
- **Candidate — two-stage-lora24-v1: Gate A PASS.** Frozen external-only Gate A compared Qwen3-8B LoRA adaptation (A1) against the same frozen backbone with trainable normalization/readout only (A0), using the provenance-audited Phase14 public subset and deterministic 3-source holdout. Exact result: `artifacts/candidates/two-stage-lora24-v1/RESULT.md`.
- **Gate A pooled BA:** A0 `0.3942430082` → A1 `0.4657129266`, delta `+0.0714699184`.
- **Gate A source deltas:** `abdullah +0.0806537702`, `semenoff +0.0664289879`, `zakir +0.1080631523`; mean `+0.0850486368`; negative sources `0 / 3`.
- **Gate A eligibility / leakage:** 13 eligible labels, 3,324 eligible held-out rows, `prompt_id_overlap=0`, `prompt_key_overlap=0`. Frozen decision: `pass`; `official_oof_authorized=true`.
- **Implementation invariants verified from the recovered result JSON:** `exact_one_epoch=true`, `triplet_padding=false`, `stock_qwen_final_norm_retained=true`. Model remains pinned to `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, `transformers==4.57.1`, `peft==0.17.1`.
- **Scientific execution lineage:** exactly one Kaggle scientific execution was launched by trigger commit `6a7168e166a719889a7f6ad77ed818a5dfee75d7`, original Actions run `41309458267`. That Actions run exhausted its polling window while Kaggle still reported `RUNNING`; this was an orchestration timeout, not a scientific FAIL. No failed-job rerun was used.
- **Attach-only recovery:** commit `62b015a90c6b44bf880e6802d43b162fdca77ce7`, Actions run `41333631967`. The recovery job performed no `kaggle kernels push`; it attached to the existing kernel, observed `COMPLETE`, downloaded and verified the existing output, and preserved the original A1 checkpoint.
- **Preserved Gate A evidence:** artifact `13094213203`, digest `sha256:9374e77a0eb59956484dc9bb052fcc55ccea4d95e14e62976ba23c4ba97fcf8e`.
- **Exact A1 checkpoint for Gate B only:** artifact `13094069155`, size `116273873` bytes, digest `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`. A retrained equivalent is not an acceptable substitute.
- **Competition boundaries remain intact:** `official_rows_scored=0`, `official_oof_runs=0`, `competition_test_rows_used=0`, `public_lb_tuning=false`, `kaggle_submission_authorized=false`. No new competition submission was created.
- **Interpretation:** Gate A supports the narrow external-transfer hypothesis and authorizes progression to Gate B; it does **not** prove hidden-distribution transfer and does not replace Phase 7 as the competition champion.
- **Still forbidden:** automatic Gate-B execution, any Stage-2 run not initialized from the exact preserved A1 checkpoint, Phase19 shadow-fold reuse, Public-LB tuning, or a new Kaggle competition submission.
- **Next authorized work:** write and review a separate Gate-B implementation plan under the frozen evaluation policy, using only the exact preserved A1 checkpoint lineage. Do not execute Gate B until that plan/implementation is separately ready and verified.
