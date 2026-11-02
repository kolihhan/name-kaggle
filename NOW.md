# NOW

Updated: 2026-11-02

- **Current competition champion:** Phase 7 gated specialist.
- **Local OOF balanced accuracy:** `0.3700521998508576`.
- **Current Public LB:** `0.369`.
- **Submission ref:** `57838845`.
- **Most important rejected experiment:** Phase 15 terms-clean public sparse — local OOF `0.4053218990802883`, Public `0.357`, submission `57861337`.
- **Validation diagnosis:** Phase 19 reproduced a substantial Phase15 − Phase7 gain on all three frozen alternate prompt-group assignments (mean delta `+0.0398434`), so one permanent fold mapping is not sufficient to explain the collapse. Those three shadow assignments are now spent diagnostic folds, not unseen future holdouts.
- **Transfer diagnosis:** Phase 18 found Phase 15 external text materially closer to official responses than the Phase 7 control; the model-controlled domain audit then found Phase 15 external text *less* separable from official text than Phase 7 (lexical domain BA `0.5811` vs `0.7553`). The simple “obvious external-source fingerprint” explanation is therefore rejected. The leading inference is official-distribution overmatching / hidden-distribution mismatch.
- **Evaluation policy:** `EVALUATION_POLICY.md` is frozen. Permanent grouped OOF remains required for comparability but is not a sole promotion oracle; Public LB cannot be used as a tuning loop.
- **Closed candidate — prompt-residual-style-v1:** Gate A failed before official OOF; external held-out projection delta `-0.0357910434`, worse on `11 / 13` sources. Exact evidence: `artifacts/candidates/prompt-residual-style-v1/RESULT.md`.
- **Candidate — two-stage-lora24-v1: Gate A PASS.** Exact result: `artifacts/candidates/two-stage-lora24-v1/RESULT.md`.
- **Gate A pooled BA:** A0 `0.3942430082` → A1 `0.4657129266`, delta `+0.0714699184`.
- **Gate A source deltas:** `abdullah +0.0806537702`, `semenoff +0.0664289879`, `zakir +0.1080631523`; mean `+0.0850486368`; negative sources `0 / 3`.
- **Gate A leakage / invariants:** 13 eligible labels, 3,324 held-out rows, `prompt_id_overlap=0`, `prompt_key_overlap=0`, `exact_one_epoch=true`, `triplet_padding=false`, `stock_qwen_final_norm_retained=true`.
- **Model identity:** `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, `transformers==4.57.1`, `peft==0.17.1`.
- **Gate-A scientific lineage:** trigger commit `6a7168e166a719889a7f6ad77ed818a5dfee75d7`; original Actions run `41309458267`. Its Actions poll timed out while Kaggle still reported `RUNNING`; this was orchestration timeout, not scientific FAIL. Attach-only recovery run `41333631967` observed the same kernel `COMPLETE`; no second kernel push occurred.
- **Preserved Gate-A evidence:** artifact `13094213203`, digest `sha256:9374e77a0eb59956484dc9bb052fcc55ccea4d95e14e62976ba23c4ba97fcf8e`.
- **Exact A1 checkpoint:** artifact `13094069155`, size `116273873` bytes, digest `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`. A retrained equivalent is forbidden.
- **Gate-B plan / contract:** plan `docs/superpowers/plans/2026-11-02-two-stage-lora24-gate-b.md`, plan commit `b39bb4b721441f016ecd18f8d6eb2deba6fc3011`; scientific source of truth remains `artifacts/candidates/two-stage-lora24-v1/CONTRACT.md`.
- **Gate-B Task 1 complete:** pure checkpoint/fold/bootstrap/decision helpers implemented fail-closed.
- **Gate-B Task 2 complete — exact checkpoint transport locked:** private dataset `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`, status `ready`, version `1`. Exact lock: `artifacts/candidates/two-stage-lora24-v1/GATE_B_CHECKPOINT_LOCK.md`.
- **Checkpoint byte identity:** exact 12-file manifest `sha256:c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`; deterministic transport SHA256 `8f3c6dabe6a3d910b80b6acafce6155ae6350dafb28b1b987f7cdb7eff7bf32a`. Attach-only verification run `41374175606`; evidence artifact `13110507634`, digest `sha256:ed0666b1d214299d1194782d021fb476a6535ebd4ed0bb0cecfda30b774e62d0`.
- **Gate-B Tasks 3–4 complete:** `scripts/run_two_stage_lora24_gate_b_gpu.py` implements fresh-per-fold Stage-2 workers and five-fold orchestration; `.github/workflows/two-stage-lora24-gate-b.yml` only launches the heavy private Kaggle job when a commit message contains `[gate-b-run]`. No competition source or submit path exists.
- **Gate-B Task 5 complete:** `scripts/run_two_stage_lora24_gate_b.py` regenerates and validates row-level Phase-7 OOF, computes plain-head Gate-B diagnostics, centered-head standalone diagnostics, paired prompt bootstrap, and applies the frozen four-condition decision mechanically.
- **Real Phase-7 preflight PASS:** Actions run `41378723918` regenerated all `2699` Phase-7 OOF rows from the frozen sparse pipeline. BA `0.3700521998508576`; fold BAs `[0.38333333333333336, 0.33888888888888885, 0.37222222222222223, 0.38703703703703707, 0.36883780332056193]`; overall/folds/per-class recalls matched the frozen report within `1e-12`. Evidence artifact `13113150078`, digest `sha256:bdd25f43b202c6bfbbbd74a99fa8dbbf815c56071d679c253c2b88f4d4171079`.
- **Final implementation verification PASS:** exact implementation source commit `eff43d97a1339f1deeea667b90b45269133dffbd`; full-suite Actions run `41379434008`; exact result `204 passed in 233.47s (0:03:53)`.
- **Scientific drift review PASS:** no drift from the frozen contract in backbone/revision, 24-layer truncation, LoRA config, pooled layers, 447-token truncation, two Stage-2 epochs, `5e-5` / `5e-4` LRs, permanent fold frame/fingerprint, seed `20261101`, fit-only class weights, plain primary head, 5,000 prompt bootstrap, or four Gate-B PASS conditions.
- **Gate B is now implementation-ready and one-shot authorized, but has NOT executed yet.** Authorization: `artifacts/candidates/two-stage-lora24-v1/GATE_B_RUN_AUTHORIZATION.md`, authorization commit `db9406819b2394cf988a1bf93fd6060f946132da`.
- **One-shot boundary:** exactly one valid official Gate-B OOF result may be consumed. Do not rerun because the score is disappointing. Infra-only retry is allowed only if no valid scientific OOF was produced and no scientific setting changes.
- **Competition boundaries remain intact:** `public_lb_tuning=false`; no competition-test inference; no Kaggle competition submission; Phase19 shadows remain spent; centered head remains diagnostic only; Phase 7 remains champion until the frozen Gate-B decision says otherwise.
- **Next authorized work:** verify the authorization commit's lightweight Gate-B contract job passes with `kaggle-gpu` skipped. Then make a trigger-only `[gate-b-run]` commit with no scientific code change and execute the one authorized Gate-B OOF run.
