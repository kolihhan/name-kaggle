# CDA-L24 — Conditional Domain-Adversarial LoRA24

Date: 2026-11-08 MYT. Status: ARCHITECTURE FROZEN, DATA CAPACITY SCREEN COMPLETE, NO MODEL FIT OR SUBMISSION.

## Purpose and scientific hypothesis

Current reproduced external Best Public BA is 0.668 (Ozertuu). Own independently trained LoRA24 scored 0.520 Public / 0.546761 grouped official OOF. Its source-transfer adapted-vs-frozen gain is +0.071470 BA; task-trained representations help. The +0.148 Public gap between whole pipelines is not an isolated architecture effect.

H1: class-conditional gradient reversal can teach an already task-adapted LoRA24 backbone to discard text-collection/source artifacts while retaining generator-model identity. In an actually disjoint source cohort, the same-head same-compute treatment should beat the no-reversal control. It is a model-learning objective, NOT source weights, SVC tweaking, frozen author geometry, a new family gate, or a second 8B model at test. Domain-adversarial learning is established prior art: the novelty claim is only a specific falsifiable application, not invention of gradient reversal.

H0: the corpus confounds source/protocol with model label or useful identity itself; source-adversarial gradients hurt or do not improve honest source transfer. A positive official OOF alone cannot validate the mechanism.

Rejected/parked directions remain rejected/parked: TF-IDF source balancing befcb1c; LUSR/DeTeCtive frozen geometry; prompt-residual LDA; Short-TRACE standalone; post-hoc V47 weight tweaks; conditional-denoising18-v1 automatic next run.

## Immutable source cohort, selected BEFORE model scoring

New open pool: Kaggle crtproduction/name-that-llm-open-pool, main ntl_open_pool.parquet SHA256 1e4200aed80f52d655d4900a301cb427ca1341e06be31a8b7d5fde9ea3ef425f, 212,623 rows, 18 exact labels, 60 source_dataset values. Read-only CPU screening Actions 42477146980. Prior Stage-1 pool SHA256 37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec, with 30,133 original Stage-1 training rows across 14 source strings. The audit found 11 capacity-eligible source IDs after excluding explicit publisher risk flags, missing/unknown license, known training/official normalized-answer matches, and known prompt-key matches.

Frozen heldout selection by predeclared SHA salt: arxitekton/ntl-claude-gen-v1, 12,463 risk-filtered rows, 16 labels with >=20 examples, 1,057 publisher prompt groups and zero exact normalized answer overlap with our original Stage-1 text and official training text. These facts are NUMERIC CAPACITY ONLY. Publisher license strings are not independently verified source terms; source IDs from differently named pools can refer to the same underlying generator corpus or prompts. The merged pool lacks exact served model revisions, request protocol, temperature and timestamps. Therefore no independent transfer or causal protocol robustness is proven yet.

Potential other capacity-eligible training source: drjcfmd/ntl-generated-v2, 20,400 rows with 17 supported labels and 1,377 prompt groups, no exact-answer overlap in the audit. Other training sources must be selected by frozen capacity/provenance constraints only, never by score. Exclude arxitekton from EVERY stage of fit, source statistics, class weights, augmentation and checkpoint selection. Audit the original source licenses and aliases; fail closed on ambiguous origins. Keep original raw rows and learned artifacts private/off the public Git repo.

## Hard data gate BEFORE any GPU experiment

1. Obtain and independently validate original dataset license, model/version mapping and source corpus relationships for both training and heldout cohorts; only use legitimately permitted public data and respect publisher output terms.
2. Verify complete source-origin, answer and prompt-bank isolation from the reserved heldout, the original LoRA Stage-1 corpus, official training and all selected training sources.
3. At least 8 exact model labels must occur in 2 or more genuinely distinct training sources, each with >=20 clean rows; otherwise conditional source adversary is not identifiable.
4. Source holdout must retain >=8 eligible models and >=300 rows after all alias/prompt/key exclusions.
5. Confirm accelerator training can be completed in a genuinely available authorized run and produce private model artifacts. No local GPU or paid API. If unavailable: BLOCKED BY RESOURCES, not rejected-science.
6. Only ONE frozen training/evaluation sequence after these conditions pass. Do not weaken the gate to obtain a submission.

## Full model architecture and fixed arms

Start from the exact existing Stage-1 A1 checkpoint of Qwen3-8B-Base / 24-layer LoRA24, rank16 alpha32, pooled layers 2/4/6/8/12/16/20/24 and an 18-way plain model head. No change to 18 class names, input tokenizer or hidden model at inference.

During EXTERNAL-only continued source training, attach a TRAIN-ONLY source classifier: representation z (32768 dims) -> LayerNorm -> Linear(32768,256) -> GELU, concatenate 32-dim learned embedding of known training MODEL LABEL, then Linear(288,n_sources). Apply gradient reversal with fixed strength lambda=0.10 to z. Train model CE plus class-conditional source CE. Class label is supplied to the discriminator only during training, never on hidden test. Only prequalified class/source support cells participate in source loss; all labels still participate in model CE.

Control C0: identical model, data, loss terms, domain head architecture, optimizer, original Stage-1 checkpoint, number of steps and seed, but lambda=0. Source head learns source, yet its gradient does not reach identity representation. Treatment C1: lambda=0.10; nothing else changes. Train one frozen epoch external; then remove source head. Perform official Stage-2 adaptation for 2 epochs fold-local with historical training recipe. Final inference stays one 8B LoRA24 pass and plain 18-class head, no test-time source IDs, gates or extra language model.

The 2026 source-balanced TF-IDF failure tested weighting without changing representation; this tests adversarial feature learning. The earlier prompt-invariant contrastive v2 was a different learning mechanism; do not reuse or retune it.

## Evidence before final submission

Primary independent check: C0 versus C1 on SAME risk-filtered verified-heldout source rows (16 supported labels); require exact-class BA(C1)-BA(C0)>=+0.020, 5000 prompt-group bootstrap replicates with 95% interval lower>0, no provider recall fall >0.010, all per-model recalls published in aggregate. No parameter, source or threshold changes on this result. Sixteen-label external BA is not eighteen-label Kaggle BA.

Secondary historical coordinate: official permanent 5-fold GroupKFold(prompt_id), exact all 2699 rows/18 labels, fold-local Stage-2. Report matched LoRA24 hard OOF 0.546761, require C1 >=0.556761, paired prompt-group bootstrap interval lower>0, label recall degradation <=0.10 and provider BA degradation <=0.010. Official folds have been reused and cannot alone justify promotion.

When BOTH data/source and model criteria pass: final fit on all official train, build real internet-disabled Kaggle Code Competition Notebook; verify full hidden size ~4000, under 9-hour runtime, all 18 labels, output columns id/model in submission.parquet. Submit ONE exact frozen Notebook version; report the accepted Submission ID, terminal status and Public BA, and compare with reproduced 0.668. No Public LB parameter loop. Lack of external strong-model row-aligned OOF is NOT a universal blocker for this independent candidate; do not pretend a paired result against Ozertuu is available.

If any gate fails, report the actual evidence and STOP this specific training objective rather than inventing a new source/feature weight adjustment. Restriction: no local GPU/paid API, no model training before original-source lineage clearance. Pivot only if genuinely new independent evidence changes the representation/data bottleneck.


## Implementation checkpoint — 2026-11-08, no model score

**Engineering DONE:** standalone train-only [conditional source GRL code](../../scripts/conditional_domain_adversarial_lora24.py) with actual model-label-conditioned source head and full-training cell-support mask; [5 synthetic tests](../../tests/test_conditional_domain_adversarial_lora24.py) prove that lambda=0 and lambda=0.10 produce the same forward source predictions, but source-feature gradients differ, and unsupported source/model combinations are excluded. [GitHub Actions 42478060946](https://github.com/kolihhan/name-kaggle/actions/runs/42478060946) **SUCCESS**. This proves mechanism code, NOT attribution accuracy.

**Upstream origin manifest check incomplete:** same Action could fetch the two named Kaggle upstream dataset metadata templates, but returned no usable original title/license/details; neither exact serving model nor sampling protocol is verified. The merged pool's four license strings are publisher-reported only. Therefore original-source authorization/provenance **FAIL-CLOSED / INCOMPLETE**, not proof of disallowed use. No GPU science run or Notebook submission was attempted.

**Still required before an end-to-end candidate:** obtain original provenance/source-alias and exact model-label generation metadata; validate crossed class-source support and data license terms; fork the Stage-1 GPU training runner from the immutable LoRA24 reference, integrate this loss without altering historical v1, persist a NEW pinned adapter/head checkpoint, then run frozen C0 vs C1 Stage-A transfer. If and only if it promotes, run official fold-local Stage B and the one-shot Code Competition packaging. The current module alone is NOT a runnable complete candidate, so do not claim it has been validated or submitted.

Never treat source capacity, synthetic tests, or absence of score as evidence that this architecture beats 0.520/0.668.


## Continuation engineering checkpoint — 2026-11-08 (NO trained result)

**Completed real executable stages (not science scores):**

- Train-only class-conditioned source adversary, GRL and crossed source support: scripts/conditional_domain_adversarial_lora24.py.
- Separate CUDA Stage-A continuation using original SHA-locked 24-layer A1 LoRA24; no mutation of v1. Fixed C0 lambda=0 vs C1 lambda=.10; source/prompt/answer exclusions, original dataset clearance manifest fail-closed, private adapter/head checkpoint output and private heldout per-row predictions. Source: scripts/cda_l24_stage_a.py.
- Frozen paired 16-label source cohort evaluation: 5,000 prompt-group bootstrap, per-class BA, family/provider protection, rescue/regression, no Public-LB/official labels at train time. Source: scripts/cda_l24_transfer_gate.py.
- NEW checkpoint loader keyed to full directory manifest SHA, source clearance manifest SHA, exact original A1 lineage and 18-label order. Does not weaken original v1 A1 checkpoint enforcement. Source: scripts/cda_l24_checkpoint.py.
- Fold-isolated Stage B implementation that reuses original train_two_stage2_epochs and _infer_validation; each fold fresh child process, full 2,699 row identity / prompt split checks and private row-level output. It remains conditional on Stage A promotion and new checkpoint integrity. Source: scripts/cda_l24_stage_b.py.
- Focused synthetic tests: tests/test_conditional_domain_adversarial_lora24.py, tests/test_cda_l24_stage_a.py, tests/test_cda_l24_transfer_and_checkpoint.py. [CI Run 42480920892](https://github.com/kolihhan/name-kaggle/actions/runs/42480920892) succeeded with **13/13 tests**, including train-only gradient reversal equivalence, strict clearance fail-close, source/prompt/text exclusion, paired prompt bootstrap and checkpoint tamper checks. Training scripts also passed import/compilation/CLI checks. Synthetic tests demonstrate engineering validity, not model quality or GPU runtime.

**Source provenance remains unresolved:** [first-party owner search Actions 42478917599](https://github.com/kolihhan/name-kaggle/actions/runs/42478917599) found exact arxitekton/ntl-claude-gen-v1 and drjcfmd/ntl-generated-v2 records, but returned license_name=None and no description/served-version protocol. The earlier Kaggle dataset_view client method was unavailable. Neither publicly accessible aggregate data, publisher's asserted license strings, nor dataset-owner title proves downstream authorization, generation metadata or independent prompt/model source. An original source README/license/generation manifest or equivalent authoritative evidence is needed before training; this is DATA PROVENANCE BLOCKED, not a failed adversarial objective.

**Work not completed:** No C0/C1 GPU science training, no trained CDA checkpoint, no heldout BA, no 5-fold CDA official OOF, no final Kaggle Code Competition inference notebook, and NO Submission ID. Original reproduced Public Best remains 0.668, independently trained LoRA24 Public 0.520.

**Future closure criteria:** authoritative source licenses, exact served labels/versions, and independent cross-source lineages; a separately authorized accelerator runtime; actual C0/C1 matched source transfer success, then Stage-B and a verified final inference Notebook. Do not infer a benefit from capacity screening or 13 green synthetic tests.
