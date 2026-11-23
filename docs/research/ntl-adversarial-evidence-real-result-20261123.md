# NTL adversarial evidence review — actual Kaggle corpus records and 0.715 static scorer (2026-11-23)

## Terminal experiments and constraints
- [Actions #45185085771](https://github.com/kolihhan/name-kaggle/actions/runs/45185085771) SUCCESS: original two datasets and exact scorer source, SHA verified.
- [Actions #45186036073](https://github.com/kolihhan/name-kaggle/actions/runs/45186036073) SUCCESS: add C1/C2 `passages.csv` data inspection after noticing the first scanner only recognized descriptive file names.
- [Actions #45186312734](https://github.com/kolihhan/name-kaggle/actions/runs/45186312734) SUCCESS: fix a **false-negative metadata detector** (`model_served` had been unrecognized); verify actual publisher-served metadata, C1/C2 prompt intersections, per-class support and run dates.
- The scripts ran CPU only, downloaded source texts **ephemerally**, and logged only aggregates. No new 0.715 inference, QLoRA, paid provider call, Kaggle GPU, score or submission. Original texts/prompts/record IDs never committed or emitted.

## Directly counted public original data files

| Publisher dataset | file SHA256 | Rows | Exact NTL class support | distinct prompt keys | 18-label holdout? |
| --- | --- | ---: | ---: | ---: | --- |
| `ankiteshtiwari/ntl-exact-corpus-v3` | `5a60017dc2e3d4a153fb1455479a9a4734366169db1140d5a046557a04f96a65` | **17,009** | **9/18** | 4,075 | **NO** |
| `ankiteshtiwari/ntl-blocks-v1` | `51d9d9d57b80a3d963428ad51e1d96e66113634e0c8ad390714984274b4a188d` | **1,800** | **2/18** | 900 | **NO** |
| `beroli/ntl-benchmark-c1-generated-panels` | pinned in CI aggregate log | **360** | **12/18** | **30** | **NO** |
| `beroli/ntl-benchmark-c2-generated-panels` | pinned in CI aggregate log | **470** | **12/18** | **50** | **NO** |

### Specific observations and non-claims

- `exact-corpus-v3` schema: `prompt_id,model,text,n_words,source`. Its 9 exact classes are `claude-fable-5`, `claude-opus-5`, `glm-5.3`, `glm-5.3-flash`, `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `kimi-k3`. This is not an 18-class benchmark, despite an exact-identity name; no distinct served-model field.
- `ntl-blocks-v1` schema: `block,prompt_id,model,text,words,target_words,source`. Exactly 900 `claude-sonnet-5` and 900 `gemini-3.6-flash` responses; no distinct served-model field. It is a **cross-family** 2-label panel, not a source-version sibling model diagnostic.
- C1 has **30 examples per class across 12 classes**, 30 unique prompt IDs: an apparently complete **12-way** 30-prompt panel. C2 has 470 across the same 12 labels, 50 prompt IDs, **uneven/incomplete class-by-prompt representation**: class counts range from 17 `gpt-5.5` to 50 for two Gemini labels. Do not assume complete blocks.
- The six exact NTL classes absent from both C1/C2: `claude-fable-5`, `gemini-3.7-flash`, both `glm-5.3` siblings, `kimi-k3`, and `qwen3.8-max`. Thus C1/C2 cannot test GLM/Kimi/Qwen family recovery, a leading current bottleneck hypothesis.
- C1 and C2 have **zero common original prompt IDs**, **zero identical output texts**, and the same 12-label support. The two corpora come from **one publisher**, both generated 2026-11-22; distinct prompt IDs and byte-unequal responses alone do not establish independent generation channels, no semantic prompt overlap, or no prior trained encoder exposure.
- C1/C2 columns include `model_requested,model_served,served_matches,setting,response_id,run_start,task_version,target_words,domain,task_type`. Publisher self-reports `served_matches=True` for **360/360 C1** and **470/470 C2** rows. Raw `model_served` value is present on all 830 rows. A naive exact comparison to official labels yields 0 (provider route namespaces), and stripping only final `/` suffix matches **210/360 C1** and **320/470 C2**, covering seven literal label names. **This is not evidence that remaining five source classes were mis-served**: model IDs can use alternate vendor aliases; the publisher's more permissive mapping is not independently certified from original provider API responses. Keep result **INCONCLUSIVE for verified provider identity**.
- C1: 2 domains, 3 task types, one setting; C2: 10 domains, 5 task types, one setting. This is descriptive dataset metadata, **not an exact protocol equivalence proof**.

## Scored-owner 0.715 source observability: real pinned code, no inference

- The exact pulled owner-snapshot code matches earlier authenticated SHA256 `202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8`. Source has 26 code cells, ~232k code characters, **9 pretrained model attachments**, 26 dataset mounts, 10 kernel attachments.
- The scoring code references `/kaggle/temp/final-logits.npz` (pre-GLM intermediate) and `/kaggle/working/submission.parquet`; internal notebook code contains `pngy_residual`, `gauss.py` and `v1_stage`. This is static code evidence only, not proof of live full-hidden output availability or stage completion.
- Previous authenticated official Kaggle saved-output checks [#45139584293](https://github.com/kolihhan/name-kaggle/actions/runs/45139584293) found publicly downloadable 3-row preview or no saved parquet, not full hidden 18-class matched predictions. This new check did **not** run the scorer, so no same-row error-rescue comparison is possible.
- **Do not treat `final-logits.npz` as the post-GLM final 18-class posterior.** Scorer needs new in-run instrumentation or genuine matched private runs before claiming final-decision evidence.

## Evidence verdict (using falsifiable claim / control / contradiction)

| Claim | Verdict | Reason |
| --- | --- | --- |
| `ntl-exact-corpus-v3` is a full 18-class independent holdout | **NOT VERIFIED** | Actually 9/18 labels, no served-ID field |
| `ntl-blocks-v1` is a full 18-class source | **NOT VERIFIED** | Actually 2/18 labels |
| C1/C2 validate full 18-model generalization or GLM family recovery | **NOT VERIFIED** | 12/18 only, specifically missing GLM/Kimi/Qwen |
| C1/C2 supply some useful prompt-matched protocol metadata for scoped diagnostics | **VERIFIED structurally** | 30 / 50 prompt IDs, class labels and request/response metadata; publisher source not authenticated |
| Claimed `served_matches=True` proves actual provider version identical to official model weight snapshot | **INCONCLUSIVE** | Publisher's aliases and records are not independently corroborated with raw first-party API responses |
| A real full-18, same-row incumbent 0.715 vs new candidate has been completed | **NOT VERIFIED** | No true matched final hidden/external 0.715 output |
| ATLAS global53+margin improves NTL official 18-way Champion | **NOT VERIFIED** | Historical **two-way** GLM +1/300 and CI crosses zero; no 18-way end-to-end score |

## Independent challenge to the project assumptions

**Previously bad assumption A:** With no pristine 18-way corpus, all research should stop. **Reject**: a fully labelled independent 12-class panel can offer a *scope-labelled negative control* for source/channel invariance; it simply cannot verify 18-class BA, missing families or a Champion uplift. Any such test must freeze model/aliases/prompt groups before observing outcome.

**Previously bad assumption B:** More provenance gates will automatically improve 0.715. **Reject**: these actions discovered data limitations and a false-negative parser bug, **not a new attribution signal**. Engineering success is not model quality.

**Previously bad assumption C:** A schema named `model_served` proves true generator identity. **Reject**: aliases and upstream provider responses still need certification. Metadata can prevent false claims, but cannot replace inference and independent labels.

## Decision and next smallest useful step

- **STOP** general catalog crawling and GLM A0/A1 incremental retuning on spent train subset.
- **PARK** new full18 supervised experiment: none of these four newly checked corpora passes the *full18* gate, and exact 0.715 same-row final predictions are unavailable.
- **PRESERVE** C1/C2 as *potential* restricted 12-label source-channel diagnostic if a strong prefit scorer can be run with acceptable one-shot GPU budget; they cannot validate GLM/Kimi/Qwen. Next question should be **how to expose current 0.715 post-stage decisions on an actually labelled disjoint cohort at reasonable cost**, not another unrelated small text classifier.
- Reopen architecture exploration as controlled development work under a frozen cheap falsifier, explicitly separated from any claim of independent 18-way transfer. No user-paid target LLM evidence and no Kaggle submission without separate approval.

**Official own scored Public Champion remains 0.715.**
