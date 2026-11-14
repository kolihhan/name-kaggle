# NTL Lightning 0.694 — Inference-only candidate, preregistered before any observed GPU result
Date: 2026-11-15 MYT. Status: PRIVATE KAGGLE QUEUED; official submission NOT authorized by this document.

## Source and hypothesis
- Exact external Kaggle public Notebook: `lightningv08/0-694-glm-5-3-likelihood-qwen3-8-27b-lora` SHA256 `9808d997185c125d92b0a67501bb763eb1813ae5223d3d06ce82a5518f205a2e`.
- Original author title claims 0.694 but neither that numerical score nor its hidden transfer is authenticated on our account.
- Compared with our own actual best 0.683 (#58295802, attributed `areebbb` 8-expert reproduction), this is a materially different **six-pretrained-source stack including a Qwen3.8-27B LoRA signal**. It also uses GLM; our isolated GLM likelihood route failed external transfer (-11/128), so novelty must come from complementarity of the entire frozen stack, not copying that failed route.
- No new neural training, Gaussian supervised fit, SVC fit, Public-LB tuning, or test-label access.

## Immutable transforms
- Builder: `scripts/ntl_lightning_0694_no_fit_candidate.py`; pinned original source and same **23 datasets / 9 public notebook sources / 6 model sources**, original 50 code cells.
- Genuine inference: changes exactly one original preview assignment to `preview = False`. Previously existing constant three-row shortcut cannot short-circuit model inference in own private preview.
- Fail closed: replaces the single live `gauss.fit(` call by a function that raises, and puts a raise at beginning of the known Atlas `if training:` branch. This is not a proof that all possible training mechanisms are absent, but a source-validated guard for both identified supervised fit paths.
- [Actual public-source CPU gate #43621922857](https://github.com/kolihhan/name-kaggle/actions/runs/43621922857) PASS. Candidate's own public description must clearly credit Lightning.
- Private Kaggle push [#43622052129](https://github.com/kolihhan/name-kaggle/actions/runs/43622052129) SUCCESS. **Actual Kaggle slug** is `kolihhan/lightning-0-694-authentic-no-fit-inference`, title-derived by Kaggle; the builder's metadata `id` names a different requested slug, so never use the requested slug to check existence or attempt a second push.
- [Authenticated private state #43622626598](https://github.com/kolihhan/name-kaggle/actions/runs/43622626598): `QUEUED` at 2026-11-14 17:43 UTC; private kernel exists and is owned; no model outputs, score or official entry.

## Advancement and stop rules
1. Inspect this SAME already-queued private kernel once it is terminal, never create duplicate GPU kernels to get a status update.
2. Require actual full inference on variable-size input (not 3-row preview cache), legal 18-label predictions and correct submission.parquet mapping. No new fit or gradient update.
3. Check whole-test 9-hour hidden inference budget at realistic input scale and model memory conditions. One private preview is NOT a scale demonstration.
4. Pair against hash-pinned 0.683 champion on genuinely source-qualified all-18 label, prompt/source-disjoint same rows, with balanced accuracy, rescue/regression, uncertainty and upstream served-model provenance. Existing external pooled 65 sources has **0 individual complete 18-label sources**, so it is NOT already qualified.
5. A positive independent, meaningful difference plus runtime/source checks is required before official promotion. A public author title or own Kaggle status is not improvement evidence. No new official competition entry from this contract automatically.
6. If GPU path errors, training guards trigger, pretrained artifacts unavailable, 18-label disjointness fails, or estimated runtime is implausible, **STOP and report**. Do not retune source/sibling thresholds against Public Kaggle score.

## Factual current score
Our authenticated official best remains **0.683**, submission **58295802**. New candidate official score **UNKNOWN**. This is an external open-source reproduction adapted for safe inference, **not a new original model**.
