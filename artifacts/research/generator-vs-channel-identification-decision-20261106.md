# Research decision — generator identity versus collection-channel fingerprints

**Date:** 2026-11-06  
**Role:** independent Research PIC; research only, no training, inference, scientific experiment, Actions launch, Kaggle submission, or deployment.  
**Decision:** **Prioritize identification of the generator-vs-channel confound before selecting another attribution architecture.** Existing OOF and public leaderboard observations cannot answer it. No architecture is promoted.

## 0. Executive verdict (what changed from previous research)

The preceding [conditional attribution diagnosis](independent-conditional-attribution-diagnosis-20261106.md) established descriptive but strongly heterogeneous OOF failures:
- Gemini provider correct **97.50%**, chiefly sibling ambiguity;
- GPT **88.00%** provider correct, substantial sibling ambiguity;
- GLM **42.00%** provider correct, primarily cross-provider errors;
- Centered-head sibling error falls **0.01819**, cross-provider error rises **0.01710**, total BA virtually unchanged.

This does **not** establish whether generator-specific signals truly differ by family. An alternative, **unfalsified**, explanation is systematic acquisition-protocol or labeling confounding. Different source banks can use different system instructions, wrappers, token budgets, temperatures, API proxy routes, serving snapshots, and prompts. A classifier may faithfully recognize those collection artifacts rather than the model that generated the answer.

**New research decision:** Treat the difference between **source-model effect** and **generation-channel effect** as a scientific identifiability problem, not another search for style features, per-family heuristics, or a new model family.

## 1. What the official Kaggle task does and does not identify

First-party [Kaggle competition overview](https://www.kaggle.com/competitions/name-that-llm/overview/abstract): **18 exact model names**, model-level **balanced accuracy**; class counts unequal, public/private mixes differ. [Official data page](https://www.kaggle.com/competitions/name-that-llm/data): **2,699 training rows over 150 prompts**; hidden ~4,000 examples use **unseen prompts**. Training has `prompt_id`, while test has `id,text` only. "Every passage is an authentic LLM completion"; light formatting cleanup.

**Useful strength:** The benchmark's same-prompt multi-model design substantially controls prompt *content* in original training (2,699 rows ≈ 150 × 18). This is not the same as controlling **system prompts, sampling settings, model/version aliases, providers, batch dates, or provenance in external data**. Organizers' complete per-model generation recipe is **not established by the checked public description**. Therefore do **not** claim that hidden prompts, protocol, or label frequencies follow an assumed batch structure. A leaderboard gain does not prove intrinsic model fingerprint transfer.

### Identification argument, not a new experiment

Let `M` = true generator model, `P` = prompt, `C` = generation/collection channel (system instruction, decoding/length settings, API gateway, snapshot date, formatting). Observed output is `Y`.

The classifier observes the marginal distribution:

`Pr(Y | M) = sum_{p,c} Pr(Y | M,p,c) Pr(P=p,C=c | M)`.

If `C` is systematically coupled to `M`, the second term changes by model. Even perfect source classification on held-out prompts cannot tell whether prediction uses `M` directly or the proxy `C`. More prompt splits with the same confounded channel assignment **do not break that equivalence**. This is a **logical non-identifiability possibility**, not proof that this specific competition was confounded.

To identify relative generator effects, we need *crossed, matched support*: at least two models observed under the **same prompt and generation channel**, and at least one model observed under **multiple channels** with exact version/date/configuration recorded. If a field is hidden, report identification limitation, not an invented causal interpretation.

## 2. Novelty / transfer literature audit — direct relevance and category errors

| Primary source | What it actually establishes | What it does NOT justify in our repo |
|---|---|---|
| [ATLAS / MAABench, Ko et al., 2026](https://komseok0109.github.io/ATLAS/) | Family-conditioned token likelihood with clipped evidence; matched prompts; **5 families**, version/domain/joint OOD evaluation. Reports **74.27%** accuracy on its joint-shift benchmark, vs **65.27%** for best baseline, different source distribution and metric. | A new 18-way sibling solution. **MAABench trains 1 version per family and evaluates newer versions as the *same family***, not as separate class labels. Applying ATLAS as a family expert would not establish Gemini 3.6-vs-3.7-vs-3.8 identification. Also OOD domain shift changes system prompt, so effects are not cleanly isolated. |
| [Profiler, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1265/) and [authors' implementation](https://github.com/MarkGHX/Profiler) | Black-box origin attribution with surrogate context-loss patterns, evaluated across several domains and **five older commercial models**. | Novelty from "adding context likelihood"; proof of 18 frontier model exact-version separation. |
| [OpenTuringBench, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1354/) | Attribution vs binary detection and multiple OOD settings in open models. | Direct cross-domain scores for our closed frontier targets. |
| [Implicit Identity survey, IJCAI 2026](https://ijcai.org/proceedings/2026/880) | Separates text-only fingerprinting from watermarking and parameter/activation fingerprinting; highlights *identifiability*, *robustness*, *deployability*. | Evidence of actual hidden watermark or privileged access to target model internals. |
| [Every Step Counts, ICML 2026](https://proceedings.mlr.press/v306/li26ir.html) | Uses **observed diffusion-LLM decoding steps** to attribute diffusion models/checkpoints. | **Not direct support for our DCT-v3**. Our [DCT-v3 contract](../candidates/dct-temporal-fingerprint-v3/CONTRACT.md) projects the token-position hidden states of a **Qwen3-8B surrogate classifier** processing a *finished response*, not the inaccessible source generator's actual decoding trajectory. Calling it a measured "generator decoding fingerprint" would be a category error. DCT may still be useful as a *surrogate sequence representation*, but its efficacy remains **UNRUN** and not independently supported by that diffusion paper. |
| [Behavioral Fingerprints for LLM Endpoint Stability and Identity, 2026](https://doi.org/10.1145/3786335.3813194) | Active endpoint probes can respond to sampling, inference-stack, and model version changes. | Direct text-only attribution on a single prompt-free completed passage, or evidence all variation reflects changed model weights. |

**Novelty veto:** ATLAS likelihood, Profiler context loss, ordinary family experts, prompt-mean centering, source-conditioned weighting, and DCT positional pooling are NOT individually novel claims for this project. Any defensible contribution must demonstrate a capability absent from them: e.g., **exact-sibling attribution under controlled protocol changes with preserved provider recall**, rather than another public-OOF/leaderboard blend.

## 3. One discriminating research design (frozen *proposal*, not authorization to run)

**Question:** For a matched prompt and a fixed model endpoint version, does changing the generation channel perturb attributed identity more than changing between close sibling models?

Proposed crossed design, for each prompt `p`, sibling models `m1,m2`, and allowed protocols `c0,c1`:

| | c0: standardized settings | c1: changed settings |
|---|---|---|
| m1 | `Y(m1,p,c0)` | `Y(m1,p,c1)` |
| m2 | `Y(m2,p,c0)` | `Y(m2,p,c1)` |

For more than two siblings, extend the same full factorial table. `c0,c1` must be concrete recorded conditions (e.g. temperature and system-prompt version), **not** a source-bank label standing in for many simultaneous unknown changes. Preserve the exact request, passage constraints, endpoint/version, timestamp, outputs and generation settings. Do not assume a commercial API supports identical semantics for every model; record unsupported settings and restrict the claim appropriately.

**Before any generation or model evaluation:** establish whether exact model/version endpoints are available; whether every requested cell is observable, license-compliant and time-stable; whether source/permission allows independent use. If the design is not sufficiently crossed, outcome is **NOT IDENTIFIABLE**, not a new research metric.

### Future observable comparisons — no numbers currently

1. **Within-model protocol stability:** On `m1` and `m2`, compare identity predictions across `c0,c1` for same `p`. This tests whether the classifier tracks channel changes in the absence of a label change.
2. **Between-sibling distinguishability:** Compare `m1` vs `m2` under the **same** `p,c`. This controls prompt/protocol and isolates the challenging version difference *within the crossed design*.
3. **Two levels of error:** exact-model recall, **wrong sibling** recall and **wrong provider** recall; do not apply family-hard routing and accidentally erase GLM/Kimi.
4. **Transfer condition:** lock prompt groups *and* source/batch/temporal provenance before model-development. Evaluate held-out prompts under previously unselected channels/batches if supported. A prompt-only grouped split is insufficient for source robustness.
5. **Controls and interpretation:** no free use of post hoc style stripping, length normalization, or synthetic paraphrases as "model-preserving counterfactuals". They can remove identity signal or introduce a new generator. Report actual system instructions and distributions when known. Do not claim a model-version intervention is analogous to varying temperature.
6. **Metric scope:** exact 18-class balanced accuracy needs all 18 labels; a two-sibling paired design can demonstrate **conditional discriminability** only, not overall competition performance. Precommit the class set, metric, errors and stop/pivot conditions before seeing future labels or scores.

### Predicted implications (hypotheses, not results)

- If the *same* model flips attribution under settings changes but distinct siblings are stable only under matched settings, **channel dependence** is a plausible bottleneck; improve data/protocol support, not an extra LoRA backbone.
- If siblings remain inseparable under strict matched settings with good provider identification, investigate a higher-information conditional representation rather than a global family classifier.
- If GLM's cross-provider failures vanish under standardized matched conditions, current GLM "intrinsic provider weakness" conclusion is **falsified**.
- If provider/cross-sibling errors remain stable across batches, prompts and protocols, then the signal is more plausibly generator-dependent, though unobserved platform configuration still limits causality.

## 4. Feasibility evidence and explicit gap

A [public Benchmarks proxy generation dataset](https://www.kaggle.com/datasets/zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations) states it contains **834 GPT/Gemini** generations with `run=ab2/ab_settings` subrows under temperature variations for some matched prompts; complete prompt/request/model/settings fields are claimed by its data card. **Not a newly independent 18-label verification dataset:** it overlaps the type of sources already considered in project research, offers limited settings variation, and its card does not prove all four cells of a crossed prompt/model/protocol design are present. This is merely a concrete *existence proof of partial logged protocol variation*, not a candidate for immediate score tuning.

Likewise [MAABench](https://komseok0109.github.io/ATLAS/) has matched prompts and deliberate OOD factors but uses **five families / one train version per family** and thus cannot directly evaluate 18 exact versions. Strong family-OOD performance and sibling-discrimination performance are different research questions.

**Gap:** At present the inspected artifacts contain no verified matched-prompt **and matched-protocol** independent 18-label holdout with exact version IDs, immutable raw response provenance, and aligned candidate predictions. Additional ad-hoc source listings do not solve this; only direct evidence of the crossing does. Do not fabricate an end-to-end evaluation plan that assumes it exists.

## 5. Research gate / what is prioritized

**Priority 1 (now):** Document the scientific identification condition and reject any study that compares per-source macro BA across differently covered label sets as if it measured domain shift. No compute is needed for this research decision.

**Priority 2 (next permitted research-only continuation):** Inspect *one* credible, published source's generation script or raw data card solely to establish whether the four-cell crossing exists for **one close-sibling pair**, with exact protocol and version. If it does not, record **unavailable** and stop; don't repeat source inventories. Candidate source dataset card above is a reference, not automatic authorization to reuse consumed examples.

**Priority 3 (only after separately authorized scientific evaluation):** Run a *single* registered matched-contrast attribution probe, as specified above, not a broad LoRA/DCT/feature search or Kaggle leaderboard submission. Prefer existing archived predictions and read-only analysis if legitimately aligned. New model training or API generation is **not authorized by this note**.

### Do not do next

- Do not launch DCT-v3 GPU training because a paper says "decoding trajectories". It is a *surrogate hidden-state temporal pooling hypothesis*, not direct generator trajectory evidence; frozen contract remains unchanged and scientific effect unobserved.
- Do not optimize SVC views, family hard gates, Hungarian/one-prompt-per-model pairing, Atlas blending, or per-class thresholds on already-seen official OOF.
- Do not call another author's public competition score our innovation; third-party reproduced Kaggle `0.668` vs own LoRA24 `0.520` changes many factors at once.
- Do not use `58045266` pending / future score as an instruction to submit or tune. Historical submission state belongs to operations, separate from research.

## 6. Falsification/stop, novelty, next decision

**FACT:** Existing OOF heterogeneity (Gemini 97.5% family correct vs GLM 42%, consumed evidence) and the documented DCT-v3 *surrogate* representation; official prompt/label/metric structure; external literature task definitions.

**INFERENCE:** Acquisition channel and generator model may be confounded in public source corpora. The difference is not identified in current project evidence.

**UNKNOWN:** Whether official organizers standardized every serving parameter; whether frontiers' exact historical version endpoints are accessible for crossed generation; how many legitimate paired cells exist in public data; whether any new approach improves 18-label hidden BA or meaningful OOD attribution.

**REJECTED:** "The source-transfer score difference measures domain effect" (unequal label sets); "family accuracy alone implies exact version accuracy"; "DCT sequence over surrogate states measures generator decoding trajectory"; "higher public ensemble score identifies the causal winning component."

**PROPOSED STOP:** If documented data cannot cross model × prompt × channel **and** support an independent split, mark the causal comparison unavailable. Do not manufacture a substitute via source shuffling, rewritten text, fake negatives, or another re-split of spent 2,699 OOF rows.

**Potential contribution if confirmed later:** A *protocol-stable, model-conditional evidence reliability* framework with model- versus channel-induced uncertainty explicitly decomposed. This is only an **idea**, not a demonstrated novel algorithm; overlaps with robust MAA, invariant prediction, and controlled evaluation design must be surveyed before a research novelty claim.

**Final action this turn:** Research synthesis only; no original model trained, no Kaggle submitted, no GitHub Actions launched.
