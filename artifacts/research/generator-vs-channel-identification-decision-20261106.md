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


## 7. Executed source-level crossed-design feasibility gate — 2026-11-07

**Research action executed, no training or model inference.** Downloaded only the third-party public 834-row CSV for \`zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations\` using the existing Kaggle token in GitHub Actions. Did not download official competition files. No text, individual prompt/request payloads, API token or model checkpoint contents were printed, added to Git, or preserved as artifacts. Dataset CSV SHA-256 \`a8f61b647b5f2b2f99b73f083b7358445d93c3ebbff8607b44423850e9fb5583\`.

**Evidence:** [Completed run 42361361854](https://github.com/kolihhan/name-kaggle/actions/runs/42361361854), \`scripts/audit_channel_cross_design.py\`, \`.github/workflows/ntl-channel-cross-design-audit.yml\`. This is a standard-library-only, reproducible source-coverage audit, with no trained/inference models and no competition submission.

**Prespecified feasibility requirement:** At least **20 complete 2-model × 2-temperature matched prompt instances** where both models share exactly the same source-bank/prompt ID, target word count, and full request text, and each has both \`rlow_t1\` and \`rlow_t0\`. Exact requested length matters because earlier research already showed label-related length targets; temperature is not the only source of variation. The model slug remains fixed within each pair; no claims about actual immutable backend snapshots.

| Observed coverage | Number |
|---|---:|
| Total public passages | **834** |
| Exact model labels | **8** |
| Distinct \`(bank, prompt_id)\` keys | **213** |
| Keys with at least two model labels | **142** |
| Multi-model keys with **different** target word counts | **141 / 142** |
| Temperature \`rlow_t1\` rows | **747** |
| Temperature \`rlow_t0\` rows | **80** |
| Same-model T1/T0 paired keys | **10 per model × 8 models = 80** |
| Uncontrolled model-pair × prompt 2×2 combinations (matching **prompt only**) | **280** |
| Controlled model-pair × prompt 2×2 combinations (matching prompt **and target words**) | **2** |
| Controlled model-pair × prompt 2×2 combinations (matching full request text too) | **2** |
| Controlled **same-provider sibling** 2×2 combinations | **0** |

Only complete strict pairs: \`google/gemini-3.1-pro-preview\` vs \`openai/gpt-6-astra\` (one prompt key) and \`google/gemini-3.1-pro-preview\` vs \`openai/gpt-5.5-2026-04-23\` (one prompt key). These are not same-provider sibling contrasts. The word-count target changes on nearly every multi-model prompt in this public dataset. It is therefore invalid to interpret the raw same-prompt panel (280) as a controlled model × protocol design.

**Decision: FAIL the feasibility gate, do not run an attribution classifier / V47 transfer experiment using this source, do not retrofit the required count after seeing the result, and do not tune a mask/feature against 2 instances.** The data support a narrower but real mechanism: *requested output length is nearly always class-coupled within shared prompts in this source*. This suggests a potential generation-channel shortcut; it does **not** measure actual attribution error, show that test is similarly confounded, or prove the existing model learned the shortcut.

**Global bottleneck refinement:** The critical data gap is **genuinely crossed prompt × model × protocol × matched-length support**, not just a shortage of total synthetic passages. Adding another unbalanced source or a larger backbone without this support would not settle intrinsic sibling identification. Future trustworthy data must require an **identical fixed target length per prompt across models**, recorded exact model version and settings, and a separate prompt/source-disjoint transfer cohort before any classifier fit. Existing public-source audit is now closed; do not repeat this dataset's inventories.


## 8. Independent source search and read-only matched-sibling check — 2026-11-07

**Aim:** After the 834-row proxy dataset failed the frozen strict model × protocol matching requirement, check one credible external dataset with real model siblings under the same source/prompt before proposing new API generation. No training, inference, Kaggle submission, or official competition-data access.

### Access gate: Maniceet external API corpus

- Source: [maniceet/name-that-llm-openrouter-openai-generations](https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations), including purportedly Z.AI-served matched GLM and official OpenAI-served GPT sibling samples.
- Independent competitor source: [dcw06 GLM pair](https://github.com/dcw06/NameThatLLM/blob/main/docs/glm-pair.md) reports 316 Z.AI-served responses (158 shared prompts); adding them raised a CPU binary classifier from 51.7% to 53.3%, still below its 57.3% incumbent. [Qwen+StyleDistance follow-up](https://github.com/dcw06/NameThatLLM/blob/main/docs/glm-pair-qwen.md) reported 56.7% → 54.0%, with its 95% interval crossing zero. Neither experiment certified exact same output-length targets.
- **Fresh attempted authenticated read-only dataset retrieval:** [Actions 42362363455](https://github.com/kolihhan/name-kaggle/actions/runs/42362363455) returned HTTP 403 Forbidden at Kaggle dataset metadata. Previously blocked access persists. Did not circumvent permissions or infer unseen row-level target lengths. The reusable [audit script](../../scripts/audit_external_sibling_request_pairs.py) compiled but its actual raw-data check was not executed because access failed. **Decision:** stop this route until normal access changes.

### Public Oleg response corpus — exact metadata audit succeeded

- Dataset: [olegzholobov/name-that-llm-response-corpus](https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus), existing Phase 8 source. Original short-view Parquet SHA-256 **15a67d022315b2508af8edd749a7c041d42cc37e266106028c73e405659c400f** matches the prior Phase 8 pinned source; no unexpected drift.
- Execution: [Actions 42362668801](https://github.com/kolihhan/name-kaggle/actions/runs/42362668801), [reproducible no-training audit](../../scripts/audit_external_sibling_multisource_support.py). Reads only public external parquet files; output is aggregate metadata, no raw passages or prompt strings and no official train.
- 13,642 wide source rows; 4,446 strictly filtered (mechanically eligible, license-known, terms-clear, exact-prompt provenance) response cells. At same source + same prompt ID under the above filters, 200 GLM 5.3 / Flash pairs, 196 GPT-5.6 Luna / Sol pairs, and 199 GPT-5.6 Sol / Terra pairs; zero qualifying Gemini and Claude Opus same-source pairs.
- **Critical provenance discovery:** all these source+prompt pairs are attributed to one upstream source, **zlab-princeton/SWEeper-Bench-traces**, an agentic software-repair benchmark with 5,200 recorded runs and 42 agent configurations, rather than a controlled single-paragraph attribution generation. [First-party source dataset](https://huggingface.co/datasets/zlab-princeton/SWEeper-Bench-traces). Its records vary in model, harness, effort, task type and other settings. The Oleg parquet metadata has no complete request, target_words, temperature, or per-run generation configuration sufficient to establish parity across the compared siblings. Same *source* is **not** same *protocol*. **Important qualification:** the upstream SWEeper main-results table lists Codex/xhigh for GPT Luna/Sol/Terra and Codex/max for GLM 5.3/Flash, so those *main-run* comparisons may hold harness and effort constant within each sibling family. The checked Oleg-derived metadata does not link every extracted short passage to an exact upstream run/config or report complete request/target-length settings; do not claim that every pair actually had different harness settings either. Even the matched main-run regime is a coding-agent interaction, not the competition's self-contained paragraph protocol.
- **Precommitted pairwise length-sign diagnostic (not a classifier):** in the same-source/same-prompt short snippets, Sol was longer than Luna in **195/196** pairs; Sol was longer than Terra in **186/199** pairs (Terra longer in 12, tie 1); GLM 5.3 was longer than GLM Flash in **147/200** pairs (Flash longer in 45, ties 8). These are actual word counts of Oleg's extracted short text, **not requested target lengths**, and no causal model-identity effect can be inferred from their asymmetry. Relative within-pair lengths are not a usable hidden-test predictor when prompt pairs are unknown or missing.

### Research decision and follow-up boundary

**No source passed strict matched-sibling + matched-request/protocol requirements.** The newly found candidate 595 pair instances are valuable as a **negative-control example of severe source-specific length asymmetry**, not as proof of a useful model fingerprint or independent Kaggle generalization. Do not ingest SWEeper-Bench traces as if they were calibrated Name That LLM single-paragraph generations, retune on their length artifacts, or treat them as validation evidence. Existing Phase 8/9 results are historical and remain unchanged.

Next credible route: obtain a *separately licensed, request-level logged* paired generation cohort: same prompt, target words, system template, endpoint/version, routing/harness/temperature, and at least two sibling models; reserve a prompt-disjoint and channel/temporal transfer set **before** attribution model fitting. Alternatively establish these fields from already-existing raw public records. No Kaggle Benchmarks-free-quota generation: a separate team documented a host discussion restriction on using this resource to make competition training sets; confirm official permission before considering that path, and never assume free proxy requests are automatically permitted.

Research outcome: **one target blocked by real 403; another source fully reproduced but rejected at the exact-protocol gate.** Stop public data inventories until a concrete documented request-level cohort is identified. No new architecture, Kaggle score or training claim follows from this audit.
