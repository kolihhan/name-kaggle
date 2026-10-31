# Name That LLM — Phase 2: External-Data Reconnaissance

Date: 2026-10-31

## Objective

Before encoder / LoRA experiments, identify public generation corpora that can improve model attribution **without contaminating prompt-grouped CV** or introducing mislabeled model snapshots.

The official competition allows freely and publicly available external data and pretrained models. The local validation contract remains unchanged: 5-fold `GroupKFold` by official `prompt_id`, with unseen prompts as the generalization target.

## Main decision

Do **not** ingest the largest public pool first.

Start with the smallest external corpus that has:

1. exact competition-label mapping;
2. row-level generation provenance where possible;
3. prompts independent of official train prompts;
4. a usable public license;
5. no exact / near duplicates of official train text;
6. outputs reasonably close to the host protocol (single self-contained passage, roughly 40–250 words, no visible chain-of-thought/tool traces).

External data is admitted only after it improves the frozen prompt-grouped OOF evaluation. Random-row CV is not accepted as evidence.

## Candidate ranking

### Tier A — first dataset to materialize and audit

#### 1. Name That LLM: OpenRouter + OpenAI Generations

Source: https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations

Observed public metadata:

- 7,753 passages;
- 10 competition labels;
- labels: `gemini-3.7-flash`, `glm-5.3`, `glm-5.3-flash`, `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`, `kimi-k3`, `qwen3.8-max`;
- per-row provenance fields include API route, provider, served model, generator, and source tag;
- prompt bodies were filtered against competition train passages and the publisher reports no exact or near duplicates of train text;
- generation constraints explicitly target single-paragraph passages and train-like length distribution;
- CC BY 4.0.

**Decision:** ACCEPT FOR RAW AUDIT FIRST.

Why: strongest currently verified combination of exact labels, provenance, independent prompts, protocol similarity, and license clarity.

Main remaining checks: actual downloaded schema/counts; served-model completeness; per-label balance; text-length distribution; independent duplicate/near-duplicate scan against official train.

### Tier B — promising, but do not train on yet

#### 2. Name That LLM: same-prompt generations v2

Source: https://www.kaggle.com/arxitekton/ntl-claude-gen-v2

Observed public metadata:

- 3,322 host-protocol passages;
- 14 candidate models;
- extends an earlier generation set.

**Decision:** HOLD FOR RAW AUDIT.

Unknowns that matter: exact 14-label list, license, row-level provider/model provenance, and whether any prompts are reconstructed/derived from official train prompts.

#### 3. Name That LLM: Extra API Passages

Public evidence: https://www.kaggle.com/code/seyitkaangunes/does-extra-data-help-ntl-extra-passages

Observed public notebook metadata describes v1 as new-prompt passages for the five Claude models and evaluates it with grouped CV.

**Decision:** HOLD / TARGETED VALUE.

Why it may matter: Phase 1 showed substantial same-family confusion. Claude-only external data could improve sibling separation without needing a huge full-label corpus.

Unknowns: source dataset card, exact generation provenance, license, counts, and protocol match.

#### 4. Name That LLM: Response Corpus

Source: https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus

Public index metadata reports 18 model columns, shared prompts, and external response provenance.

**Decision:** HOLD.

Reason: potentially excellent full-label coverage, but the original dataset card/license/provenance details still need direct verification before training use.

### Tier C — useful discovery pool, not trusted training data yet

#### 5. Trust-Worthy Pool Dataset — 254k passages, all 18 models

Discovered in the current Name That LLM Kaggle discussion index as a merged and deduplicated pool of public datasets.

**Decision:** DO NOT USE YET.

Size and label coverage are attractive, but an aggregate pool can hide:

- duplicated upstream datasets;
- mixed licenses;
- alias/model-snapshot mistakes;
- prompt leakage from reconstructed official prompts;
- very different generation protocols;
- provenance loss during merging.

It becomes useful only after source-level lineage is recoverable.

## Rejected for now

### Qwen3.8-Max Distillation 50K

Source: https://huggingface.co/datasets/r0b0tlab/qwen3.8-max-distillation-50k

Reasons:

- teacher is identified as `qwen3.8-max-preview`, not necessarily the exact competition `qwen3.8-max` snapshot;
- dataset contains reasoning/tool-use style traces unlike the competition's self-contained short passages;
- heterogeneous upstream licensing and explicit terms caveats are documented by the dataset publisher.

**Decision:** REJECT FOR PHASE 2 TRAINING.

## External-data admission gate

Every external source must pass all applicable checks before training:

| Gate | Requirement |
|---|---|
| Label identity | Exact competition label or independently verified exact model snapshot |
| Provenance | Generation route/model/provider documented; row-level preferred |
| License | Publicly usable under competition rules; attribution recorded |
| Prompt independence | New prompts preferred; official-prompt-derived data requires fold-isolated handling |
| Duplicate safety | No exact duplicates; near-duplicate similarity audited against official train |
| Protocol match | Prefer single passage, ~40–250 words, no headings/tool traces/visible CoT |
| Coverage | Per-label counts recorded; imbalance handled explicitly |
| Honest validation | Improvement must appear on the frozen `prompt_id` GroupKFold OOF |

## Leakage rule

External generations derived from reconstructed or recovered **official train prompts must never be globally added to every training fold**. That would expose validation-prompt semantics during CV and inflate OOF performance.

If such a source is ever tested, its prompt lineage must be mapped back to official `prompt_id` and excluded from the corresponding validation fold's training data. The simpler default is to reject official-prompt-derived external data during model selection.

## Experiment order

1. Materialize Tier-A OpenRouter + OpenAI corpus.
2. Run raw audit: schema, label mapping, counts, lengths, provenance completeness, exact/near duplicates.
3. Run the current classical baseline with **official train only vs. official + Tier-A** under the exact same frozen folds. This is a cheap fidelity test, not the final model.
4. If external data improves honest OOF without damaging minority labels, use it in the first encoder/LoRA experiment.
5. Audit Tier-B sources one at a time. Add only sources with incremental OOF benefit.
6. Do not touch the 254k aggregate pool until upstream lineage is auditable.

## Evidence status

- **VERIFIED:** official competition allows freely/publicly available external data; 18 labels; balanced accuracy; hidden test uses unseen prompts / train-like short passages.
- **VERIFIED:** Tier-A dataset public card reports 7,753 rows, 10 exact competition labels, row-level provenance, train-overlap filtering, and CC BY 4.0.
- **VERIFIED:** same-prompt generations v2 public page reports 3,322 passages and 14 candidate models.
- **VERIFIED:** public notebook metadata shows Extra API Passages is being tested as new-prompt Claude data with grouped-CV ablation.
- **PARTIALLY VERIFIED:** Response Corpus identity and broad schema are visible through public indexing, but the original card needs direct inspection.
- **PARTIALLY VERIFIED:** 254k all-label pool exists in the current discussion index; detailed lineage has not yet been established.

## Phase 2 checkpoint

Reconnaissance is complete enough to choose the first source. The next action is **not** LoRA yet: materialize and independently audit the Tier-A 7,753-row corpus, then run a frozen-CV augmentation ablation.
