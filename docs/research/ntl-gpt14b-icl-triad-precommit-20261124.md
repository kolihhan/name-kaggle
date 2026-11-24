# Qwen3-14B In-Context GPT Sibling Attribution — frozen contract (2026-11-24)

**Owner instruction:** No small-model training in Name That LLM. This experiment trains **no** models, does not fine-tune weights, uses a pretrained 14B-capacity decoder only for inference, and does **not** submit to Kaggle.

## Problem and earlier evidence
- The verified owner Public BA **0.718** is an outside public Notebook reproduction; its complete same-row labelled 18-class external predictions are not yet available. Never claim this standalone diagnostic beats 0.718.
- Genuine spent [Oleg216 oracle inspection #45140011226](https://github.com/kolihhan/name-kaggle/actions/runs/45140011226) showed for the earlier **0.691** source scorer **GPT family 33 correct/60, 27 wrong sibling, zero wrong other family**. This is not a result for 0.718.
- A 3-class TF-IDF+LinearSVC GPT gate already tried [#44128906627](https://github.com/kolihhan/name-kaggle/actions/runs/44128906627) **1 rescue / 3 regressions**, macro BA 0.412037→0.402778 on the spent Oleg216. Do **not** revive/retrain that weak head.
- The last observed full-stack Fable252 [read-only #45429334099](https://github.com/kolihhan/name-kaggle/actions/runs/45429334099) remained QUEUED (not score evidence).

## Mechanism
**Question:** Can **a large pretrained causal decoder** use 3 matched, same-prompt GPT-version exemplar triplets to recognize one independent single response's variant without knowing the hidden-test prompt?

- Fixed inference weight mount **Qwen3-14B** `qwen-lm/qwen-3/Transformers/14b/1`, pinned Kaggle provider model version; 4-bit NF4 inference, no training/gradient. If exact weight/tokenizer is not mounted or numerical inference is unsupported, report BLOCKED; never substitute a different model after results.
- Three exact sibling labels: `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`. All are classes in official competition.
- Choose a fixed deterministic **24 prompt groups** from 150 official train prompts (using SHA256 sorted prompt_id, no model/text or outcome selection). The heldout 24×3=72 official responses cannot be shown as examples.
- Choose 3 distinct remaining official prompt groups (fixed hash order) as demonstrations. Each triplet holds the **same prompt** with Luna/Sol/Terra outputs side by side. Important: pairwise matching is only in public training exemplars; **never use heldout or test prompt_id** in prompt construction.
- External source: publisher-labeled **Beroli C1** `beroli/ntl-benchmark-c1-generated-panels` `passages.csv`, SHA256 `8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177`; exactly 30 matched prompt panels and 12 labels, use all 30×3=90 distinct GPT responses as source-transfer probe. Source provenance remains publisher-reported, not independently verified.
- All heldout + external texts checked for normalized **exact duplicate contamination** against demonstration exemplars (abort on any). 72+90 = **162 unique texts**, 3 classes, two domains. Texts/IDs/logits remain private.

## Three fixed arms, scored with same model weights
1. **ZERO**: class names and a query response, no examples.
2. **PAIR**: same task but add 3 same-prompt training triplets, each of 3 GPT versions, correctly labelled.
3. **PLACEBO**: same demonstrations and prompt lengths, but apply *different cyclic label rotations per triplet* (invalid model identity mapping, content controlled).

Use **balanced answer-position rotation** based on SHA256 of query ID, fixed across all arms, so a raw A/B/C preference is not mistaken for identity. Predict from the next-token conditional probability of *one-token* A, B, C after `Answer:`, using `AutoTokenizer` with fail-closed exact single-token IDs. Never use truth labels in prompt or exemplar selection; labels only after logits for paired evaluation. Fix context/truncation at tokenizer max 3072 tokens (fail on too long, do not silently drop examples). Freeze prompt, all exemplar row IDs, digest, stage order and seed before first GPU execution.

## Acceptance and falsification
- Test **same-row** (identical 162 passage IDs, model, tokenizer, decoding/scorer) ZERO vs PAIR vs PLACEBO. Report per-panel 3-way macro BA, paired rescues/regressions, label recall, confusion, prompt-cluster bootstrap CI and exact runtime/passage.
- Promising **new large-model evidence**, NOT promotion: PAIR–ZERO >= +0.08 macro BA **in both** official heldout and Beroli C1; PAIR > PLACEBO **in both**; no class has zero recall in PAIR; and standard errors/paired prompt-bootstrap lower bound >0 for PAIR–ZERO. No rescoring earlier observed Oleg data. Otherwise diagnose source shift/label-prior/prompt-format/LM capability without discarding all ICL.
- A 3-way conditional comparison is not 18-class BA. **No competition submit** until actual 0.718 champion same-row full18 predictions on a qualified source, 18-class inclusion safe, per-label rescues/regressions, and a measured full 4k-row 9h runtime estimate.
- Kaggle one-shot private GPU kernel, no paid provider API, no new training, no background scheduler. Stop rather than retry if model mount/GPU quota/protocol is unavailable. Do not claim official Public improvement.

## Caveats
- Even if PAIR wins the 3-way diagnostic, an 18-class runtime-suitable inference gate is a **different engineering problem**; triad oracle membership is not available for wrong-family predictions. A pretrained 14B open decoder is a proxy classifier, not the original GPT-5.6 model's own causal likelihood; cannot assert exact served identity.
- Official heldout prompts have already been inspected across many rounds; this is an exploratory paired mechanism check, not an untouched independent holdout. Beroli C1 has been used before and shares publisher with C2. Do not optimize prompt or exemplars after seeing their scores.
