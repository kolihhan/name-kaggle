# MPCF hash-reference feasibility — real one-shot result

**Executed:** 2026-11-10 16:30 UTC (2026-11-11 00:30 MYT).
**Decision: REJECT this exact no-fit char-hash paired-reference mechanism.** No scientific transfer or Kaggle promotion. Contract was [precommitted](CONTRACT.md) at commit `97e2cf072d7f1d5e9298934ed5f4afbae5f44ca5` **before** this real-data comparison. One execution: [GitHub Actions #42855077344](https://github.com/kolihhan/name-kaggle/actions/runs/42855077344), job `129965793036`, **SUCCESS**. Six synthetic unit tests passed (2 geometry invariants + 4 provenance/prompt split tests). An earlier CI attempt [#42854901364](https://github.com/kolihhan/name-kaggle/actions/runs/42854901364) failed in Python import path **before any actual MPCF result**, and was rerun after only changing `python path/to/file.py` to `python -m scripts.run_matched_panel_pilot`. No candidate tuning.

## Source and exact scope

Public `olegzholobov/name-that-llm-response-corpus`, with same-origin panels from `zlab-princeton/SWEeper-Bench-traces`. Exact short, prompt-matched, provenance-filtered pairs: **196 Luna/Sol; 199 Sol/Terra; 200 GLM 5.3/Flash** after 40–250 word eligibility. Fixed SHA256-sort 70/30 prompt-disjoint reference/heldout split (137/59, 139/60, 140/60 pairs). Both responses of a test pair are independently classified; query answers do not receive paired prompt metadata. Data extraction source and corpus hashes verified by [existing Oleg audit](https://github.com/kolihhan/name-kaggle/actions/runs/42855077344). The public Kaggle dataset advertises `License(s): other`: permissive reuse/redistribution is **NOT established**. No raw passages or per-item predictions were uploaded to the public repo.

All reported values below are **same-source held-out binary balanced accuracy** with a deterministic char n-gram HashingVectorizer (n=4096, ngrams 3–5), top16 reference pair retrieval, one fixed shuffled-prompt negative control, deterministic unpaired class-prototype control and a reference-median *length-only* shortcut diagnostic. Hash features are transformed without `.fit()`; reference labels are required by nonparametric lookup, but no neural/iterative supervised model optimization, GPU, Kaggle hidden test or official submission was performed.

| Pair / heldout prompt pairs | MPCF matched | B prompt-shuffled | Unpaired prototype | Length-only |
|---|---:|---:|---:|---:|
| GPT Luna vs Sol / 59 | **0.576271** | 0.584746 | 0.762712 | **0.949153** |
| GPT Sol vs Terra / 60 | **0.525000** | 0.566667 | **0.875000** | 0.829167 |
| GLM 5.3 vs Flash / 60 | **0.591667** | 0.616667 | **0.800000** | 0.683333 |

**Predeclared gate:** for every pair, MPCF BA ≥0.55 AND MPCF−shuffled ≥0.05 AND MPCF−prototype ≥0.05 AND MPCF>length-only. **FAIL**: matched loses to shuffled and prototype in all three, and to length-only in all three. This is a strong matched-vs-shuffled negative for this *hash-vector / SWEeper* implementation; it does not disprove all pretrained semantic embedding or causal paired-contrast designs. No reruns with changed k, ngrams, splits, sources, lengths, or weights are authorized from this outcome.

## What we actually learned

The surprising signal is **the dataset can identify its own labels from answer lengths**: 0.949153 for Luna/Sol and 0.829167 for Sol/Terra. This sharply raises the risk of confusing **generator identity with collection-channel and generation-length behavior**. It is NOT proof of intentional leakage or proof that served models lack intrinsic length preferences; actual requested token caps, prompt/system settings, sampling configuration and endpoint revisions are not fully recorded. Even if length is intrinsic in this source, the result is not evidence it transfers to Kaggle's unseen prompts/collection conditions.

The current MPCF residual direction has **no demonstrated independent matched-prompt benefit** under the chosen frozen representation: disrupting pair integrity was not worse, and simpler retrieval prototypes beat it. No full-stack strong-anchor same-row prediction outputs or independent source-matched panel were available, so **a Kaggle score change cannot be predicted or claimed**.

**Do not promote** MPCF-hash into the 0.668/0.681 third-party reproduction pipeline. Avoid turning the length finding into a public leaderboard heuristic or a new training objective on spent OOF. Instead revisit the *global* bottleneck: we need matched **same model and prompt across distinct recorded generation channels**, not just model A vs B on a single unknown channel. That crossed support is currently missing. The first next gate is inspect protocol/endpoint/source metadata for an actually controlled short-text paired reference cohort; without it, mark `BLOCKED_BY_IDENTIFIABILITY`, not "find another k".

## Research and competition separation

This study does **not** validate any 18-model scheme, does **not** evaluate against strong champion on common heldout labels, and does **not** improve the official Kaggle 0.668. As of the separately checked Kaggle authenticated ledger at 2026-11-10 16:29 UTC, existing submission **58167751** (Lightning published-0.681 frozen inference) remains `PENDING` with empty Public Score, while preview Kernel v2 is complete. Do not claim a score for it.

**No further training, Kaggle run/submission, GPU work or official data exposure in this experiment.**
