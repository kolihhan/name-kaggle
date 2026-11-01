# Phase 14 — public sparse reconnaissance

Date: 2026-11-01

## Decision question

Can the strong sparse component behind the public `cosmorin/name-that-llm-public620-reproduction` system be turned into an honest local experiment by retraining it from scratch on the permanent grouped folds, using only a conservative provenance-audited subset of the public passage pool?

This reconnaissance does **not** treat the public notebook's leaderboard score as a local-CV result, and it does not reuse its fitted SVC or LoRA checkpoints for OOF evaluation.

## Public reference architecture

Static inspection of the public 0.622 reproduction showed an ensemble containing at least:

- a cached four-view TF-IDF + LinearSVC member;
- Qwen3-14B LoRA/readout heads;
- Qwen2.5-14B LoRA/readout heads;
- a DeBERTa-v3-large readout;
- downstream ensemble / Sinkhorn logic.

The cached SVC is a **final-fit artifact**, not OOF evidence. Its machine-readable manifest states it was fit on:

- official competition train: 2,699 rows;
- public pool: 75,783 rows;
- total: 78,482 rows;
- no test rows.

Because that SVC saw all 2,699 official training rows, it must not be used to score the same official rows as OOF validation.

## Exact sparse recipe recovered from public source

The original public notebook source and the reproduction manifest agree on four views:

1. `char`: char TF-IDF 3–5, max 250,000, weight 1.0;
2. `word`: word TF-IDF 1–3, max 180,000, weight 0.7;
3. `shape`: transform runs of letters to `X` / `x`, then char TF-IDF 2–6, max 60,000, weight 0.5;
4. `small`: preserve a fixed set of small/function words and punctuation, replace other alphanumeric words with `CONTENT`, then word TF-IDF 1–3, max 50,000, weight 0.5.

All four use:

- `lowercase=False`;
- `sublinear_tf=True`;
- `min_df=2`;
- `dtype=np.float32`;
- word analyzer token pattern `(?u)\b\w+\b|[^\w\s]`.

Classifier:

`LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")`.

The fixed `SMALL_WORDS` set was recovered directly from the public source and will be copied exactly into the Phase 14 implementation.

## Public pool audit

Raw source: `nishantdahal/ntl-handwriting-blend`, `pool_v4.parquet`.

Observed raw file SHA-256:

`37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`

Raw audit:

- rows: **75,783**;
- labels: **18 / 18**;
- normalized unique texts: **75,783**;
- normalized duplicate rows: **0**;
- normalized overlap with official training answer text: **0**;
- external prompt IDs: **8,278**;
- external prompt keys: **7,274**;
- direct string overlap between external and official `prompt_id`: **0**.

The direct `prompt_id` non-overlap is not proof of semantic prompt independence because official prompt text is unavailable in the competition train file.

## Source-level prompt provenance

The 75,783-row pool merges 25 public datasets. Their Kaggle dataset metadata was audited individually where available.

Strong prompt-provenance evidence includes:

- `seyit`: 1,000 newly written prompts, explicitly not competition training prompts, topic-similarity filtered;
- `willrice`: explicitly uses the Seyit public prompt bank rather than competition prompts;
- `hailey`: hand-written prompt bank, with topics filtered against community-reconstructed train topics and other public banks;
- `keshav`: 100 independently authored prompts; no official or reconstructed competition prompt sent to a model;
- `prvsiyan_v1`: 60 new prompts, screened against reconstructed train topics;
- `prvsiyan`: the consolidated pool contains exactly **231** `prvsiyan:n...` prompt groups, matching the upstream dataset's 231 quality-passing `new` prompt blocks rather than its dangerous `train_regen` split;
- `lucasboesen`, `abhinav`, `kaplinskiy`, `zbigniew`, and much of `zakir`: explicitly reuse the audited public prompt banks;
- `maniceet`: its Kaggle dataset page documents own/public prompt bodies filtered against competition train passages with word-TF-IDF cosine <= 0.25 and no train-text exact/near duplicates;
- `semenoff`: aggregation of audited public prompt-bank sources plus its own generations on the Seyit bank.

Some source metadata is blank or too weak to establish prompt provenance directly. Those sources are **not** admitted merely because they are in the 75k pool.

## Conservative subset rule

The frozen candidate subset is built without looking at permanent-fold scores:

1. seed only rows from the provenance-supported sources listed in the build workflow;
2. collect their exact consolidated `prompt_id` values;
3. allow rows from otherwise undocumented sources only when their `prompt_id` exactly matches one of those already-audited prompt IDs;
4. do not select by model, setting, classifier confidence, CV behaviour, or leaderboard performance.

Result:

- seed rows: **38,210**;
- safe prompt IDs: **3,736**;
- retained rows: **43,781**;
- retained labels: **18 / 18**;
- inherited-only rows: **5,571** from `alperen`, `dophuchung`, `abdullah`, and `jadkutbi`, all by exact safe-prompt identity;
- excluded rows: **32,002**.

Retained per-label counts range from **877** (`qwen3.8-max`) to **4,326** (`gpt-5.6-luna`).

The generated subset artifact SHA-256 observed in the audit run is:

`f646c8294007b05d550e1bf3b00f3f02db0abbcf195f2cc7c6c26b2bb0db368c`

## Residual uncertainty

The host does not publish official prompt text in `train.csv`, so semantic equality between an external prompt and an official hidden prompt cannot be mathematically checked. The conservative subset relies on source-level evidence that prompts were newly authored / independent / filtered, plus exact prompt-identity propagation only within the external pool.

This limitation is recorded rather than hidden. It does not authorize weaker source selection after seeing Phase 14 results.

## Next step

Freeze one Phase 14 experiment before looking at OOF:

- same permanent `GroupKFold(prompt_id)` mapping;
- official validation rows only;
- rebuild the 43,781-row external subset from the pinned raw-pool hash;
- retrain the exact four-view sparse recipe inside each official training fold;
- compare once against promoted Phase 7 (`0.3700521998508576`).

No Kaggle submission is authorized by this reconnaissance.
