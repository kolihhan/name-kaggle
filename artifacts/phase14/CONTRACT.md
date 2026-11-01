# Phase 14 — conservative public 4-view SVC contract

Date: 2026-11-01

## Question

Does a full 18-class sparse classifier trained with a conservative 43,781-row public external corpus beat the promoted Phase 7 system under the existing permanent grouped validation, when the public sparse recipe is frozen before evaluation?

This is a **single permanent-fold shot**. No result-dependent tuning is allowed after this contract is committed.

## Frozen validation

Use the existing permanent validation unchanged:

- official rows: **2,699**;
- official prompt groups: **150**;
- labels: **18**;
- validation: 5-fold `GroupKFold(prompt_id)`;
- fold fingerprint: `bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d`;
- metric: balanced accuracy;
- scored rows: official validation rows only.

No external row is ever scored.

Reference promoted system:

- Phase 7 balanced accuracy: **0.3700521998508576**.

Also reproduce/report the canonical official-only Phase 5b score for context when available, but Phase 7 is the promotion threshold.

## Frozen external pool

Raw source:

- Kaggle dataset: `nishantdahal/ntl-handwriting-blend`;
- file: `pool_v4.parquet`;
- required raw SHA-256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`;
- expected raw rows: **75,783**.

The experiment must rebuild the conservative subset mechanically from the raw pool. It must not train from the public final-fit SVC cache or pretrained LoRA/DeBERTa assets.

### Frozen seed sources

Exactly these source labels are provenance seeds:

- `abhinav`
- `hailey`
- `kaplinskiy`
- `keshav`
- `lucasboesen`
- `maniceet`
- `prvsiyan`
- `prvsiyan_v1`
- `semenoff`
- `seyit`
- `willrice`
- `zakir`
- `zbigniew`

For `prvsiyan`, assert before training:

- exactly **231** unique `prompt_key` values;
- every retained `prompt_key` starts with `prvsiyan:n`.

This prevents upstream reconstructed-train-prompt splits from silently entering through source drift.

### Frozen inheritance rule

1. Retain every row whose `source` is one of the seed sources.
2. Compute the set of exact consolidated `prompt_id` values in those seed rows.
3. Retain a row from any other source only when its `prompt_id` is exactly in that seed prompt-ID set.
4. Make no model-specific, setting-specific, score-based, confidence-based, or fold-result-based selection.

Expected rebuilt subset:

- rows: **43,781**;
- unique prompt IDs: **3,736**;
- labels: **18 / 18**;
- seed rows: **38,210**;
- inherited-only rows: **5,571**;
- excluded raw-pool rows: **32,002**.

Expected retained label counts:

- `claude-fable-5`: 1,193
- `claude-opus-4-6`: 1,352
- `claude-opus-4-8`: 1,364
- `claude-opus-5`: 1,240
- `claude-sonnet-5`: 1,350
- `gemini-3.1-pro-preview`: 1,409
- `gemini-3.6-flash`: 2,880
- `gemini-3.7-flash`: 1,704
- `gemini-3.8-flash`: 2,778
- `glm-5.3`: 3,240
- `glm-5.3-flash`: 4,236
- `gpt-5.5`: 3,525
- `gpt-5.6-luna`: 4,326
- `gpt-5.6-sol`: 3,879
- `gpt-5.6-terra`: 3,875
- `gpt-6-astra`: 3,044
- `kimi-k3`: 1,509
- `qwen3.8-max`: 877

Any count drift is a contract failure, not a reason to adjust the filter.

## Frozen text transforms

### `shape`

Replace every contiguous run of Unicode letters with:

- `X` when the run starts uppercase;
- `x` otherwise.

Digits, punctuation and other non-letter structure remain.

### `small`

Tokenize with `\w+|[^\w\s]`.

Keep a token lowercased when it is in the exact fixed small-word set below, or when it is not alphanumeric. Replace every other alphanumeric token with `CONTENT`.

Fixed small-word set:

`a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little`

## Frozen feature views

For each fold, fit all vectorizers only on:

- the four official training folds; plus
- all 43,781 frozen external rows.

Transform the untouched official validation fold separately.

Views, in this exact order:

1. `char`: original text; analyzer `char`; ngrams 3–5; max features 250,000; view weight 1.0.
2. `word`: original text; analyzer `word`; ngrams 1–3; max features 180,000; view weight 0.7.
3. `shape`: `shape` transformed text; analyzer `char`; ngrams 2–6; max features 60,000; view weight 0.5.
4. `small`: `small` transformed text; analyzer `word`; ngrams 1–3; max features 50,000; view weight 0.5.

Every TF-IDF vectorizer uses:

- `lowercase=False`;
- `sublinear_tf=True`;
- `min_df=2`;
- `dtype=np.float32`.

For word analyzer views, use token pattern:

`(?u)\b\w+\b|[^\w\s]`

Multiply each sparse view matrix by its fixed view weight before horizontal concatenation.

No normalization or additional style features are added beyond what `TfidfVectorizer` itself performs.

## Frozen classifier

Fit exactly:

`LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")`

No tuning of `C`, class weights, feature caps, view weights, ngram ranges, preprocessing, sample weights, or source composition is allowed after this contract is committed.

## OOF output

Produce one prediction for each of the 2,699 official rows, preserving:

- `id`
- `prompt_id`
- true `model`
- `fold_id`
- Phase 14 prediction

Report:

- overall balanced accuracy;
- five fold balanced-accuracy values;
- all 18 per-model recalls;
- row/ID/null invariants;
- exact external subset counts used in every fold.

## Promotion rule

Promote Phase 14 only if all are true:

1. final OOF balanced accuracy is greater than **0.3700521998508576** by more than numerical equality (`atol=1e-12`, `rtol=0`);
2. all 2,699 official rows have exactly one OOF prediction;
3. validation rows contain no external rows;
4. raw-pool hash and frozen external subset counts exactly match this contract;
5. all model/feature/classifier settings exactly match this contract.

Otherwise reject this mechanism as tested.

A marginal numeric tie is a rejection, not a promotion.

## One-shot consequence

After the result is visible, do **not** tune on these permanent folds:

- seed-source membership;
- inherited prompt-ID rule;
- external sample counts;
- per-label balancing or resampling;
- SVC `C` / class weight;
- view weights;
- ngram ranges or feature caps;
- the `SMALL_WORDS` list;
- preprocessing or post-hoc label overrides.

A rejected result requires genuinely new evidence or a separately justified mechanism frozen before another permanent-fold evaluation.

## Kaggle rule

This contract authorizes only the local permanent-fold experiment. It does **not** authorize a second Kaggle submission.
