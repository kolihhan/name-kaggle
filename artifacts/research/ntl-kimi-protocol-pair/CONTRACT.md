# Kimi K3 reasoning-protocol invariance — frozen same-prompt matched-pair test

Date: 2026-11-11 MYT. **PREREGISTERED BEFORE champion predictions.** Intended user goal: meaningful independent attribution improvement, without new training or repeated Public-LB tuning.

## Mechanism and falsifiable prediction

The target Kaggle task attributes exact generating **model identity**, not the sampling configuration used at generation time. A model identifier ideally stays stable when *the same model serves the same prompt* under different reasoning effort. A model attribution classifier relying on generation-protocol artifacts can produce very different identities for `high` versus `off`.

The observed public 0.679 champion and existing own A1 are not guaranteed protocol-invariant. The test asks whether the frozen **0.679** champion's recall of **Kimi K3** materially changes between high- and off-reasoning examples drawn from matched prompts. This is **descriptive, not causal**: served-model metadata can match without actual version immutability across generation dates and additional protocol fields may change.

## Data and frozen sampling

Source `willrice/name-that-llm-openrouter-harvest` from Kaggle, October 2026. `calls.jsonl` contains 12,308 raw calls; authenticated metadata screening from [Actions #42972737341](https://github.com/kolihhan/name-kaggle/actions/runs/42972737341) found **1,000 Kimi K3 prompt IDs with both `high` and `off`**. This is only *preliminary* matching on `(model,prompt_id)`, not proof of identical request conditions.

Before any champion inferences, eligible pairs must satisfy ALL:
- `model == kimi-k3`, exactly one `high` and one `off` for that `prompt_id` in `calls.jsonl`.
- Both `text` nonempty strings; `finish_reason == stop`; `served_model`, `served_provider` and `model_id` equal across the two.
- `target_words` equal, non-null, and `n_words` 40–250 on both responses; do not claim that response-length equality is enforced. Record residual length difference separately.
- No duplicate response texts or duplicated selected prompt IDs. Sample deterministic first **144 pairs** ordered by SHA-256 of `"ntl-kimi-v1:"+prompt_id` as a fixed, label-agnostic selection; **288 passages** total. If fewer than 144 eligible pairs exist, stop — do not change thresholds or selection to create a positive result.
- Preserve complete original source provenance, including the file SHA256 and documented license/attribution. Do not download raw competition train/test to GitHub or print data text in GitHub logs.
- If public dataset terms disallow private research copying, stop before private input construction. Do not release raw passages.

## Scoring and controls

1. Reuse exact own scored `0.679` private Notebook `kolihhan/ntl-ozertuu-0683-guarded-no-train` and all three no-supervised-training guards (Gaussian, Atlas, `--train-final`). No learned weights, fitting, threshold selection, routing, or ensemble changes.
2. Pair texts as new `id,text` file in ephemeral/private Kaggle Notebook only. Do **not** include setting, prompt or true model in model inputs. Retain private matching `id,prompt_id,setting,true_model` file for evaluation after inference. Use original 18-label output without limiting candidates to Kimi.
3. Primary: **matched Kimi K3 recall** `high` versus `off`, count `(high correct, off wrong)` and vice versa, signed paired recall difference and exact two-sided McNemar/binomial discordance test. Secondary: predicted-label switch rate, most common off/high wrong predicted models, provider-family confusions and residual length/word-count differences. Include both numerator/denominator for all rates.
4. Also verify in each pair that actual served-model metadata and target words match. Check actual text hash uniqueness/overlap where accessible. Do not call this a guaranteed causal experiment because response collection may differ in time, request text/parameter settings, and provider revisions.
5. **Decision rule predeclared:** Only if the high/off recall delta is at least 15 percentage points, direction is stable across four deterministic SHA-order blocks, and exact discordance `p<0.01` may we conclude **material protocol sensitivity** (not new model success). Otherwise deem evidence weak/mixed. Regardless of result, no Kaggle score promotion, no new sibling expert or protocol-specific routing from these inspected examples. Before any modification to the champion, require a separate 18-class source/protocol-controlled validation set covering the six absent labels and independent same-row rescues/regressions.
6. No new GPU training or competition submission. One private frozen Kaggle inference job if preflight passes; no repeat to choose a favorable score.

## Strategic fork

This test distinguishes **protocol signature** from **model identity** for Kimi, one of the six labels missing in the previous 347-row external test. It complements earlier Gemini/GPT sibling confusion but does not replace the need for true 18-class validation or directly establish a path from 0.679 to 0.700.
