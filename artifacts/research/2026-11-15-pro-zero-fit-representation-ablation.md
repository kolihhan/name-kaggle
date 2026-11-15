# PRO round 2 — stateless representation-class test (executed)

**2026-11-15. Real CPU experiment, not a new competition score.**

- Exact [GitHub Actions run](https://github.com/kolihhan/name-kaggle/actions/runs/43683503696) **SUCCESS**, script `scripts/ntl_zero_fit_prompt_disjoint_representations_20261115.py`. No model fit, no GPU, no Kaggle hidden inference, no new submission.
- Input: hash-checked official `data/train.csv`, **2,699 responses, 150 prompts, 18 true classes**. The training data have already been adaptively investigated. Matching prompt groups were completely excluded from each retrieval query's reference pool.
- Research question frozen in code before the Action result: Does zero-fit lexical retrieval retain model-specific information under unseen prompts, and how much of that comes from channel formatting?
- Same-row 1-NN control: fixed hashed character 3–5-grams. Alternatives: the same after punctuation/format stripping and fixed hashed word 1–2-grams. No IDF, nearest-neighbor reference-label fit, learned weights, tuning, test labels or prompt-specific thresholds.

| Frozen representation | Right / 2,699 | 18-class BA | Rescues vs raw char | Regressions | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw character 3–5-grams | 375 | **0.138938** | — | — | — |
| Format-stripped character 3–5-grams | 353 | **0.130780** | 56 | 78 | **−22** |
| Word 1–2-grams | 286 | **0.105959** | 170 | 259 | **−89** |

## Research decision

**REJECT all three as a stand-alone competitive model class.** Stronger than chance does not imply competitive. Raw char > stripped char under this fixed retrieval method, but it does not establish that punctuation *causes* attribution or transfers to new generation protocols. The significant number of word-only rescues is an oracle ceiling, NOT evidence for a deployable ensemble: it also regresses 259 rows.

This is a deliberate **representation-class falsifier**, not a tune-the-hash-size exercise. No more changing n-gram bounds, nearest-neighbor k or blend weights on the same spent data. Do not compare this BA with the official champion **0.683** as if they were the same evaluation. This result does **not** rule out trained neural encoders, conditional likelihood, model family features, or robust source-conditioned representations. Existing no-new-training and untouched 18-class same-row transfer rules still govern candidate promotion.

## Learning from Yuanzhe Zhou: evidence, not mythology

The [CMI 1st-place team writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution) explicitly reports grouped-by-subject five-fold validation, different input/architecture branches, and weaker individual models that improve ensembles through complementarity. The authors report member-specific contributions; do not attribute all code to Zhou personally. The [Santa 2024 3rd-place team writeup](https://www.kaggle.com/competitions/santa-2024/writeups/santamizers-3rd-place-solution) experiments with alternative candidate-generation operations and search paths. The transferable lesson is **change representation/search space and measure independent contribution**, not simply 'always add an ensemble' or 'always run tiny tests'.

## Next decision-relevant evaluation

- Verify status of already-created conditional GLM source/prompt paired experiments before spending further GPU. A private kernel being queued/completed is not evidence of model accuracy.
- For a genuinely new 18-label candidate, freeze an unseen, provenance-qualified full-class same-row panel against the exact 0.683 champion; report sibling, source, rescue/regression and cost. If no eligible panel exists, **do not publish a claimed 0.7** or tune on old folds.
- Gemma 4 agent research has a separate, coding-task-specific mechanism; do not transfer a text 1-NN finding to agent quality.
