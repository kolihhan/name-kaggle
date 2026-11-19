# NTL Apex Forensic Result — 2026-11-19 (completed)

**Decision:** STOP interpreting residual-only logits as an 18-way classifier. The reliable result is a **structural family-mass invariant plus 23 potential typography true-label contradictions** in a *historically spent* external 18-class panel. This does **not** demonstrate 0.702 Apex regressions or license a scored rule change.

## Data, code and execution

- [Prereg: Apex evidence](../apex-forensics-20261119/PREREG.md); [prereg: typography mechanism](PREREG.md).
- [Terminal GitHub Action #44485246316](https://github.com/kolihhan/name-kaggle/actions/runs/44485246316) **SUCCESS**. It pulls the existing scored `kolihhan/ntl-0702-apex-pinned-nov18` source, verified **complete code SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`**, and executes only its original `pngy_residual.py` on CPU. Residual source SHA256 `4f4c943dcd098ab4e833a4108388697b932b12b2878dcb4c50c6911956d1d3a0`; scoring `apex_postprocess.py` fixed rule snippets verified against pinned source.
- Frozen official TRAIN `2,699` responses, 150 prompts, 18 labels; historical `Oleg216` selector yielded 216 distinct prompt groups, exactly 12 per class; pinned dataset SHA checked. Old strong `0.691` source's frozen matched hard predictions = **89/216**, BA **0.412037**. This is **not** the current `0.702` model's scores, and the panel was used many times previously.
- No new paid or external data search, no Kaggle GPU, no official Kaggle submission or hidden test access. Raw passages, IDs, logits and row-level truth remain only in ephemeral Actions memory; **only aggregates** logged.

## 1. True Apex Residual is family-local: measured mathematical invariant

Original `refine(z,evidence,families,strength=.1)` computes model-family-specific adjusted softmax vectors and re-embeds them using each **unchanged `logsumexp(z[:,family])`**. Four scored families: `claude` (5 labels), `gpt` (5), `gemini` (4), `glm` (2). `kimi` and `qwen` do not have a residual submodel.

A 32-row × 18-class reproducible numerical experiment on the exact imported `refine` verified:

- **Max absolute change in log family mass: `8.88e-16`**.
- All ungrouped class scores unchanged (`0` difference).
- This test demonstrates **score-level family-mass preservation**, not that final argmax cannot change families. `prompt_pair_adjustment`, post-refine argmax, GLM keep, and typography can still affect final labels.

**Correction to earlier crude proxy:** Directly taking `argmax` over a class-wise `evidence` matrix initialized to 1 generates huge pseudo-preference for unmodeled Kimi/Qwen classes. This is **not a legitimate global residual classifier**. The diagnostic standalone numbers `R=12/216`, `P0=13/216`, `P1=87/216` should be kept only as an example of an *invalid surrogate representation*, **not as evidence of actual Apex weakness, and not as grounds to promote a family veto**. Those scores are intentionally retired from scientific comparison with `0.702`.

## 2. Source-faithful typography contradiction audit — historical Oleg216

These rules use a **probability ×0.15 soft penalty** on disallowed classes, **only when the model proposed at that stage is disallowed**. Texts and known true classes allow detection of potential **true-label penalization**, NOT an actual scored prediction flip.

| Source rule | Rows containing marker | Rows where true class is excluded | Historical frozen strong pred correct on exposed row |
| --- | ---: | ---: | ---: |
| spaced em dash `" — "`: penalize all except Claude Opus 4.6/5 and GLM 5.3/Flash | 46 | **21** | **5** |
| right curly quote `"’"`: penalize Claude or GLM | 69 | 0 | 0 |
| any em dash `"—"`: penalize Gemini | 131 | 2 | 0 |
| **Union (nonadditive)** | — | **23/216 = 10.65%** | **5/216 = 2.31%** |

The **23 true-label contradictions** are distributed: Kimi K3 **8**, Claude Fable 5 **6**, Claude Sonnet 5 **5**, Claude Opus 4.8 **1**, Gemini 3.1 Pro **1**, Gemini 3.7 Flash **1**, Qwen3.8 Max **1**. At the original 0.691 frozen hard-prediction stage, **8/216** rows' predicted class plus text would meet at least one penalty rule. The exact final Apex label at the time of typography differs and is unknown.

Interpretation: the fixed `" — "` -> only four allowed labels assumption is **not universally true** across this corpus. Five known-correct historical labels are **potentially endangered if the current proposal remains correct when the rule applies**. **There is no evidence all five actually flipped**; the penalty is soft, and before-Apex 0.702 probability vectors are unavailable.

## Decision and no-go

1. **Reject the earlier cross-family veto as first choice.** Its premise was weak because the fitted residual already preserves per-family probability mass. Do not let bogus `argmax(evidence)` BA drive this direction.
2. **The simplest new falsifiable mechanism is typography intervention correctness**: does a hard-coded `0.15` penalty ever change a correct incumbent label to an incorrect one when the full Apex pipeline is run? Test must compare actual `before_typography` and `after_typography` predictions on exactly the same model-scored, labeled rows, retaining genuine 18-way logits/probabilities and fit provenance. A mere marker count is insufficient.
3. Need a future **authorized scored inference or an already-produced real stage-logits cohort** to test that outcome. Do not use fresh Kaggle GPU under the expired one-shot diagnostic authorization, and do not submit typography removal on the above marker contradictions alone.
4. Score remains official **0.702 #58514818**, target >=0.720. This is a substantive stage-level mechanism diagnosis, **not official score improvement**.
