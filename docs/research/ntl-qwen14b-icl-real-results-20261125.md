# NTL Qwen3-14B frozen GPT-triad ICL — first COMPLETE real GPU result (2026-11-25 MYT)

## Authenticated actual run
- Submitted one PRIVATE Kaggle kernel `kolihhan/ntl-qwen14b-gpt-triad-icl-private-20261124` via [Actions #45430337038](https://github.com/kolihhan/name-kaggle/actions/runs/45430337038), source QA and GPU quota passed, no training, no competition submission.
- Independent **read-only post-run aggregate download** authenticated via [Actions #45437807909](https://github.com/kolihhan/name-kaggle/actions/runs/45437807909) at 00:30 MYT November 25. Private Kaggle worker **COMPLETE**, output `ntl_qwen14b_gpt_icl_aggregate.json` downloaded exactly. This is an executed real large pretrained 14B study, not a planned experiment.
- Exactly **486 forward calls** = **162 unique labelled three-model texts × 3 arms**. Median 3.127 s/forward; p95 3.807s, longest input 1427 tokens, wall time 1430.3s (~23.84 min), modeled 4,000 test rows **one arm 3.47h** before existing strong 0.718 baseline's own full inference. All three arms full 4k = 10.42h by this naive estimate, >9h contest limit. These are measured costs for the diagnostic; actual integrated runtime still unverified.
- Qwen3-14B (pretrained, NF4 inference only). Official labelled 72 rows = 24 prompt groups × 3 GPT models; Beroli C1 publisher-reported labelled 90 rows = 30 prompt groups × 3. Unseen official prompts vs 3 training exemplars; zero, true matched-prompt triplet ICL, and mixed-label placebo. Official holdout is train-spent, external C1 is earlier-used source and its served-model ID is unverified.

## Exact source-stratified three-class balanced accuracy

| Domain | ZERO | PAIR true examples | PLACEBO mixed examples | PAIR - ZERO | PAIR - PLACEBO |
|---|---:|---:|---:|---:|---:|
| Official heldout 72 | 0.291667 (21/72) | **0.458333 (33/72)** | 0.388889 (28/72) | +0.166667 | +0.069444 |
| External Beroli C1 90 | 0.333333 (30/90) | **0.322222 (29/90)** | 0.388889 (35/90) | −0.011111 | −0.066667 |

- Official PAIR-vs-ZERO: **23 rescues / 11 regressions**, 42 changed predictions; prompt-group bootstrap 95% **[+0.069444,+0.277778]**.
- Official PAIR-vs-PLACEBO: **9 rescues / 4 regressions**, 20 changed; prompt-cluster bootstrap 95% **[0,+0.138889]**, touches zero.
- External C1 PAIR-vs-ZERO: **21 rescues / 22 regressions**, 65 changed; 95% **[−0.111111,+0.088889]**.
- External C1 PAIR-vs-PLACEBO: **9 rescues / 15 regressions**, 40 changed; 95% **[−0.166667,+0.022222]**.

Per-true-label three-class recall:

| Model | Official ZERO | Official PAIR | External C1 ZERO | External C1 PAIR |
|---|---:|---:|---:|---:|
| GPT 5.6 Luna | 0.708333 | **0.416667** | 0.766667 | **0.166667** |
| GPT 5.6 Sol | 0.083333 | **0.416667** | 0.133333 | **0.266667** |
| GPT 5.6 Terra | 0.083333 | **0.541667** | 0.100000 | **0.533333** |

## Diagnosis — separate facts from causal hypotheses
**FACT:** PAIR meaningfully changes predicted class allocation; Sol and Terra improve after exemplars, while Luna degrades substantially in BOTH domains. Thus positive Official overall is partially a rebalancing of initial Luna-heavy predictions, not a universal sibling-discrimination improvement.

**FACT:** External PAIR's net improvement fails: −1/90 overall, 21 rescues and 22 regressions; true-label C1 BA is not a clean independent provider-verified result. The main competitive gate was positive +0.08 BA and PAIR superiority versus placebo **in both** cohorts — it was NOT met.

**FACT / limitation:** `PLACEBO` as actually submitted has rotations +1,+2,+0 across 3 demonstration triplets, meaning **one of three triplets is correctly labelled**. Treat comparisons as partially corrupted demonstrator evidence, NOT a fully permuted negative control. Registered before output at [execution caveat](ntl-qwen14b-icl-execution-and-control-caveat-20261124.md). Also exact choice-token probability uses A/B/C and a fixed chat rendering: possible letter/token preference is not independently measured.

**H1 plausible (not proven):** Current exemplar prompt contributes primarily **label-prior calibration**, not robust generator-version identification. This predicts strongly unbalanced prediction histograms and a comparable effect from correct-label format with unrelated example text. Needs out-of-sample controlled test distinct from the already viewed C1/OFF cohorts.

**H2 plausible:** Real source/protocol shift changes prompt, sampling, formatting or provider routing. This predicts that at least some model-version fingerprints rank differently when matched by same genuine provider request settings. Requires provider-verifiable traces; existing C1 is self-reported.

**H3 plausible:** The 14B generic proxy's **A/B/C one-token readout** is inadequately calibrated for the task; revised scoring with explicit full model-name token likelihood or answer-position-balanced controls could alter results. This is a distinct hypothesis and must be frozen BEFORE testing independent new data. Do not patch same seen 162 rows and report it as generalization.

## Decision
**`DIAGNOSE / RETIRE_SPECIFIC_RECIPE` for the exact 3-exemplar ICL recipe as a promoted competition candidate, not `REJECT_SIBLING_DISCRIMINATION`.** Positive Official result establishes model sensitivity to examples but 0 or negative external generalization. With full 4k single-arm 3.47h *before* incumbent 0.718 model inference, substantial runtime/cost barriers to integrating it even if calibrated.

No 18-class score measured, no same-row paired comparison versus 0.718, no official Kaggle submission, and no 0.760 or Top-10 claim. The user requested no small-model training; none ran.

## Next critical experiments, strictly budgeted
1. Build **one independent prompt-free three-class external real-provenance test** if accessible, using pretrained model's log-likelihood or original-generator evidence rather than arbitrary 14B answer letters; verify label/version against provider responses. If provenance is unavailable, label source-only result as exploratory and explicitly block numerical promotion.
2. CPU-only inspect **pretrained strong 0.718 scorer's model-specific logits and runtime** once full private Fable252 finishes; currently still **QUEUED** in [Actions #45437807909](https://github.com/kolihhan/name-kaggle/actions/runs/45437807909). Do not launch duplicate expensive GPU inference or train more models.
3. Only promote a new method if it delivers **net rescues without unacceptable regressions** on same-row actual 18-class incumbent predictions, survives sources, and fits 9h full hidden inference runtime. Existing PAIR-zero delta on a three-class triad is categorically NOT a leaderboard delta.
