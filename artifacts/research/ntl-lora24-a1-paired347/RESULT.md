# Paired independent LoRA24 A1 — 347 fresh-source evaluation result

**Verdict: REJECT as a replacement or global verifier.** Date 2026-11-11 MYT. Real Kaggle private GPU inference and paired diagnostic [Actions #42968916125](https://github.com/kolihhan/name-kaggle/actions/runs/42968916125), `source-preflight` and `gpu-inference` both **SUCCESS**. Private inference ref `kolihhan/ntl-a1-paired347-inference` reached `COMPLETE`, and both private prediction parquets were compared by exact `id,source,prompt_id,true model` alignment on a temporary runner. No competition submission or training.

## Frozen scientific definition

[Predeclared contract](CONTRACT.md) before any A1 candidate predictions: Compare the **existing own Stage-1 A1 8B LoRA24** (frozen external Stage1 checkpoint) with the stronger own **0.679 Public champion** (submission `58210835`, protected attributed Ozertuu 0.683 public source reproduction). Do not confuse Stage1 checkpoint with own Stage2 `0.520` submission.

Private public-generated texts: Toan 107 rows / 9 prompts, CC0-1.0; Nimitha 240 rows / 20 prompts, CC BY 4.0; the same 12 observed true labels, with unrestricted 18-label predictions. Their source-provenance and byte SHA are locked in [CONTRACT.md](CONTRACT.md).

## Actual paired results

| Measure | 0.679 champion frozen | Own A1 independent |
| --- | ---: | ---: |
| Correct / 347 | 260 | 218 |
| Wrong / 347 | 87 | 129 |
| 12-observed-label conditional BA | **0.749795** | **0.628387** |
| Exact prediction disagreements | — | 117 |
| Rescued champion errors by A1 | — | **24** |
| Regressions (champion correct, A1 wrong) | — | **66** |
| Net rescues − regressions | — | **-42** |

| External source | Champion correct | A1 correct | Rescues | Regressions | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| Toan / authored | 82/107 | 68/107 | 5 | 19 | **-14** |
| Nimitha / public prompt bank | 178/240 | 150/240 | 19 | 47 | **-28** |

- 19/24 rescues were on sibling misclassifications.
- For Gemini provider: rescued **12**, regressed **19**.
- For GPT provider: rescued **7**, regressed **25**.
- Independent Gemini 3.8 Flash recall **44.83%** (13/29) vs prior frozen champion **27.59%** (8/29); this is class-level evidence only, not evidence that post-hoc routing would improve 18-class BA.
- `pass_sourcewise_rescue_gate = false`, **REJECT** A1 as a global replacement/verifier on these observed cohorts. Do not combine, weight-tune, adjust thresholds, or submit as a correction.

## What the result proves and what it does not

**Proof:** There is some independent information (24 corrected champion errors), but a direct switch is worse by 42 predictions across 347 and loses on **both** independent data-generation sources. The strongest frozen independent own checkpoint tested here cannot serve as an unconditional verifier. No path to 0.700 is established.

**Does not prove:** that all sibling-specific methods fail; that a principled independent verifier couldn't be trained or designed; that any Gemini-only gate would transfer; or that this 12-of-18 cohort predicts Kaggle Public/Private. Targeted Gemini gain was noticed after inspecting the 347 samples and thus **these data are spent for model selection**. The provider-distribution and prompt routing differ from official Kaggle; no exact 18-target fresh holdout exists in this test.

**Next global pivot:** stop blanket blending/replacement. Seek a **mechanism with genuinely new independent conditional evidence** (e.g., direct causal serving-protocol identification, non-incumbent prefit model likelihood with actual model-level attribution, or an independently sourced 18-class attribution corpus covering missing GLM/Qwen/Kimi/3.7 classes). Analyze the existing private paired predictions read-only for class-specific rescue vs regression to clarify research priority, **without fitting any gate from these data**. Explicitly differentiate old official grouped OOF, external proxy-source experiments and the scored hidden Public leaderboard.

## Complete no-text aggregate JSON

```json
{
  "by_source": {
    "nimitha": {
      "both_right": 131,
      "both_wrong": 43,
      "candidate_conditional_12_target_ba_18_predictions": 0.625,
      "candidate_correct": 150,
      "candidate_per_label_recall": {
        "claude-opus-4-6": 0.55,
        "claude-opus-4-8": 0.4,
        "claude-opus-5": 0.95,
        "claude-sonnet-5": 0.7,
        "gemini-3.1-pro-preview": 0.8,
        "gemini-3.6-flash": 0.55,
        "gemini-3.8-flash": 0.35,
        "gpt-5.5": 0.7,
        "gpt-5.6-luna": 0.65,
        "gpt-5.6-sol": 0.7,
        "gpt-5.6-terra": 0.25,
        "gpt-6-astra": 0.9
      },
      "candidate_wrong": 90,
      "champion_correct": 178,
      "champion_wrong": 62,
      "disagree": 85,
      "paired_net": -28,
      "paired_regressions": 47,
      "paired_rescues": 19,
      "provider": {
        "claude": {
          "n": 80,
          "regression": 18,
          "rescue": 4
        },
        "gemini": {
          "n": 60,
          "regression": 11,
          "rescue": 9
        },
        "gpt": {
          "n": 100,
          "regression": 18,
          "rescue": 6
        }
      },
      "rescue_from_same_family_incumbent_error": 16,
      "rows": 240
    },
    "toan": {
      "both_right": 63,
      "both_wrong": 20,
      "candidate_conditional_12_target_ba_18_predictions": 0.633102,
      "candidate_correct": 68,
      "candidate_per_label_recall": {
        "claude-opus-4-6": 0.777778,
        "claude-opus-4-8": 0.333333,
        "claude-opus-5": 1,
        "claude-sonnet-5": 0.888889,
        "gemini-3.1-pro-preview": 0.375,
        "gemini-3.6-flash": 0.555556,
        "gemini-3.8-flash": 0.666667,
        "gpt-5.5": 0.777778,
        "gpt-5.6-luna": 0.333333,
        "gpt-5.6-sol": 0.333333,
        "gpt-5.6-terra": 0.666667,
        "gpt-6-astra": 0.888889
      },
      "candidate_wrong": 39,
      "champion_correct": 82,
      "champion_wrong": 25,
      "disagree": 32,
      "paired_net": -14,
      "paired_regressions": 19,
      "paired_rescues": 5,
      "provider": {
        "claude": {
          "n": 36,
          "regression": 4,
          "rescue": 1
        },
        "gemini": {
          "n": 26,
          "regression": 8,
          "rescue": 3
        },
        "gpt": {
          "n": 45,
          "regression": 7,
          "rescue": 1
        }
      },
      "rescue_from_same_family_incumbent_error": 3,
      "rows": 107
    }
  },
  "candidate": "existing own Stage1 A1 Qwen3-8B LoRA24 checkpoint (NOT Stage2 0.520)",
  "champion_ref": "58210835",
  "competition_submission": false,
  "fresh_holdout_claim": false,
  "novel_model_claim": false,
  "original_kaggle_champion_public_ba": 0.679,
  "paired": {
    "both_right": 194,
    "both_wrong": 63,
    "candidate_conditional_12_target_ba_18_predictions": 0.628387,
    "candidate_correct": 218,
    "candidate_per_label_recall": {
      "claude-opus-4-6": 0.62069,
      "claude-opus-4-8": 0.37931,
      "claude-opus-5": 0.965517,
      "claude-sonnet-5": 0.758621,
      "gemini-3.1-pro-preview": 0.678571,
      "gemini-3.6-flash": 0.551724,
      "gemini-3.8-flash": 0.448276,
      "gpt-5.5": 0.724138,
      "gpt-5.6-luna": 0.551724,
      "gpt-5.6-sol": 0.586207,
      "gpt-5.6-terra": 0.37931,
      "gpt-6-astra": 0.896552
    },
    "candidate_wrong": 129,
    "champion_correct": 260,
    "champion_wrong": 87,
    "disagree": 117,
    "paired_net": -42,
    "paired_regressions": 66,
    "paired_rescues": 24,
    "provider": {
      "claude": {
        "n": 116,
        "regression": 22,
        "rescue": 5
      },
      "gemini": {
        "n": 86,
        "regression": 19,
        "rescue": 12
      },
      "gpt": {
        "n": 145,
        "regression": 25,
        "rescue": 7
      }
    },
    "rescue_from_same_family_incumbent_error": 19,
    "rows": 347
  },
  "pass_sourcewise_rescue_gate": false,
  "scope": "same-row already-inspected 12-label diagnostic, NOT full 18-class Kaggle BA",
  "supervised_training": false
}
```

**Privacy:** Only aggregate counts entered the public repository. Per-row `id,source,prompt_id,true/predicted` remains in private Kaggle kernels; no raw completion text, tokens, credentials, competition official data or model checkpoint artifacts were committed.

**Official Kaggle competition champion remains 0.679, submission `58210835`.** No candidate score promotion.
