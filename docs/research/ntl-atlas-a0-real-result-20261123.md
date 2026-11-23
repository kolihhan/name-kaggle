# ATLAS (2026) A0 — ACTUAL REAL TOKEN likelihood readout result

**Experiment completed SUCCESS**: [GitHub Actions #45152458986](https://github.com/kolihhan/name-kaggle/actions/runs/45152458986), November 23, 2026. All steps completed, original equation smoke PASSED, cache versions/hashes matched, real evaluation finished.

### Research question

Does the exact robust bounded per-token **ATLAS argmax-prod** readout with fixed temperatures T = {1,2,4,8} improve already-computed GLM-5.3 / GLM-5.3-Flash same-token likelihood scoring, **without any additional model forward pass or paid API**?

This is **ONLY the original paper's readout**, not its trained source prefix/LoRA model and not a valid claim to reproduce the published MAABench result. It is a two-model GLM subset, *not* the NTL 18-way scored Champion.

### Frozen genuine data

- Real official NTL TRAIN **2,699** unique rows; **300** true GLM-5.3 vs GLM-5.3-Flash labelled rows, **150** prompt groups, preserved 5-fold groups.
- Actual Kaggle public historic token cache publisher versions 1 (800) and 2 (1899) recovered, with 8 exact original source file SHA256 pinned against previously successful [#45016238278](https://github.com/kolihhan/name-kaggle/actions/runs/45016238278). Exact full/flash passage IDs, prefix boundaries and every scored token identity checked. **0 new model inference**.
- Real response suffix token length min 20, median 105, max 131, no GLM–Flash token align failures. No output row text/ID/token data persisted.
- Cohort is repeatedly studied official TRAIN and **not prospective source transfer**, thus cannot certify unseen score improvements.

### Verified results — same 300 true labels

| Scorer | GLM 2-way BA | Correct / 300 | Recall GLM5.3 | Recall GLM Flash |
|---|---:|---:|---:|---:|
| C_mean: plain mean log-likelihood | **0.89333333** | **268** | 0.8266667 | 0.9600000 |
| **C_prod: original ATLAS multi-temp argmax-prod** | **0.91333333** | **274** | 0.8733333 | 0.9533333 |
| C_sign: raw aligned token winner fractions | **0.91000000** | **273** | 0.8866667 | 0.9333333 |
| **C_global: frozen historical global-statistics supervised 5-fold strong control** | **0.94333333** | **283** | 0.9400000 | 0.9466667 |

**True same-row paired differences**:
- C_prod **vs C_mean**: **14 changed labels**, **10 rescues**, **4 regressions**, **+6 net**, **+0.020000 BA**. 150-prompt grouped conditional bootstrap 95% delta: **[-0.0033333, +0.0433333]**; crosses zero. Fold accuracy differences by 60 rows: +0.0333, -0.0333, +0.0333, +0.0333, +0.0333.
- C_prod **vs strong C_global**: **21 changed labels**, **6 rescues**, **15 regressions**, **−9 net**, **−0.030000 BA**. Grouped conditional bootstrap CI **[-0.0566667, 0.0]**. Fold accuracy differences: 0, -0.05, -0.1333, +0.0333, 0.
- C_mean **vs strong C_global**: **−0.050000 BA**, 7 rescues, 22 regressions.
- C_sign **vs C_mean**: +0.0166667 BA, but zero-fit and not decisive.

### Scientific verdict

**A0 readout improves plain mean by +2pp in this historical 300-row GLM binary setting, but fails the stronger predeclared criterion of beating C_global (−3pp) and confidence bound does not establish nonzero gain vs Mean.** It should **NOT** replace our strong GLM class scorer and must **NOT** be blended into the official 0.715 18-way baseline. Also do not sweep temperatures, posthoc thresholds or use single-fold outliers to rescue it.

Do not claim full ATLAS has failed; paper's fundamental learned source-conditioned distribution was **not implemented**. That fuller method still requires source-specific LoRA and K conditioned model passes and conflicts with current user no-new-QLoRA and zero-additional-inference constraints. Keep it as literature inspiration, not a promoted code path.

### Related results that have also actually completed

- [POS syntax-order #45146166103](https://github.com/kolihhan/name-kaggle/actions/runs/45146166103) SUCCESS: source-shift old Oleg216 C1 0.125 BA vs historical strong 0.412037 BA, so weak standalone syntax not competitive. On official 2,699 OOF C1 0.16004 vs shuffled 0.10669 and POS bag 0.15335: real ordering signal but far too weak as a standalone replacement and no validated incremental 0.715 gain.
- [Prompt-SupCon synthetic engineering test #45149856087](https://github.com/kolihhan/name-kaggle/actions/runs/45149856087) SUCCESS: 18-class gradient/paired batch mechanics functional; real strong-embedding source availability still only **source-code paths**, no verified same-row external 0.715 real supervised embeddings and true labels. No measured real BA for SupCon.

**Official Champion remains 0.715 Public Balanced Accuracy. No new model forward, commercial API, QLoRA, GPU or Kaggle competition submission in the A0 experiment.**

Reproduce from [script](../../scripts/ntl_atlas_exact_token_readout_a0_20261123.py) and [workflow](../../.github/workflows/ntl-atlas-token-a0-real-20261123.yml).
