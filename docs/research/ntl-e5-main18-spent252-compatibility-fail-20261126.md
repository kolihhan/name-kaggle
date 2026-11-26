# NTL frozen E5 vs existing main 18-class predictions — compatibility FAIL

**Observed 2026-11-26 MYT.** Result source: successful read-only/CPU [GitHub Actions #45748413705](https://github.com/kolihhan/name-kaggle/actions/runs/45748413705). **NO Kaggle competition submission, private kernel launch, GPU, model training, or incumbent modification.** Owner Public Best stays **0.718** (#58755304).

## Decision
**REJECT direct E5 Kimi K3 ↔ GPT 5.6 Luna switching as an 18-class augmentation.** The single frozen counterfactual reduced correct labels on the previously spent 252-row external panel **151/252 → 148/252**, net **−3** (4 rescues vs 7 regressions). The 252 panel is historically spent, so this is a safety/compatibility falsification only, **not fresh validation and not an official Public BA comparison**. No threshold sweep, follow-up tuning, or promotion against this panel.

## Validated sources
- Main model: actual completed *previously executed* Evgen-v36 0.718-lineage private diagnostic `kolihhan/ntl-fable-preservation-252-v2-20261125`. Its authenticated output `submission.parquet` had 252 unique `id`, `model` columns and SHA256 `1a5496f0f42c745a78ec37348828bf829310e9b64a34e8957e125e3a92b3f767`, original 151/252 correct.
- Publisher-labelled 252 true model responses: `nischaydnk/ntl-topic-synthetic-18llms/generations.csv`, exact SHA256 `bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c`, fourteen source/topic examples for each of 18 official classes. Source provenance is **publisher-declared, not independently verified**.
- Exact 252-output and public sample availability confirmed independently in read-only [Action #45748164667](https://github.com/kolihhan/name-kaggle/actions/runs/45748164667).
- Authenticated owner-notebook inventory [Action #45748293987](https://github.com/kolihhan/name-kaggle/actions/runs/45748293987) recovered this existing 252 private checkpoint and other *different-scoring-pipeline* outputs, but **did not identify an independently held 0.718-lineage 18-class prediction panel**. Absence in that limited name-search inventory is not proof that no other artifact exists.

## Fixed counterfactual protocol
- Encoder `intfloat/e5-large-v2`, revision `d94960066016d2064f2a3a2dbc65107a2b48230d`, CPU-only; same E5 readout as the previous 80-prompt source-transfer experiment.
- Binary Kimi-vs-GPT Luna class-centroid direction fitted from exactly **70 official train prompt pairs**, with no encoder fine-tune and no additional fitting on the 252 panel.
- The original 18-class prediction remains untouched unless its **predicted label**, not the truth, is exactly `kimi-k3` or `gpt-5.6-luna`. If eligible, use the fixed E5 binary margin to select either of those labels. No score-based gating, oracle label access, thresholds, or optimized blend weights. For all other original predicted labels, retain original label.
- Full 18-class labels and row IDs must agree with the archived source; strict invariants ensure no novel labels or orphaned IDs.
- Evaluate same-row paired prediction changes on the *spent* panel. Every decision change classified as rescue, regression, or still incorrect.

## Actual matched 18-class outcome

| Metric | Original | Fixed E5 Kimi↔Luna counterfactual |
|---|---:|---:|
| External 18-class correct | **151/252** | **148/252** |
| External balanced accuracy (14 rows/class) | **0.599206** | **0.587302** |
| Original predictions eligible for Kimi/Luna switch | — | 30 |
| Changed predictions | — | 13 |
| Wrong → correct rescues | — | 4 |
| Correct → wrong regressions | — | 7 |
| Wrong → another wrong | — | 2 |
| **Net rescues−regressions** | — | **−3** |

By true class:
- **Kimi K3**, n=14: baseline **1** correct → **5**, 4 rescues, 0 regressions.
- **GPT Luna**, n=14: baseline **12** correct → **5**, 0 rescues, 7 regressions.
- No other true class contributed a correctness rescue/regression under this exact gate. Two additional decision changes remained incorrect.

## Why this is decisive enough to STOP, but not a leaderboard result
1. The frozen E5 had source-transfer **70% binary BA** on external Kimi/GPT Luna but exactly matched a single em-dash feature; same-row E5-vs-em-dash correct-only counts were 30 vs 30. The preceding 10-style comparison showed E5 helped GPT, hurt Kimi on those source-specific responses. These studies are not consistent with a universal safe gate.
2. This older 252 panel now shows the reverse practical harm under the full 18-class baseline: E5 recovers Kimi but regresses more GPT Luna. Thus 2-class separation **does not imply** a positive 18-class switching policy even if the encoder contains real identity information.
3. Results belong to an experimental Fable-protected historical v36 lineage, **not** a verified bitwise copy of the owner's Kaggle hidden-test scored 0.718 run. Underlying publisher labels, model checkpoint source exposure, prompt distribution, and source shift remain caveats. We are **not** claiming the full Kaggle BA would decrease from 0.718 to any specific value.
4. The only previously known unspent realistic external cohort of 4,302 prvsiyan rows overlaps the current Open Pool by 173 Gemini 3.7 texts, and the incumbent already mounts multiple source datasets and prefit checkpoints. There is **no established independent same-row 18-class baseline on that cohort**, and GPU retraining or new Kaggle scoring would violate the current execution boundary.

## Next research decision
- Retire **E5 direct Kimi/Luna classifier switching**. Do not tune thresholds on the 252 samples, resurrect Confidence Fusion, start extra GPU runs, or submit Kaggle without fresh user authorization.
- The missing experimental capability is a *source-provenant, prompt- and generation-protocol-disjoint 18-class dataset with actual frozen incumbent SAME-ROW outputs*, plus a verified cost/label/license contract. Without this capability, claims of better leaderboard BA from any new expert are speculative.
- A different and genuinely independent research mechanism would need to address the underlying 18-class attribution evidence (e.g. true generator likelihood / tokenization), rather than repeating reweighted style features, frozen semantic embeddings, or expert confidence voting.

**Reproducibility:** `scripts/ntl_e5_vs_main18_spent252_falsifier_20261126.py` and `.github/workflows/ntl-20261126-e5-main18-spent252-falsifier.yml`. Executed output aggregate contains no raw response texts or truth row IDs.
