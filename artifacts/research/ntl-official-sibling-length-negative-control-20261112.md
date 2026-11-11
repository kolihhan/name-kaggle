# Official 2,699-row sibling length negative control — 2026-11-12 MYT

**Result class:** real *descriptive* audit of already-consumed official 18-label training rows, **not independent model evaluation, classifier training, a new LB score or a Kaggle submission**.

## Data and audit integrity

- Source: existing repository `data/train.csv`, Git blob SHA `d892e62d0f9e13ce1619ceb9da349a1e6092320e`, via authenticated linked GitHub read only.
- Parsed CSV with quoted multiline-aware field parser, retained aggregate numeric counts only; no raw passages or row-level prompt identifiers in output.
- **2,699 rows**, **150 distinct prompt IDs**, **18 model labels**, **149 fully observed 18-class prompt panels**. One original label/prompt response missing.
- Word count: whitespace-delimited token count from passage text, not asserted to be the hidden-generation target length. Matched comparison counts model A longer, model B longer, equal, on prompts where both labels exist.

| Matched models (A vs B) | Prompts | A longer | B longer | Equal | Median A−B words |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Luna vs Sol | 150 | 74 | 75 | 1 | −1 |
| GPT-5.6 Sol vs Terra | 150 | 74 | 75 | 1 | −0.5 |
| Gemini 3.7 Flash vs 3.8 Flash | 150 | 90 | 60 | 0 | +12.5 |
| Gemini 3.6 Flash vs 3.8 Flash | 150 | 82 | 66 | 2 | +14 |
| GLM 5.3 vs GLM 5.3 Flash | 150 | 69 | 81 | 0 | −8.5 |
| Claude Opus 4.6 vs 4.8 | 150 | 71 | 78 | 1 | −6 |

Overall median whitespace-word counts, from GPT subset: Luna 104.5, Sol 111.5, Terra 100.5. **Despite different overall medians, their paired within-prompt length signs are almost perfectly split.** Do not promote global median as a stable sibling rule.

**Negative-control conclusion:** A deterministic sibling length ordering is unsupported on the existing official train prompt panels for GPT Luna/Sol/Terra. Historical SWEeper source reported Sol longer than Luna in 195/196 rows, while a distinct source with controlled requested words had only 20/47 supporting that direction (see `generator-vs-channel-identification-decision-20261106.md`). This is a source-specific feature warning, not a new classifier performance statistic.

## Re-check the global bottleneck without confirmation bias

The earlier `artifacts/domain-separability/RESULT.md` (2026-11-01) also **weakens a simplistic source-artifact explanation**: Phase 15 is less distinguishable from official than Phase 7 by the frozen within-label lexical domain diagnostic (0.5811 vs 0.7553 macro BA), yet Phase 15 improved repeatedly reused OOF and **fell** from 0.369 to 0.357 on Public. That is evidence of a dev/public mismatch, **not proof** that Phase 15 learned more obvious source artifacts or that a single source-adversarial objective will fix the hidden shift.

The 347-row 0.679 diagnostic found 71 of 87 misclassifications within provider, but its actual truth support is **12 of 18 classes**, not an independent whole-competition error breakdown. Generalizing these rates to missing GLM/Kimi/Qwen or hidden prompts would be unjustified.

## Research decision / resource discipline

1. **Do not use PPO**, length ordering, or a source/adversarial-loss patch as the next automatic experiment. There is no validated source-independent advantage from such a rule.
2. **Keep sibling contrastive STABLEPAIR conditional on documented data**. The existing frozen STABLEPAIR design already tests C0 pointwise vs C1 correctly matched prompts vs C-shuffled; do not invent another copy.
3. `scripts/stablepair_protocol_gate.py` now rejects claimed `served_model` that differs from exact `model` label (conservative until a canonical alias map is separately audited). This fixes a false-pair acceptance risk. The 10 synthetic tests include wrong-served-model, variant version, and different request keys.
4. **Do not overrestrict all future work to a pristine 18-label external holdout**: partial-label cohorts can support correctly labeled exploratory mechanism diagnostics when independent, source owner provenance and readout exposure are audited; they cannot certify a full 18-class official BA gain. Without such a cohort, mark the proposed supervised training **blocked**, not scientifically disproved.
5. A reasonable fresh mechanism hypothesis would be **source-robust within-family residual evidence that corrects an aligned strong model**, measured by rescued-minus-regressed rows on withheld prompt families; none has passed that gate yet.

**Current best last authenticated Public BA remains 0.679, attributed public-source reproduction `58210835`.** MGT and WeMM scores remain pending in the last saved official ledger; this audit did not requery Kaggle's private account.

No raw competition content, hidden test IDs or Kaggle credentials were copied into this report.
