# NOW — Name That LLM
Last reviewed: 2026-11-18 · Decision: STOP proxy experiment sprawl

## Actual goal
**Official Kaggle Public Balanced Accuracy >= 0.720**, with an authenticated score receipt and reproducible, legal, offline <=9-hour submission. Do not substitute local validation or green CI.

## Facts
- **Current best: 0.691**, official submission **#58375806**. No new NTL submission or confirmed score improvement from the November 18 experiments.
- The 0.691 Lightning pipeline is an **attributed prefit public ensemble**, not an original model that can be honestly scored as 5-fold OOF on official training data. Its fitted artifacts already declare official-row exposures. It already contains **stylegram / stylometry**.
- Style+char vs weak char-only model: source-held-out BA **0.2137 → 0.2841** (+0.0704); **not comparable to 0.691** and likely partly redundant with incumbent.
- Prompt residual fails sparse-group gate; source-sensitive char pruning **0.2841 → 0.2737** (−0.0104). **Neither is a submission candidate.**
- There is **no credible 0.72 challenger today**. Missing: a non-redundant strong full-stack candidate and credible same-row generalization evidence. Full-18 independent served-model provenance remains incomplete.

## Single next objective: ONE strong, score-facing challenger
Stop optimizing char/style/residual proxy models. **Compare a substantively different, legally usable full 18-label inference/training architecture against the existing 0.691 pipeline**, not a new weight or prompt rule. Use available vetted public pretrained assets or owner-approved public-data training (Colab T4; CPU Actions only) while keeping all competition rules and model constraints intact. Require complete hidden-size-safe inference, SHA-pinned assets, offline <=9h runtime and clear mechanism. For any novel candidate with no genuinely independent strong paired cohort, be explicit about that uncertainty rather than fabricate champion OOF.

**End point:** ONE authenticated, rules-compliant Kaggle submission when quota/7-day probe rule allows and candidate differs materially. Record exact Kaggle ID and real Public score. If no viable challenger, report BLOCKED — no more retrospective proxy experiments, YAML workflows, manifests, source surveys or score claims as substitutes.

## Keep history without working from a 341 KB status log
- [Pre-Ponytail NOW.md (full historical record, read-only Git commit)](https://github.com/kolihhan/name-kaggle/blob/50dfd6fc53bb7cb68d3e0b3eec6ed9ea4085c633/NOW.md)
- [Evidence, negative results and scientific gate](BREAKTHROUGH_PLAN.md)
- [Competition rules and permissions](RULES.md) · [Evaluation policy](EVALUATION_POLICY.md)
