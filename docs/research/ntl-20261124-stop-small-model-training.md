# NTL research decision: STOP small-model training (2026-11-24)

**User instruction: do not train more small models for Name That LLM.** This supersedes proposals to try DistilGPT2, RoBERTa-base, ModernBERT-base, frozen encoders with learned small heads, or another small QLoRA fine-tune in this competition. It does NOT apply to unrelated Text-to-SQL / GAIA projects. No scheduled jobs, new small training or automatic public Kaggle submissions are authorized by this file.

## Verified evidence, not estimates
- Authenticated Kaggle ledger, [read-only Actions #45393551650](https://github.com/kolihhan/name-kaggle/actions/runs/45393551650): owner best finished **0.718** submission **58755304** (Evgen public-code reproduction); Sanpier V3 **0.717** submission **58811345** (reproduction); original Rescue-per-Token **0.714** submission **58808007**, worse than owner's best. None of these establishes an original GOLD gain.
- RoBERTa last-two-layer trainable **CE vs CE+Prompt-Contrastive**, private actual Kaggle worker COMPLETE, same read-only aggregate:
  - Official Fold0 (18 classes): CE **0.2796296**, Contrastive **0.2555556**, strong Sparse **0.3833333**.
  - C1 (12/18): CE **0.25**, Contrastive **0.2694444**, Sparse **0.3416667**.
  - C2 (12/18): CE **0.2068995**, Contrastive **0.1955562**, Sparse **0.270307**.
  - Pooled external prompt bootstrap interval B−CE **[-0.0136194,+0.0219886]** includes zero. Frozen pilot pass gate FALSE.
- Previously tested SCaL, PACE, A0/A1 ATLAS and CPR v0 did not beat strong same-row controls or demonstrate full-18 independent source transfer. Treat existing stopped contracts as stopped, not as phase invitations.
- Data caveat: official folds repeatedly spent; C1/C2 from same publisher with 12 class support and self-reported served-model identity; neither substitutes for independent 18-class verification.

## Next target: LARGE-MODEL-INFERENCE-ONLY original evidence
- **Research objective:** improve 18-way hidden attribution above **0.718**, without training any small or large new model as a default. Base candidate on existing pre-trained high-capacity models and real likelihood experts already mounted in the strongest scored pipeline; do not start from a bare small encoder classifier or reproduce another public notebook.
- **First intervention class:** source-sensitive conditional likelihood/ranking from large native generator signals, not yet another GLM-only priority heuristic (the last one scored 0.714). Analyze which source identities current large models can *genuinely* distinguish, and use evidence at the 18-class decision level, not a 300-row GLM sibling surrogate. Do not claim an uplift until measured.
- **Must check before new official submission:** same-row relevant validation, actual model-specific rescue versus regression, runtime <= Kaggle cap with hidden-set scaling margin, source/version pinning, reproducible exact change to incumbent code and full 18 legal class schema. No leaderboard-tuned threshold searching, no overclaims from 3-row preview. Where strong-model full18 validation is missing, record this limitation and make a bounded experimental decision explicitly.
- **Restriction:** no small-model training for NTL. **WHY:** repeated trained small models underperform even simple Sparse across official and external comparisons; costly experimentation is not transferring. **PIVOT TRIGGER:** only a new explicit user request to permit training, or otherwise a documented large pre-trained inference-only pathway. **STATUS:** ACTIVE per owner request.

This records an immediate decision, not evidence that a new inference-only model is already implemented or scored.
