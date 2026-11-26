# C1 mechanism study — preregistered before first real 2699-row evaluation

Date: 2026-11-26 MYT. User goal: deliver an actual, score-worthy new official Name That LLM submission, not another fragile blend. Existing official Public BA = **0.718**. Candidate currently **NOT** authorized for scoring unless gates below pass.

## Single primary hypothesis
**H1:** The existing strong frozen 335M encoder's class evidence benefits from supervised cross-prompt same-generator contrast, with stronger hard negatives from the *same prompt across different generators*, beyond ordinary CE on the same representation.

**H0:** Same-prompt negatives mostly teach topical/source artifacts, or CE already captures the available signal; CE+contrastive does not improve held-out prompt-group 18-class BA.

## Frozen study (CPU-only, cannot submit)
- Model: `intfloat/e5-large-v2` 335M frozen, exact revision `d94960066016d2064f2a3a2dbc65107a2b48230d`, max 192 tokens, CPU, same as previously verified source-transfer E5. No pretrained encoder parameter updates.
- Data: **official Kaggle TRAIN only**, 2,699 responses, 150 `prompt_id` groups, 18 exact target labels. This avoids unverified external provider output training rights. **Historical/adaptively spent DEV only**; it is *not* independent hidden generalization.
- Exact fold strategy: `GroupKFold(n_splits=3)`, groups `prompt_id`, every prompt solely in train or evaluation in each fold; each response held out once. Input embedding computed once with frozen encoder and may not use labels.
- Same architecture in both arms: `Linear(1024,256) → GELU → L2 normalize → Linear(256,18)`, AdamW lr 0.002, weight_decay 0.01, 8 epochs, 5 entire prompt groups per batch, same starting seed `20261126+fold`, identical batch order. Only changed variable: coefficient of prompt-matched supervised InfoNCE `alpha=0` vs `alpha=0.2`.
- Contrastive loss: positives are same true model across **different** prompts; hard negatives are different models within **the same** prompt, denominator weight ×2; temperature 0.16. No hidden labels or test-text access.
- Primary metric: **18-class macro balanced accuracy** on the 3 prompt-held-out folds; paired same-row rescues/regressions; 1,000 bootstrap replicates resampling complete prompt groups, with seed frozen.
- Minimum useful effect / early kill gate: candidate BA **≥0.500**, paired BA gain **≥+0.025**, paired prompt-bootstrap 95% lower bound **>0**, rescues > regressions. This is a bounded model-mechanism gate, NOT an estimate of 0.718 Kaggle score superiority.
- Predeclared competing explanation: a *same-representation*, same-projector supervised CE control may match or beat contrastive. Such an outcome RETIRES this specific C1 mechanism immediately. Do not sweep alpha/epochs/dimension on the consumed folds or use Kimi-specific gates.
- External source comparison cannot yet be credibly established: previous `prvsiyan` pool overlaps some incumbent input, and API-derived external records need provider-output training rights. Do not claim clean source-disjoint evidence.
- No Kaggle notebook or competition submission; GitHub Actions CPU development only. Do not upload raw official data, embeddings, model heads or predictions to public GitHub. Aggregate scores/counts allowed.

## C1 launch criteria (not satisfied by this study alone)
A pass supports only the next **strong 4B/LoRA source-robust architecture** investigation, if compute and data rights permit. A new official competition submission additionally needs a complete 18-class model whose same-row decisions justify replacing an expensive stage of the 0.718 scorer, offline full-size real-runtime proof within 9 hours, successful `submission.parquet` contract tests and a fresh quota check. Never relabel low OOF BA as a new 0.718 competitor.

## Test-first engineering evidence
Local CPU four-unit suite originally FAILED because the contrastive implementation module did not exist (TDD RED), then PASSED after implementing the minimal module (TDD GREEN). Unit tests assert prompt-group no leakage, real positive gradient, hard-negative importance, matched initialization output shape, and explicit leakage rejection. Run the exact GitHub copy before model inference. The prior submission-file engineering contract had 10 green tests ([Actions #45751742419](https://github.com/kolihhan/name-kaggle/actions/runs/45751742419)), but that contract does **not** establish model quality.

Implementation:
- `scripts/ntl_c1_prompt_contrastive_20261126.py`
- `scripts/ntl_c1_official18_prompt_oof_20261126.py`
- `tests/test_ntl_c1_prompt_contrastive_20261126.py`
