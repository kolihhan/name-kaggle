# Trainable Prompt-Contrastive Encoder — fixed Fold-0 feasibility (2026-11-24)

## Why this test now
The current best authenticated Public BA = 0.718 (submission #58755304, external Evgen Notebook reproduction), **not an original method**. Real PACE, SCaL and counterfactual patch retrieval failed same-data competitiveness tests. The prior ModernBERT prompt-residual LDA route reduced cross-source accuracy for 11/13 datasets. Archived Prompt-SupCon source trained only a small head on *frozen embeddings*; it does not test a fine-tuned representation.

### Hypotheses
- H1: learning a context-specific representation directly for 18-way attribution, with explicit same-model-across-prompts positives and same-prompt-different-model hard negatives, improves over ordinary CE under prompt/source shift.
- H0: adaptation learns prompt/collection artifacts or loses sparse style evidence; contrastive learning hurts or does not transfer. Existing CE may be as good as the contrastive candidate.

### Frozen one-variable intervention
- Backbone: `FacebookAI/roberta-base` revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` with fast tokenizer; first 10 transformer blocks frozen and their contextual outputs cached once; **last 2 blocks + 256-dimensional mean-pooled projection/classifier trainable**. This is genuine encoder adaptation, not simply frozen-feature SupCon.
- Identical head, frozen starting weights, examples, prompt-pair batches, epochs, random seeds, optimizer and input length for both arms. Difference: **A CE-only**, **B CE + 0.2 × supervised contrastive loss** (temperature 0.15), positives only from **same model in different prompt** within each paired minibatch; same-prompt other-model examples are explicit negatives. No per-model hard-coded routing or inference-time prompt IDs. Neither arm sees validation source labels during training.
- Official TRAIN SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`; permanent fold SHA256 `94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d`. Frozen Fold0, 2159 train, 540 heldout; 18 true classes, 150 prompt groups original. No folds changed.
- Paired prompt/label sampling: every train prompt is an anchor, partner is a distinct train prompt, common exact labels shuffled into groups of up to six. Full train prompt coverage each epoch. Each CE/contrast arm uses the **same** deterministic 2-epoch paired batches; explicit 2-prompt/≥2-label invariant. Max length160; float16 frozen encoding and mixed-precision last2 training; batch up to 12; AdamW LR 3e-5 trainable blocks, 1e-4 projection/classifier; global max grad norm1.0. No training on external panels.
- C1 dataset `beroli/ntl-benchmark-c1-generated-panels` 360/12/30 prompts SHA `8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177`. C2 dataset `beroli/ntl-benchmark-c2-generated-panels` 470/12/50 prompts SHA `bf25ef1e953d5c37b177f27b841a04098ae9b35f45fc43fa11036d91e8831e02`. Both from SAME publisher; not independent first-party served identity and not full18. All 18 classes remain possible predictions.
- Control D: exact existing `build_fold_features` + `make_svc` Strong Sparse trained on the same official 2159 rows, evaluated on the **same** three panels. No model ensembles, rank tweaks, retraining excluded datasets, paid APIs or hidden-test access.

### Numeric evidence and stop rule (frozen before labels evaluated)
- Record 18-way BA on official Fold0, 12-true-class BA on C1 and C2 (no narrowing 18 prediction labels), per-class recall, paired rescues/regressions, prompt-cluster bootstrap CI, training and inference wall-clock/CPU/GPU, truncation and input/resource integrity.
- Qualify for *further independent research* only if B improves A official Fold0 by ≥0.02 BA, improves A on BOTH external panels by ≥0.02 BA each, pooled external prompt-group bootstrap lower bound >0, AND B reaches or exceeds same-row Strong Sparse on official Fold0 and average of external C1+C2 BA. These gates are intentionally high given prior negatives. Failed candidate stops; no posthoc LR/lambda/epoch search or blind official submit.
- A separate stage would be required for 5-fold and fully independent verified 18-source transfer, actual final hidden inference wall-clock, licensing and legal provider ID before any competition submission. A valid first fold is *development*, not the current 0.718 champion's hidden BA.

### Runtime and safety
- One PRIVATE Kaggle T4 Notebook, only official and already public publisher datasets mounted. Internet used solely to obtain pinned public MIT RoBERTa weights; do not export official texts, labels, features, logits, model weights or private assets to public GitHub. Aggregate metrics only.
- CPU GitHub Actions prepares the one-shot Kaggle Notebook and submits **only the private research kernel**; no official Kaggle competition submission and no Scheduler.
- If private job runs out of time/GPU memory, records missing, or pretrained weights unavailable, report infeasible. Do not silently turn an evaluation on 3 preview rows into scientific evidence.

**Originality boundary:** Prompt-contrastive supervised training is existing prior art, not a novel research claim. This is a serious test of whether a trainable representation is worth pursuing; an original architecture should be proposed only after a distinguishing mechanism survives this control.
