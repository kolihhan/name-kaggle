# Claude Independent Blind Review — Round 1 (do not open the repository yet)

You are an external Principal ML Researcher and Kaggle competition strategist. Think independently. This is a **blind architecture review**, not a request to endorse the existing author's approach. Your task is to decide whether incremental improvements are still the best strategy or whether to replace the problem formulation, representation, learning objective, data program, or entire system.

## Information you're allowed for Round 1

- Kaggle **Name That LLM**: given an AI-generated response, predict which of **18 exact LLM identities** produced it; official metric is **macro balanced accuracy**.
- Public competition TRAIN contains **2,699 labeled responses, 150 known prompt groups**, all 18 classes. Hidden-test prompts are not the same as known training prompts. Do not assume IID random-row validation.
- Attributed outside resources may be allowed if public, legally usable, correctly labeled, and feasible offline; audit licensing and exact served model identity. No hidden labels, no private data, no paid inference APIs.
- An existing externally sourced multi-stage submission scored **0.702 Public BA** on this user's Kaggle account; target **at least 0.720**, ultimately medal-level hidden generalization. Its implementation **is intentionally withheld in Round 1** to avoid anchoring.
- Kaggle code scoring is offline with maximum **9h** runtime; local development can use bounded GitHub Actions CPU and owner-authorized Colab GPU for admissible external-data training. Private Kaggle exploratory GPU is restricted. Prompt-disjoint development on the existing training set **is allowed**, but historical folds are not a pristine holdout.
- A new perfect 18-label external validation set is **NOT a prerequisite for exploratory research** or a tightly controlled one-shot official score bet. It IS helpful for credible out-of-domain transfer claims. Do not reply only “collect more labeled data.”
- You have no current repo source code, implementation details, or trial-and-error history yet. State what you need rather than silently inventing it.

## Your independent assignment

1. **Model the data-generating process.** What observable signals can genuinely identify an LLM across unseen tasks, prompts, domains and serving versions? What apparently predictive signals are just sources/wrappers/formatting and likely to collapse?
2. Generate **four materially different solution families**, at least two of which are not ad hoc style rules, vanilla TF-IDF, a new posthoc threshold, or a simple extra expert vote. You may suggest pretrained discriminative encoders, generator-conditioned scoring, contrastive/domain-invariant learning, retrieval, calibrated mixtures, or something else—**but do not assume any listed example is the answer**. Explicitly consider whether exact model weights are obtainable for likelihood methods.
3. For each family, identify (a) differentiating hypothesis, (b) what data/weights are actually required, (c) compute and engineering cost, (d) expected failure mode, (e) smallest experiment that could falsify it with the existing training corpus, and (f) a sober uncertainty estimate. Keep evidence, estimates and speculation separate.
4. **Forced decision:** choose one of **INCREMENTAL**, **PARTIAL PIVOT**, **FULL ARCHITECTURE PIVOT**. Argue the strongest reason against your own choice. Explain what evidence would change your mind.
5. Design one **executably specific 48-hour research plan** with one main new mechanism, frozen control, exact data split that respects prompt IDs, score/diagnostics, stop/go rules, and feasible Kaggle 9h path. Do NOT invent a 0.702 out-of-fold score.
6. Explicitly say if a +0.018 Public BA gain is not plausibly predictable from the available evidence. Do not fabricate a predicted win.

## Required output format

- **Decision** (≤100 words)
- **Four competing methods** in a compact table
- **Single highest-value experiment** (command-level outline; what to measure and why)
- **Strongest falsifier and costs**
- **What additional facts would change this decision**
- **Top 3 questions for the owner** (only questions that genuinely block execution)

**STOP after Round 1.** Do not access repo, search our past conversations, or ask to see previous proposed fixes until you've written and committed to your independent solution ranking. We will provide the repository and negative results in Round 2.
