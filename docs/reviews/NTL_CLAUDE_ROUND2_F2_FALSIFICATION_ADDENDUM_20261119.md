# NTL — Claude Blind Round 1 F2 adversarial facts (2026-11-19)

This is provided **only after Claude's Round 1 was frozen**. It must not contaminate that independent diagnosis retroactively. The point is to test a falsifiable structural hypothesis, not to defend an incumbent plan.

## Exactly verified owner-snapshot facts

- [Github Actions #44532913111](https://github.com/kolihhan/name-kaggle/actions/runs/44532913111) **SUCCESS**, aggregate-only CPU census on SHA256-pinned `data/train.csv` `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`:
  - train: 2,699 rows, 150 prompts, 18 labels;
  - **149 prompts contain exactly one response for each of 18 labels; one prompt contains 17**; 0 duplicate (prompt,label) pairs; exactly 1 missing pair. Claude's arithmetic hypothesis was **correct for TRAIN**.
  - locally mounted `data/test.csv`: **3 fixture rows**, columns `id,text`, **no prompt_id**. This is a Kaggle example fixture, **not** the hidden rerun sample. It is incorrect to infer the hidden test contains only 3 rows or resembles 150 complete 18-response blocks.
- Official [Kaggle Data](https://www.kaggle.com/competitions/name-that-llm/data) and [Overview](https://www.kaggle.com/competitions/name-that-llm/overview/description): hidden run ~4,000 unseen-prompt responses, unlabeled `id,text` only; true hidden prompt grouping and one-model-per-group multiplicity **not disclosed**. The public/private model proportions differ and passages per class are not equal, according to the competition overview.
- Our [previous genuine blinded grouping CPU test](https://github.com/kolihhan/name-kaggle/actions/runs/43884199474): existing TF-IDF agglomeration pair precision 0.998056, recall 0.626870 on all 2,699 train rows; k=5 sparse subsamples recall 0.478667–0.500667. **Not an adequate full hidden prompt-ID reconstruction**, though high precision could still support very small partial-group ideas.
- Our [prior real historic frozen prediction collision test](https://github.com/kolihhan/name-kaggle/actions/runs/43895516290) flagged 13 of 51 historical wrong predictions (22 flagged responses), below predeclared 15-error go gate; older 0.691 system rather than current 0.702, spent TRAIN sample.

## Strong independent direct contradiction — third-party competition discussion

A Kaggle participant [Noussair Mighri, “Does the test set keep the one passage per model per prompt structure?”](https://www.kaggle.com/competitions/name-that-llm/discussion/751802) **reports** substantially the same F2 approach:
- train TF-IDF clustering approximately 0.95 ARI;
- train prompt-group Hungarian one-per-model assignment raises grouped OOF **0.374 → 0.481**;
- on official hidden Public, the **same idea falls 0.353 → 0.350**.
- Original poster **hypothesizes** test duplicates/multiplicity or other generator shift; this is **not proof** of hidden test group cardinalities. Their reported results are a participant's own public claim, not independently replicated on our 0.702 model.

**Interpretation:** A major F2 training-only boost plus Public regression has already occurred externally. Therefore Claude's 65% structural and 40% overall success-rate estimates cannot be treated as empirical frequencies. The exact `Hungarian + one-of-each` F2 should lose priority unless Claude supplies a DIFFERENT prediction not already contradicted by this negative result and a testable low-cost mechanism.

## Correct scientific action after this addendum

- **Round 2**: ask Claude to update its hypothesis ranking explicitly, not to declare that all group-based inference is impossible.
- Avoid 48 hours of Hungarian reimplementation without an argument why it avoids the published failure.
- If F2 is demoted, compare genuinely distinct alternatives (e.g. frozen/pretrained representation extraction versus source shift vs conditional likelihood), with actual usable weights/data/compute budgets and an early falsifier.
- Keep attributed 0.702 official score as frozen champion; no claims of OOF for it unless truly honest and available. No new training, GPU scoring or competition submissions were performed by this data census.
