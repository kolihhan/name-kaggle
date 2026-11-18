# BREAKTHROUGH_PLAN · Name That LLM
2026-11-18 · Phase A frozen at blind_review_A.md · append-only evidence; no rewrite of Phase A.

## Decision
[事实] Best officially observed Public BA **0.691** (#58375806); target **>=0.720** (+0.029). 18-class balanced accuracy. Host describes ~4000 hidden examples and ~40% public; **exact** public class denominators and paired SE unknown. An equal-size independent-row calculation gives single-model SE ~0.0116, not an observed paired SE.

[事实] The original same-prompt column-contrast CPU trial, 5-fold by prompt, raised BA 0.2804 -> 0.2848; shuffled-column comparison scored 0.2907. Matched-vs-row paired prompt CI95 [-0.0037,+0.0130]. It does NOT justify T4 training.

[事实] Independent author-conditioned character order CPU test [Actions 44283630240](https://github.com/kolihhan/name-kaggle/actions/runs/44283630240), commit 61f23a0, on frozen 2,699 rows / 150 unseen prompts / 18 classes: char unigram NB 0.230788 vs fixed char 1–3-gram NB **0.291892**, change **+0.061104**, prompt-bootstrap CI95 **[+0.042205,+0.080731]**, 394 rescues / 229 regressions. This isolates a real sequence signal *within a weak model*. It does not demonstrate marginal gain over the incumbent Public 0.691.

## Ranking / pre-mortem
1. **Character-sequence likelihood — real weak-model signal, no demonstrated marginal benefit.** [事实] A second grouped five-fold CPU experiment [Actions 44284622488](https://github.com/kolihhan/name-kaggle/actions/runs/44284622488) returned a discriminative char baseline BA 0.320783; char NB alone 0.291892. Three pre-registered blends: 25% char BA 0.321897 (delta +0.001114, paired CI95 [-0.005926,+0.008148]); 50% BA 0.321529 (delta +0.000746, CI [-0.010733,+0.012608]); 75% BA 0.313753 (delta -0.007030, CI [-0.022227,+0.008166]). [判断] There is no demonstrable independent benefit in this probe. Stop blend-weight tuning and do NOT spend T4 on this hypothesis alone.
2. **Strong independent member with honest prompt-disjoint OOF.** [判断] Compare multiple out-of-fold model probabilities only if every training stage excludes validation prompt, including external train sources. Failure: source shift makes CV and Public move in opposite directions; the earlier +0.0353 OOF / -0.012 Public example demonstrates this risk. Kill if there is no honest matching CV+Public pair. Do not re-audit invalid Lightning training predictions.
3. **Per-class decision calibration without public fitting.** [判断] With nearly equal class counts and class-conditional error, fixed training-only class calibration may alter marginal BA. Failure: little leverage because training priors already ~equal. Only test after a valid OOF source; lock any parameters before Kaggle inference.

[判断] No one candidate currently merits P(Public>=0.72) >15%; structural character-order value is demonstrated on weak CV, not on incumbent. No new T4 training or submission until the strong-member marginal gate can be evaluated. **The user controls acceptance thresholds.**

## Queue (one change per submission)
- **Completed / no submission:** char-order complementarity [Actions 44284622488](https://github.com/kolihhan/name-kaggle/actions/runs/44284622488), commit 9f65cc8. Best delta +0.0011 with CI crossing zero. Not promoted.
- **Conditional train:** only after a genuine independent lift against a credible strong member, freeze selected recipe and build an honest fold-independent inference source. Training: same competition 18 labels; never use in-sample Lightning pseudo-OOF.
- **Conditional Kaggle entry:** 3-line prereg: (i) exact added character-likelihood score, (ii) expected Public BA improvement as a hypothesis (not guarantee), (iii) predeclared decision rule. Verify daily quota; no repeated blend tweaking. Official submission ID required; no claim of >=2 SE 'breakthrough' absent confirmed paired evidence.

[事实] Official submit of any new NTL candidate: **not yet performed**.
