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

## Phase D continuation — prompt-invariant residual candidate (2026-11-18)
[事实] Published participant reports show one-to-one Hungarian assignment can improve official train prompt-held-out CV substantially yet reduce or flatten Public (e.g. [Noussair Mighri](https://www.kaggle.com/competitions/name-that-llm/discussion/751802), plus [independent write-up](https://github.com/satriadaffa/name-that-llm)). Official hidden passages have unequal per-model counts; train one-per-model grouping is not guaranteed in hidden test. Do **not** submit a hard one-per-prompt label quota.

[判断] Next candidate is **prompt-invariant residual style, without matching**. Same folded char TF-IDF+SVD representation, but subtract group centroid before fitting a second classifier. Infer evaluation prompt groups using *unlabeled* text-only word TF-IDF / agglomerative cosine clusters (distance 0.90), not hidden prompt labels; report oracle-groups diagnostics separately from deployable blind clusters. The critical negative-control is sparse five texts per prompt, to model hidden unequal coverage. Fix one raw/residual blend weight=0.25, no sweeps, no Kaggle submission or GPU.

[事实] Pre-registered implementation [scripts/ntl_prompt_residual_sparse_v1_20261118.py](scripts/ntl_prompt_residual_sparse_v1_20261118.py), commit a4c2d23. **GitHub Actions 44287492092 launched** by workflow commit 46cfb4c: [CPU evaluation](https://github.com/kolihhan/name-kaggle/actions/runs/44287492092). Run was in_progress at initial check. Artifact is `ntl-prompt-residual-v1.json`: raw BA; oracle residual BA (upper-bound diagnosis); blind-cluster residual BA and +0.25 blend BA; sparse k=5 blind blend BA, cluster ARI, paired prompt-bootstrap CI, rescues and regressions.

**Stop/advance:** If blind clustering or sparse condition cannot beat raw with a meaningfully positive paired interval, stop on this mechanism regardless of oracle boost. If it does, its transfer to an honest strong-model member remains required; never extrapolate weak proxy BA to current 0.691 Public without independent evidence. No new NTL Kaggle submission ID.
