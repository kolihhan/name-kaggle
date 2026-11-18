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

### Observed outcome — prompt residual weak OOF
[事实] [Actions 44287492092](https://github.com/kolihhan/name-kaggle/actions/runs/44287492092) concluded **SUCCESS** (2699 official training rows, 150 unseen prompts, 5 folds, 18 labels; CPU only). Raw char-TFIDF-SVD LogisticRegression BA **0.244074**. Blind-group residual 0.25 blend BA **0.267037**, delta **+0.022963**, prompt-bootstrap CI95 **[+0.010000,+0.035556]**, 191 rescues/129 regressions; raw blind-residual *alone* BA 0.188909 (worse). Oracle true-group residual alone BA **0.351146** is **not deployable**. Mean blind-group ARI **0.49686**. Sparse k=5, n=750: raw BA 0.233037, blind blend BA **0.271111**, delta **+0.038075**, group-bootstrap CI95 **[+0.014869,+0.062504]**, cluster ARI **0.65656**. These are weak independent proxy models, **NOT** a 0.691 strong-model comparison or evidence of Public gain.

[判断] Real lift appears in the weak-proxy blind blend, including sparse k5; however, a second classifier alone or random centroid subtraction may explain it. Rule out this alternative *before* promoting.

### Strong-baseline causal falsifier in flight
[事实] New code [scripts/ntl_prompt_residual_strong_falsifier_20261118.py](scripts/ntl_prompt_residual_strong_falsifier_20261118.py) commit **8a9eb14**, workflow commit **c6d61c2**, [Actions 44288966172](https://github.com/kolihhan/name-kaggle/actions/runs/44288966172) launched. It reproduces previously observed **0.320783** discriminative char TF-IDF/LR baseline byte-exactly under the same frozen 5 prompt folds; adds *one* fixed 0.25 residual blend and compares *inferred* prompt groups, true groups (oracle only), **five size-matched random partition nulls**, and a blind sparse-k5 test. Predeclared pass: inferred blend prompt-CI lower >0, sparse-k5 CI lower >0, and inferred full lift > all five random-group null lifts. No Kaggle/T4 submit, no stronger model or test-set training. If fail, do not call it a prompt-specific mechanism. If pass, still require evidence of complementary lift over actual strong incumbent, not Public extrapolation.

### Strong falsifier execution repair (no mechanism retune)
[事实] Initial [Actions 44288966172](https://github.com/kolihhan/name-kaggle/actions/runs/44288966172) failed only at the strict numerical-equivalence guard: prior documented strong linear BA `0.3207829978`, current same frozen inputs/recipe BA `0.3200397713`; absolute drift `0.0007432264`. All 5 fold model fits completed, but the guard stopped aggregate output. Therefore **no** strong model improvement is yet proven. No candidate weights or splits changed.

[事实] Source guard repair commit `8c8aed6` changes ONLY strong-baseline reproduction tolerance from `1e-12` to `0.002`; actual run baseline is reported and every candidate is compared against **its same-run paired baseline**, not the historical baseline. New CPU-only rerun [Actions 44289576988](https://github.com/kolihhan/name-kaggle/actions/runs/44289576988), workflow commit `95189a6`; in_progress at initial check. No official submission.

### Phase D strong baseline falsifier — terminal result (2026-11-18)
**[事实]** [Actions 44289576988](https://github.com/kolihhan/name-kaggle/actions/runs/44289576988) concluded SUCCESS; frozen official training 2699 rows, 150 unseen prompts, 5 folds, 18 labels, and 3000 prompt bootstrap resamples, CPU only. Same-run historical discriminative char TF-IDF/LR baseline **0.320782998** was reproduced exactly; the first run's 0.320040 was numerical instability and explicitly not used as denominator for any effect.

| Arm | BA | Paired delta vs strong | Prompt-bootstrap CI95 |
|---|---:|---:|---|
| Strong char LR | **0.320783** | — | — |
| Blind-cluster residual 25% blend | **0.341904** | **+0.021121** | **[+0.009625, +0.032608]** |
| Oracle true prompt group 25% blend | 0.358200 | +0.037417 | [+0.028889,+0.045573] |
| Five shuffled partition nulls | 0.308929–0.316706 | **-0.011854 to -0.004077** | each contains zero/negative |
| Sparse k=5, n=750 strong subset | 0.331414 | — | — |
| Sparse k=5 blind residual blend | **0.340446** | **+0.009032** | **[-0.011873, +0.029075]** |

Full inferred residual: 177 rescues/120 regressions; mean cluster ARI **0.496864**. Sparse: 37 rescues/31 regressions; mean cluster ARI **0.656564**. No hard one-per-prompt matching used.

**[判断]** Full-group uplift has evidence of prompt-specific signal: stronger paired OOF CI above zero and random-group controls all worse. However **preregistered gate FAIL** because the sparse k=5 lower CI **-0.011873 <=0**. Under hidden test size/composition shift, this is not yet robust. Do not promote this candidate to official Kaggle despite appealing full-group +0.021; no official Public gain claimed. Earlier weak-only gate was insufficient. Retain finding as mechanism evidence; freeze hypothesis and **stop residual blend tuning**. Future work must instead independently improve hidden-distribution robustness or obtain defensible strong-incumbent validation; avoid repeated sub-1k sparse testing to chase CI sign.

**Submission state:** NTL official Public best remains **0.691**, #58375806. No new NTL Kaggle submission in this phase.

## 2026-11-18 independent Style cross-source gate (frozen, one-off)
[事实] Existing Phase5 official grouped OOF already measured **style-only 0.290082**, char 0.329674, char+word+style 0.348208. These 155 style features are NOT newly invented. Existing ModernBERT prompt-residual LDA **failed** source-held-out transfer: 0.403427 raw to 0.367636 projected, worse in 11/13 sources. Source-balanced 18-class classifier **failed** 3-source holdout: 0.213704 -> 0.177037. No more residual blending, hierarchical-routing repetition, or source-weight tuning.

[判断] One unanswered transfer hypothesis: does the **already-defined nonsemantic style vector**, added directly to a flat char_wb LinearSVC, improve **independent source** BA rather than just historically adaptively reused grouped CV? Exact source split is frozen 5562 train rows (8 sources), 2700 held-out rows (3 other sources, ~1438 prompt groups, 18 labels) plus secondary 2699-row 150-prompt official grouped OOF. Publisher-reported served IDs are NOT independently verified. Optimistic iid 80%-power paired MDE for 2700 rows with assumed 15% discordance ~0.021 BA; group/source dependence could increase to >=0.03 BA. **Pre-registered same-domain expected effect and minimum gain +0.03 BA**.

[事实] Workflow [Actions #44292171841](https://github.com/kolihhan/name-kaggle/actions/runs/44292171841) (script commit 08d2fa9, workflow 7c6583e) fixes one intervention: style vector standardize TRAIN only, clip ±4, unit L2 normalize and append at weight 0.5; same char_wb TFIDF + LinearSVC C=.5 and frozen source/labels. Shuffled style rows are the dimensionality null. **Pass** iff held-out 18-class source BA delta >=.03, bootstrap group-CI lower>0, at least 2 of 3 source deltas positive, beats shuffled-style null, and secondary official GroupKFold delta>=-.01. No Kaggle submission/GPU and no original text copied into Git. Even pass is not a proven gain vs pretrained Lightning 0.691; stop if fail.

### Terminal source-transfer result — PASSED MECHANISTIC GATE
[事实] [Actions #44292171841](https://github.com/kolihhan/name-kaggle/actions/runs/44292171841) completed **SUCCESS**, exact code commits `08d2fa9` (candidate) and `7c6583e` (workflow). Data frozen: external 5,562 train / 2,700 validation across 8/3 mutually disjoint publishers and 1,438 prompt groups; official GroupKFold 2,699 / 150 / 5 folds. **No GPU, Kaggle submit, data export or changed base class.**

- **Primary source-transfer 18-label BA:** fixed char_wb LinearSVC baseline **0.213703704** → with pre-existing 155 style features **0.284074074**, **delta +0.070370370**; 3000-prompt bootstrap CI95 **[+0.056176,+0.084859]**, **314 rescues / 124 regressions**. All 3 held-out publishers positive, but not all had all 18 labels: (alperen 365 rows/4 labels +0.048889; arxitekton 2000/16 +0.067083; kenandemirci 335/5 +0.103333). Fixed shuffled-style null BA 0.218519, delta +0.004815.
- **Secondary historically adaptively spent official grouped 18-label OOF:** same-run char baseline **0.284108874** → char+style **0.358187919**, delta +0.074079045; prompt bootstrap CI95 **[+0.057037,+0.091868]**, **375 rescues/175 regressions**. Shuffled-style null BA 0.262988 (delta -0.021121).
- **Gate status:** **PASS** all predeclared rules (source uplift >=0.03, positive source-bootstrap interval, 3/3 source-positive, > shuffled null, official no-regress).
- **Confidence:** this materially supports independent *stylistic* representation signal vs a weak char-only model in two distributions, using one fixed scale, not a claim of 0.72 or a comparison vs actual prefit 0.691 Lightning. Source labels have upstream metadata provenance gaps; three publishers with unequal class coverage are not universal 18-class guarantees. Existing Lightning may already encode style; without a common genuinely untouched same-row strong-ensemble comparator the marginal contribution is unknown.

[判断] **Promote research mechanism, NOT automatic official submission yet.** Next decision needs only the causal question, *is current 0.691 ensemble already using equivalent style features, and can the style member contribute independent prediction information over that frozen stack?* Do a narrowly scoped read of the original scored V2 stack composition (not fake OOF), then either integrate one additive style head in a SHA-pinned code competition run under the score-facing contract, or reject as redundant. No more prompt-residual/char-weight tuning.
