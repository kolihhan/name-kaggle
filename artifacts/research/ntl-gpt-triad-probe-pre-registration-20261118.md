# NTL Frozen GPT sibling-triad experiment — before seeing scores

Date: 2026-11-18 MYT. Scoped experiment: **one exact mechanism** on pinned incumbent `#58375806` (0.691 Public BA). This is **exploratory** because the 216 Oleg rows have already been inspected and champion prefit overlap is not certified.

## Hypothesis and scope

- H1: an independently fitted GPT-family-only three-way stylistic classifier can correct current Lightning errors between **gpt-5.6-luna**, **gpt-5.6-sol**, and **gpt-5.6-terra**; H0: apparent confusion is caused by source/domain mismatch or label noise, so attempted correction causes as many regressions as rescues.
- Selection was motivated by already inspected Oleg216 confusion and cannot be described as a blind hypothesis. **One paired diagnostic on those exact spent 216 rows is for downselection/probe eligibility only**. No promotion to science claim.
- C0 = frozen predictions of the actual 0.691 Lightning inference on Oleg216, SHA-controlled 216 IDs, 18 true classes, 12 rows/class.
- C1 = C0 except if C0 belongs to the fixed three-model GPT trio and a prefit **three-class** `LinearSVC` confidence top-vs-second score gap >= **0.20**; then choose the SVC label. If its predicted label equals C0 or gap below 0.20, unchanged. All other C0 labels **bit-for-bit unchanged**.
- Fit uses **only official public competition train.csv**, not Oleg truth, hidden test, Public LB feedback or test-time labels. Fixed character TF-IDF `char_wb`, ngrams 3–5, `min_df=2`, `max_features=100000`, sublinear TF, `LinearSVC(C=0.5, class_weight='balanced', random_state=20261118, max_iter=3000)`. No hyperparameter selection from Oleg.
- Dataset audit must check official train row labels, actual training row count and exact duplicate normalized text across train and Oleg for the GPT trio; overlap -> abort, not remove post-hoc rows. Training uses only labeled 3-class official rows.
- Diagnostic is CPU-only, no competition submission or Kaggle private GPU; export only aggregate counts, no raw official passages, Oleg text, IDs, trained weights or predictions.

## User-delegated provisional decision gate — frozen before experiment

1. **Eligibility to consider exactly ONE official diagnostic `PROBE` submission:** Oleg216 18-class paired BA delta **>= +0.03**; 95% cluster/bootstrap confidence interval lower bound **strictly > 0** using 5,000 deterministic (seed 20261118) resamples of disjoint prompt groups; **net rescues > regressions**; per-true-label regressions explicitly disclosed; no outsider prediction changed. Invalid logits/provenance, incomplete classes, or any non-GPT prediction modification => FAIL.
2. This is **not a clean independent validation pass**. Oleg is already spent, source channel uncalibrated to Kaggle and n=12/class. A gate-passing candidate is at most a **probe**, not a proven improvement. Prefer an independent exact-model-ID data cohort if cheaply available. Do not claim hidden uplift from Oleg scores.
3. Before a probe *submission*, verify it runs as a **single new notebook** on 3-row and full hidden-sized synthetic input, with pinned all original Lightning assets, no internet, no external paid API, and <9h runtime. Freeze new code SHA, submit only once with description explicitly `PROBE`. Only if these checks succeed and Kaggle `competition_get_submission_limits` confirms >=1 available slot may submit. No daily slot spent before this gate.
4. Official numeric interpretation: new scored Public BA **>0.691** = a potentially stronger observed leaderboard package, **not causal proof**; **<=0.691** = stop this exact GPT-triad expert without Public-parameter tuning; blank score, evaluator error or PENDING = no comparison. Do not recycle a losing variant with after-the-fact threshold changes. Private result unknown.
5. If the new model cannot be packaged with original 0.691 lineage, per-sample inference or valid external-data rights, **NO SUBMISSION**, regardless of diagnostic BA. Do not replace original Lightning with a wholly different public notebook and call it an ablation.

This frozen threshold is a **delegated researcher operational gate**, not a universal statistical theorem or Kaggle medal line.
