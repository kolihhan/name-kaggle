# Conditional source-balanced risk — executed negative result

Date: 2026-11-08 MYT. **Decision: REJECT this frozen treatment.**

The one-shot CPU experiment completed in 147.83 seconds. Source-balanced
training did not improve over ordinary training or pass its shuffled-source
control threshold. No new competition notebook or submission was made.

## Framework and controlled variable

Use the existing four rich Phase14 TF-IDF views and the same 18-head LinearSVC.
Only per-row training loss weights change. B uses unit sample weights; S
equalizes each observed source's sample-weight mass within each true model
label; P permutes S's exact weights within labels. At inference all arms see
only passage text. No generation, evaluator LLM, GPU, encoder training or
test-time source label is involved.

This tests a learning-objective hypothesis; it is not a novel attribution
architecture or an isolated explanation of the public ensemble's strength.
The preregistration deliberately did not inherit or weaken STABLEPAIR's
failed matched-protocol data gate.

## Provenance and execution

- Preregistered contract commit: `dc5f603ce733f3c3d53c56e4a193f5c20a207149`.
- Evaluated executable commit: `1e8fe58b4dd1f0c6a4e94b5f5ea4c397e5610864`.
- Branch: `research/attribution-bottleneck-review-20261106`.
- Original contract SHA256: `4740a7f6c4e2a3311d4c962b6fbba68e2544f9e7f41fdf4f956dcafa2a239b17`.
- Evaluated script SHA256: `e3ccd0e5614afc8907505c53d3ca0770c3dea67781b805ada40fc31042075728`.
- Public pool SHA256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`.
- Aligned scratch prediction archive SHA256: `7ba537d7fd758e632fbcd34fcafe9483f13960a29c58e6c43470a626b798fc2d`.
- Aggregate execution evidence: [aggregate.json](aggregate.json).

Training: 30,133 external rows and all 18 labels. Evaluation: 3,324 rows,
3,321 eligible for source/class primary metrics. Features: 505,436.
Prompt-ID, prompt-key and normalized-answer overlaps: all zero. No extra
answer-overlap rows needed removal. Vectorizers fit training text only.

The three fits converged in B/S/P respectively 34/52/38 iterations, taking
29.70/35.38/32.38 seconds. Weights, classifier settings, source membership and
feature caps were not tuned. S and P had identical effective sample size
10,629.33, versus 30,133 for B; identical weight range 0.223934–23.723077.

## Scientific results

These are source-dependent eligible-label BAs under an 18-way classifier.
The source label sets differ. Their mean is the predeclared diagnostic
statistic, not Kaggle's 18-label Public metric.

| Source | Eligible labels / rows | B ordinary | S source-balanced | P shuffled | S minus B |
| --- | ---: | ---: | ---: | ---: | ---: |
| abdullah | 8 / 705 | 0.486693 | 0.475989 | 0.485969 | -0.010704 |
| semenoff | 11 / 1,889 | 0.423697 | 0.418615 | 0.402777 | -0.005082 |
| zakir | 8 / 727 | 0.484960 | 0.478827 | 0.478574 | -0.006133 |
| Unweighted mean of sources | — | **0.465117** | **0.457810** | **0.455774** | **-0.007306** |

S minus P: **+0.002037**, below the required +0.005. Paired prompt-cluster
bootstrap: 672 clusters, 2,000/2,000 retained replicates; 95% interval for
mean S-minus-B **[-0.022724, +0.007424]**.

| Frozen continuation condition | Observed | Verdict |
| --- | --- | --- |
| Mean S−B >=0.010 | -0.007306 | FAIL |
| Mean S−P >=0.005 | +0.002037 | FAIL |
| Every source S−B >=-0.010 | Minimum -0.010704 | FAIL |
| Paired 95% interval above zero | [-0.022724, +0.007424] | FAIL |
| At least 1,900 bootstrap replicates retained | 2,000 | PASS |

On eligible rows S rescued 152 B errors and regressed 151 B-correct rows.
That net row count does not imply improved balanced accuracy: labels and
sources have different metric weights. Source-averaged macro provider error
and sibling error both increased. Exact per-label recalls and paired counts
are retained in the aggregate JSON; aligned 18-way margins/predictions are
in the scratch archive, not published passages.

## Interpretation and decision

**Verified:** this exact reweighting treatment failed every effect-size and
uncertainty promotion condition. All three source point estimates fell.
**Not established:** a statistically certain adverse effect; the interval
includes zero. Nor does this reject all robust objectives, demonstrate an
intrinsic representation ceiling, or establish causal protocol invariance.

The previously examined sources are disjoint from training in this run but
are consumed project evidence, not fresh independent confirmation. Missing
model/source cells and incomplete generation-protocol provenance remain.
The source-shuffled control preserves the weight distribution but cannot
make observational source labels into randomized protocols.

Close this objective/representation treatment. Do not adjust weight clipping,
source sets, regularization, seeds or blending in response to these results.
Do not run official OOF or submit this candidate. S's small advantage over P
does not establish useful residual information against the 0.668 anchor.

Generic StyleDistance/LUAR reuse was reconsidered after this result. The
repository had already researched those methods. The independently reopened
[competitor primary report](https://github.com/dcw06/NameThatLLM/blob/main/docs/glm-pair-qwen.md)
tests adding matched external training passages to Qwen+StyleDistance, not
adding StyleDistance itself. Its negative augmentation result therefore
**does not reject the style representation or measure its marginal value**.
Using that as evidence against the encoder would be a category error.
The literature can motivate a separately contracted representation test;
adding an existing component alone is not an architectural contribution or
a solution to the missing strong-anchor validation. No additional
style-encoder fit/inference was run, and this risk-weighting result does not
select which representation should come next.
See [earlier mechanism review](../contextual-choice-attribution-decision-20261106.md)
and [source/protocol evidence](../generator-vs-channel-identification-decision-20261106.md#8-independent-source-search-and-read-only-matched-sibling-check--2026-11-07).

No scientifically eligible replacement submission was found in this cycle.
A credible competitive promotion still needs a predeclared materially new
mechanism, admissible independent transfer and aligned fold-safe strong-anchor
outputs/training provenance. Own LoRA24 OOF cannot substitute for the 0.668
system. This experiment supplies no positive evidence for a larger evaluator.

## Engineering and independent review

Command: `python -m pytest tests/test_conditional_source_balance.py tests/test_phase14_public_sparse.py tests/test_two_stage_lora24_gate_a.py -q`.
**20 passed.** Scope covers risk-mass preservation, shuffled-weight control,
cross-source prompt/text exclusion, balanced-error decomposition, paired
prompt bootstrap and the reused feature/split machinery. No full-repository
GPU/training suite was run.

A fresh reviewer inspected the whole new change blind to scores and found
one documentation issue, independently checked against the pinned sklearn
source: multiclass one-vs-rest LinearSVC's balanced class weights equalize
positive budgets per binary head, not total true-label risk across all heads.
The contract wording was clarified; the original frozen commit/hash remain
above. **No estimator, data, weight, threshold or prediction was changed and
no scientific rerun was needed.** No other critical/important issue was found.

## Official Kaggle benchmark, freshly checked

[Authenticated read-only history 42393822063](https://github.com/kolihhan/name-kaggle/actions/runs/42393822063),
2026-11-08 00:56:59 MYT: 14 total submissions; no newly submitted candidate.

| System | Exact submission | Official Public BA |
| --- | --- | ---: |
| Ozertuu reproduction V3 / Apex V1 | 58022340 / 58045266 | **0.668 / 0.668** |
| Lightning V47 reproduction | 58008463 | 0.666 |
| Frozen V47 length40 | 58078151 | 0.663 |
| Own LoRA24 | 57911893 | 0.520 |
| Short-TRACE standalone | 58093864 | 0.093 |

All six exact records are COMPLETE; private scores remain blank. V47
length40's completed score is within its original inconclusive band, and
its stale PENDING entry has been corrected separately. The new CPU
diagnostic is **not** a new Kaggle score or a comparison against these
whole-system Public values.
