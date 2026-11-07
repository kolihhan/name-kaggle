# Pretrained model-signature geometry — frozen model-level study

Date: 2026-11-08 MYT.
Status: FROZEN BEFORE ANY NEW ENCODER PREDICTIONS.
Branch: research/attribution-bottleneck-review-20261106.

## Global decision and three competing hypotheses

CURRENT BEST: reproduced Ozertuu Public 0.668; own LoRA24 Public 0.520.
REAL EVIDENCE: own aligned grouped LoRA24 OOF BA 0.5467611235396471,
with same-family and cross-provider errors, plus the official whole-system scores.
MAIN FAILURE: existing evidence does not identify rich transferable signatures
of exact modern siblings; learned semantics, source artifacts and model identity
remain entangled. Source-balanced sparse risk at befcb1c is CLOSED, not retuned.
MODEL-CLASS LIMITATION: hypothesized, not established by the small CPU failure.

Compare the following BEFORE implementation:
- H-R (representation learning): a reconstruction-trained, semantic-factorized
  style encoder can retain model-identifying choices that generic semantic
  pooling misses. LUSR freezes a semantic encoder during paraphrase inversion
  pretraining. Its public RoBERTa-base checkpoint predates this competition.
  Risk: useful idiosyncrasies also occur in content; human-style recovery and
  binary detection do not prove exact-18 attribution. This differs from
  prompt-residual LDA, which removed useful information, and generic
  StyleDistance reuse is not by itself an original contribution.
- H-M (objective / model family): multi-level GENERATOR/AUTHOR contrastive
  pretraining learns a source geometry that transfers to unseen exact labels.
  A nonparametric class-conditional density can preserve multiple reference
  modes lost by one mean/prototype or a single closed classifier boundary.
  SELECTED FIRST: DeTeCtive pretrained source encoder, evaluated against its
  SAME unsupervised SimCSE backbone and against centroid scoring. Existing
  model-conditioned contrastive priors are materially different from our
  frozen generic encoder + prompt-residual LDA and weak prototype v2. Main
  risk: old generators' geometry does not separate these 2026 siblings.
- H-C (conditional attribution / generalization): source-conditioned
  generative likelihood or matched-request ranking can separate identity from
  prompt/channel nuisance. ATLAS already exists at family level in V47.
  Exact-18 source-conditioned weights are not verified available; STABLEPAIR's
  strict protocol gate failed. No standalone GPU pilot, local GPU training or
  paid API is allowed; CPU learning is not generally prohibited.
  Not selected: an 18-way generative test is currently less executable than
  testing a task-relevant existing source checkpoint; its restrictions are
  not turned into a permanent ban on independent candidates.

Why H-M is strongest now: prior directly trains source relationships; checkpoint
and original backbone are released, dated and comparable; exact-label adaptation
can be tested with only labeled reference texts. It tests a different model
inductive bias rather than a sparse reweighting or a larger surrogate rank LM.
If H-M fails its promotion gates, execute H-R as the predeclared fundamentally
different reconstruction-objective pivot, not a retune of H-M.

Difference from strong public systems: audited V47 contains learned
classification heads, sparse votes, prompt centering/Gaussian heads and family
likelihood experts. This study isolates PRE-COMPETITION author/source-specific
pretraining and per-class empirical directional density; no causal claim about
their 0.668/0.666 gains and no assumption that all upstream training is known.
The source encoder, contrastive method and density estimation are established
methods; we claim a falsifiable adaptation/complementarity investigation, not
invention of those methods.

Primary sources:
https://proceedings.nips.cc/paper_files/paper/2024/hash/a117a3cd54b7affad04618c77c2fb18b-Abstract-Conference.html
https://github.com/heyongxin233/DeTeCtive
https://huggingface.co/heyongxin233/DeTeCtive
https://arxiv.org/abs/2606.10099
https://huggingface.co/rrivera1849/LUSR
https://proceedings.mlr.press/v267/sun25z.html

## Frozen assets and resource boundaries

H-M: heyongxin233/DeTeCtive revision
cefc9833290189b414a683d0775edb900c1832c1, Deepfake_best.pth
SHA256 31a47928ea17f1fc872312e9896a0952c1dae48dba5660b9594ee5e05f2487e1
(published 2024). Select this documented cross-domain/model default checkpoint,
not the winner among multiple checkpoints.

Control: princeton-nlp/unsup-simcse-roberta-base revision
db28710348cf9f33a2be25c505f98f0fbbbfe768, pytorch_model.bin
SHA256 25ff6c2ce68194d2b5636bc19f4f5556318fbd9636ff72f332e714e976deabcf.

Conditional H-R pivot: rrivera1849/LUSR revision
ab7852825ad1be40112bbfdaf1f9e2315a1727b7, model.safetensors
SHA256 9dc936bd31f349d874a79880549a5a9a9ca0a7ead64bebb83904cac2666cd5b9.
Model card: MIT, RoBERTa-base 124.6M. Do not silently substitute the paper's
mentioned larger research encoder for the actual released checkpoint.

Frozen encoder CPU inference and the specified CPU readout fitting ONLY in
this study. Torch 2.6.0+cpu; Transformers 4.48.3; sklearn 1.6.1. FP32 encoders,
eval + inference_mode; max_length 512; FIXED padding to 512 for the two H-M
models, batches of 16; 2 CPU threads per encoder, at most two independent CPU
encoder jobs concurrently. No new encoder fine-tuning, API
generation, GPU or hidden labels in this study.

PRE-ENCODING SOURCE-FAITHFULNESS AMENDMENT: upstream training pads to 512 and
text_embedding.py calls masked_fill without assigning the result. Thus its
published normalized representation includes padding-query hidden states.
The valid-length divisor cancels under final L2 normalization: it equals L2
normalization of the mean of ALL 512 states. Use that exact definition for
H-M PRIMARY, with fixed padding so unrelated batch lengths cannot alter it.
SimCSE and DeTeCtive use identical tokenization/pooling for each comparison.
Also retain correct valid-token masked mean (including valid special tokens)
from the SAME forward as a predeclared padding/length robustness control;
no extra checkpoint or result-based pooling selection. Require positive
DeTeCtive-minus-SimCSE masked-density deltas on official OOF and source mean
in addition to the primary gates. A gain only in the published padded view
does not pass. H-R uses valid-token masked mean.
This amendment follows independently inspected source BEFORE model predictions;
it avoids treating a changed inference recipe as the trained method's failure.
Verify fixed-padding batch invariance and the masked-pooling implementation.
Operational equivalence optimization, before any OOF/source score: these
RoBERTa models use absolute positions and give every masked pad token the same
padding position ID. With dropout off, all pad-query states in a passage are
identical, and masked keys cannot influence valid states. Compute one padding
query, then restore its multiplicity (512 minus valid length) in the published
mean. Sort computational batches by valid token length and restore original
row order. Logical max length and BOTH representation definitions are unchanged.
Require a real full-512 first-batch comparison for EACH checkpoint with maximum
normalized-feature difference <=1e-5, in addition to real batch invariance;
otherwise INVALID and use/fix only an equivalent execution path. Tiny real
RoBERTa tests separately verify the identity. This is a speed optimization,
not a new candidate, lower token budget or a performance-based model change.
Load checkpoints weights_only=True / safetensors and validate architecture/key
coverage. No trust_remote_code or execution of downloaded model code.

Cache embeddings/predictions only outside Git; fingerprint ordered text hashes,
tokenizer/model revisions, weight hashes and pooling settings. No official
passages, IDs or row-level predictions are published.

## Data and leakage-safe research comparison

Official train SHA256:
40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707
Permanent folds.csv SHA256:
94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d.
All 2,699 rows, 150 prompt groups, 18 classes. Reuse the permanent five folds;
never create a new salt. Encoders never update from these rows. All reference
selection, fitted scaling and learned readouts are fold-local.

Own LoRA24 OOF original archive SHA256:
fa675fcf8ac3860984b449748cf067c3da29389a921417e5e1047bb22a7be8ce,
Actions 41398543856 / artifact 13125872633.
Verified exact id/prompt/model/fold alignment and BA 0.5467611235396471.
It contains hard predictions ONLY, not probabilities. Do not invent logits or
claim an evaluated LoRA probability blend. Use exact paired correctness,
rescues/regressions, provider/sibling errors and per-class recalls.

Transfer: frozen public pool_v4.parquet SHA256
37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec,
existing conservative external subset, original held-out source membership
abdullah / semenoff / zakir (3,324 rows before any official-overlap exclusion).
These sources were already consumed project diagnostics; explicitly distinguish
this from fresh independent confirmation. No external rows fit the candidate:
transfer reference database = official training rows only. Exclude exact
prompt_id/prompt_key and normalized-answer overlaps before evaluation. Report
all exclusions, >=20 rows/eligible label and >=8 labels/300 rows per source;
do not choose a favorable subset if support fails.
This is a predeclared source shift plus matched backbone/readout ablation under
EVALUATION_POLICY, not causal generation-protocol invariance.

## Frozen arms and inference

For L2-normalized query z, reference vectors z_i of each class y:
centroid score = mean_i(z dot z_i) / tau.
density score = logmeanexp_i((z dot z_i) / tau).
Use tau=0.07, the upstream pretraining temperature, no temperature/neighbor/
normalization sweep. Common vMF kernel normalization cancels between classes.
The density retains multimodal reference evidence; centroid keeps its first
moment only. Reference class frequency is normalized out. Both produce all
18 finite scores and an argmax label. No source ID or prompt ID at inference.

H-M factorial controls, PRIMARY published 512-state pooling:
B0: unsupervised SimCSE + centroid.
B1: unsupervised SimCSE + density.
M0: DeTeCtive + centroid.
M1: DeTeCtive + density (PRIMARY H-M CANDIDATE).
No selection of M0 instead of M1 after seeing results.
Repeat the same four scorers on the predeclared masked-state view only for
mechanism/robustness evidence; do not promote that view as a post-hoc winner.

H-R, executed only after H-M fails:
R1: LUSR + StandardScaler(train-only) + balanced multinomial logistic regression
(C=1, max_iter=2000, lbfgs, seed=20261108).
R0: same readout/settings on the frozen SimCSE features.
This compares reconstruction-pretrained and semantic-pretrained encoder
PACKAGES under a matched readout, not the isolated causal effect of the
reconstruction loss or frozen semantic encoder. SimCSE is not LUSR's original
initialization. A benefit motivates an objective-isolating follow-up; it does
not prove why the pretraining worked. It is not a DeTeCtive rescue or sweep.

## Criteria fixed before encoding/scoring

Primary metric: exact-18 official grouped OOF BA, same rows as LoRA24.
For either candidate to advance, ALL must hold:
1. OOF BA >=0.5567611235396471 (>=+0.010 over own LoRA24);
2. paired prompt-bootstrap candidate-minus-LoRA24 97.5% percentile interval
   strictly above zero, 5000 replicates, seed 20261108; explicit central
   percentile quantiles [0.0125, 0.9875]. Multiplicity protection applies only
   to these TWO predeclared candidates, not earlier adaptive project research;
3. own matched control improvement >=0.020 (M1 vs B1, or R1 vs R0);
4. no six-provider balanced accuracy reduction >0.010 against LoRA24;
   no label recall reduction >0.10;
5. external source mean candidate-minus-matched-control BA >=0.020 and
   at least two of the three source point deltas positive.

Additional H-M mechanism criterion: M1 OOF BA exceeds M0 by >=0.005.
Every score/class/support and same-row rescue/regression is reported whether
favorable or not. A nonpositive/wide interval means no established improvement,
not statistical proof the model can never help.

Also report paired candidate-only versus LoRA-only correct counts and
conditional error patterns, without post-hoc gates or fabricated ensembles.
These diagnose missing information; standalone strength and complementarity
are different claims. Scientific validation can use our own honest anchor;
missing third-party OOF is NOT a universal submission blocker.

If H-M fails, close this frozen source-contrastive-density candidate and
execute the predeclared H-R pivot; do not change weights, pooling, tau or data.
Checkpoint/key coverage, fixed-padding/masked-pooling invariants, isolation
or logistic convergence failures are INVALID engineering evidence. Discard
affected predictions, repair only the demonstrated defect without changing
scientific settings, and do not treat an invalid run as a scientific rejection
or a trigger to pivot. Convergence warnings are errors, not usable results.
If H-R fails, reconsider the global hypothesis from actual representation/
provider/sibling evidence and pursue a materially different defensible test
if one exists within authorized resources. Do not stop merely because a test
failed, and do not start another small retune.

## Promotion, notebook and safeguards

A passing candidate can be an independent new submission, compared once to
Public Best 0.668. It need not have third-party OOF to exist. This does not
assert a private-score gain, intrinsic novelty, or causal protocol invariance.
Before submission build an offline Notebook using its actual frozen model,
all official training references and hidden test texts; validate 18-class
outputs, every id exactly once, columns id/model, submission.parquet, true
inference path and runtime. No preview is evidence of a hidden full run.
GPU is permitted only in an official Code Competition rerun if genuinely
required; this encoder/readout is intended to be CPU-runnable.

Existing public-data guards remain. No official data, credentials, raw
reference embeddings or per-row predictions go into Git/Actions public
artifacts. Preserve earlier negative results and contracts.

## Restriction and implementation plan

RESTRICTION: frozen backbones for this study; no standalone GPU pilot, local
GPU training or paid API. WHY: freezing isolates existing pretraining geometry
from new task adaptation. Only the GPU/paid prohibitions come from current
user scope; CPU backbone training is not prohibited by the user.
STATUS ACTIVE for this study. PIVOT: evidence that task adaptation is the
missing capability can justify a separately contracted CPU training test;
GPU/paid activity still needs changed user authorization.

1. Implement and test masked pooling, safe checkpoint loading, unit-vector
   density/centroid scoring, true reference-fold exclusion, alignment and
   paired prompt bootstrap. Demonstrate real padding invariance with CPU tensors.
2. Encode the two frozen H-M models on official and transfer texts; run the
   frozen factorial OOF/source comparisons once; archive aggregates and private
   aligned outputs. If rejected, encode H-R and execute its frozen comparison.
3. Apply gates; independently review and update NOW/results. If promoted,
   build/push/submit one exact runnable Notebook and retrieve official status.
