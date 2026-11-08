# Conditional denoising source attribution — frozen v1

Date: 2026-11-08 MYT. Freeze before fitting or scoring this candidate.
Execute after the already-frozen H-R test rejects; no H-M/H-R retuning.
The user requests research through a real, scientifically eligible submission.

## Global hypothesis and selection

Source balancing at befcb1c is CLOSED. Old-generator contrastive geometry has
already failed (.0922 OOF versus .1226 matched semantic-density control); a
recovery replay is needed because its private scratch cache was reset. LUSR
reconstruction geometry is the predeclared H-R test, still unscored when this
contract is written. Neither tests learning present-day exact-model token
choices under their contexts.

H-C: a source-conditioned model of masked word choices can learn evidence
that a pooled embedding or compressed rank histogram loses. Train token
conditionals, not a direct 18-way source classifier. The backbone is a frozen
MLM, with small source-specific residual maps before its vocabulary decoder.
This is a conditional-denoising surrogate, NOT the original generator's
probability, a joint text likelihood, or a proper wordwise PLL. Selected words
are masked JOINTLY in ONE corrupted-context encoder pass per document.

Material alternatives considered: (1) current-label end-to-end contrastive
encoder adaptation would address task mismatch, but its CPU backward passes
are substantially more expensive and it does not isolate conditional choice;
(2) a larger frozen semantic backbone adds capacity without source adaptation,
and earlier evidence does not establish size as the bottleneck; (3) the present
conditional objective directly learns current-label choices with a real
context-destroying, capacity-matched control. Select (3), subject to an actual
external-only screen. This is not another SVC, source weight or density retune.

ATLAS already trains source-conditioned causal LMs, including source-specific
low-rank maps and relative token evidence. Profiler already studies contextual
surrogate logits. No claim of inventing conditional attribution. Differences
here: exact 18 labels, bidirectional jointly masked word choices, one shared
frozen CPU encoder, output-only residual adapters, and explicit marginal-word
and context-destroyed controls. Audited V47 has family likelihood experts;
this is not a claim that likelihood evidence is absent from V47 or that a
larger/source-conditioned model necessarily improves its 0.668 reproduction.

Primary sources:
- https://arxiv.org/html/2610.19583v1 (ATLAS; its family/OOD results do not prove exact sibling performance)
- https://aclanthology.org/2025.emnlp-main.1265/ (Profiler)
- https://aclanthology.org/2020.acl-main.240/ (true one-token-at-a-time PLL)
- https://aclanthology.org/2023.acl-short.80/ (within-word leakage)
- https://huggingface.co/FacebookAI/roberta-base

## Data, exclusions and evidence status

Public pool_v4.parquet SHA256:
37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec.
Use existing build_conservative_subset and build_stage1_split exactly.
Preliminary train: 30,133 rows, all 18 labels. Heldout sources remain exactly
abdullah/semenoff/zakir (3,324 preliminary rows). No source reweighting.

Additional public predictive check: zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations,
ntl_proxy_generations.csv, 834 rows, 8 exact labels, SHA256:
a8f61b647b5f2b2f99b73f083b7358445d93c3ebbff8607b44423850e9fb5583.
Metadata and length/protocol audits were already consumed; NO attribution
predictions have previously been evaluated on this file. Do not call it
wholly fresh or an independent 18-label benchmark. It has label-dependent
requested lengths, proxy endpoint uncertainty and shared prompt banks.
Canonical key = bank sy -> seyitbank:, bank pv -> prvsiyan:, plus prompt_id.
Before this freeze, metadata-only overlap checking found 11 rows whose keys
occur in the preliminary training set. Exclude those rows mechanically:
the predeclared proxy check is the remaining 823 rows, 8 labels, >=64/label.
Do not select a different subset on performance. Report per-bank support and
BA, marking unsupported banks rather than inventing equal coverage.

Purge external TRAIN against the union of ALL official texts/prompt IDs,
the three source holdouts, and the fixed proxy holdout: prompt_id, canonical
prompt_key and normalized answer (NFKC, lower, whitespace collapse). This
global official exclusion permits one external initialization to be shared
across OOF folds. Record every removed count and prove zero residual exact
key/text overlap. Unknown semantic prompt identity remains a limitation.
Do not see hidden test data while fitting. Keep every raw row, hidden state,
prediction and learned checkpoint outside public Git/Actions artifacts.

All three old source checks are reused development diagnostics. The proxy
check is a new predictive evaluation with previously inspected metadata,
not a causal model×protocol test. STABLEPAIR's failed matched-request causal
gate stays CLOSED; this contract makes no protocol-invariance claim.

## Model and corruption, fixed before results

FacebookAI/roberta-base revision e2da8e2f811d1448a5b465c236feacd80ffbac7b,
model.safetensors SHA256:
5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81.
MIT, standard RobertaForMaskedLM, FP32 CPU, eval, frozen parameters. No remote
code. Torch2.6.0+cpu, Transformers4.48.3, max_length512, right dynamic padding.
Cache z = pretrained MLM dense/GELU/LayerNorm output at each masked subpiece;
decoder W and original bias b remain frozen. Record all asset hashes, ordered
input hashes, corruption specification and real full-model equivalence proof.

Word groups = whitespace-delimited spans with at least one tokenizer subpiece
inside the retained max512 context. Drop a group cut by truncation rather
than leave part of a scored word visible. Hash full passage UTF-8 with seed
20261108 to seed one label/source/ID-blind deterministic word permutation.
Jointly mask k=min(ceil(.15*n),32,n-1) groups for n>=2; for n=1 mask that one
word (explicit unconditional fallback). Reject empty/no-scorable passages.
Replace EVERY subpiece of selected words with mask_token_id, 100% replacement;
no target subpiece survives in the encoder input. Score all their subpieces.
Each selected word has equal weight; subpieces within each word equal weight;
each document has equal total weight. Report corruption fractions/lengths.

For masked contextual state z and source m:
  logits_m = W @ (z + B_m @ A @ z) + b + c_m.
A: shared rank16 down map, normal std .02 initialization.
B_m: source-specific 768x16 up map, ZERO initialization (A is not zero).
c_m: source-specific 50,265-vocabulary bias, ZERO initialization.
All 18 source heads share the exact pretrained W,b and context representation.
No label prior or collection-source metadata is used at inference.

Arms, identical rows/corruptions/optimizer schedule/order/initialization:
C0: c_m only, B/A absent; contextual base remains genuine.
C: A,B_m,c_m with GENUINE residual context (PRIMARY).
Cs: identical capacity and budget to C, but only residual-context donors are
shuffled. Base Wz+b stays GENUINELY row-matched, so this is not a trivial
corruption of the pretrained base model.
Donors must come from a DIFFERENT prompt in the same exact-label x collection
source x wordpiece-width x subpiece-index stratum. Build one deterministic
permutation with seed20261108; keep unsupported entries unchanged and report
their fraction. A valid shuffle requires >=90% of train token weight to have
different-prompt donors. Otherwise INVALID/unsupported design, not rejection.
At inference ALL arms receive genuine contextual states; no true labels or
collection sources are needed. Word/label/source marginal targets stay fixed.

## Training and exact scoring

External Stage A: one complete epoch, no sampling/cap/early stopping, all
retained training rows/selected subpieces. Train all three arms with full
50,265-vocabulary cross entropy for the TRUE source condition only. Source
labels select heads, never an 18-label softmax objective. No sampled softmax,
top-K normalization or logits mistaken for probabilities.
AdamW, A/B LR3e-4, vocabulary-bias LR1e-3, betas(.9,.999), eps1e-8,
weight_decay.01 (A/B only; bias decay0), grad_clip1, no warmup or LR search.
Microbatch128 target subpieces; use fixed seeded permutation of whole docs,
flatten their selected wordpieces in stable word/subpiece order, then split
into batches. Weight = 1/(selected_words * wordpiece_width) times inverse
TRAIN document class frequency, normalized to mean1 over training documents.
Loss in each minibatch is mean(weighted token NLL), using the globally fixed
weight scale, not a per-minibatch renormalized ratio. Steps=ceil(total retained
target subpieces/128), exactly one epoch. Seed20261108, CPU threads4 maximum
for this candidate; memory/runtime measured and logged. No GPU or paid API.

Report loss at identical predeclared fixed training-monitor subpieces before
and after fitting; a nonfinite/flat/increased objective is INVALID/underfit
for a mechanism claim, not evidence against all conditional attribution.
No score-based epoch/LR/rank/word-mask rescue is allowed.

Inference computes proper full-vocabulary token log probabilities for ALL18
heads, including on 8-label holdouts. Outside-eligible-label guesses are errors.
Use ATLAS's published fixed bounded aggregation, not an adaptive clipping rule:
for each subpiece t, r_m,t(T)=exp(-sum_j max(l_j,t-l_m,t,0)/T), T in {1,2,4,8}.
Average first over T/subpieces within word, then words within passage. Argmax
over all18; lexicographic label order breaks exact ties. Do not claim that this
MLM surrogate reproduces ATLAS's causal/prefix/attention-adapter model.

## Stage A falsification and continuation

Before any official label fit/OOF, evaluate external-only trained arms on the
three source holdouts and fixed proxy823. Within each old source, eligible
true labels require >=20 rows and training coverage, >=8labels/>=300rows.
Proxy: all8labels require >=20rows, at least700total. If support fails INVALID,
no favorable source/label substitution. Report every label recall, provider
BA, sibling/cross-provider error, rescues/regressions against BOTH controls.

Continue to fixed Stage B only if ALL:
1. proxy823 exact eligible-label BA: C-C0>=.020 AND C-Cs>=.020;
2. paired prompt-cluster central97.5% percentile intervals for both proxy
   deltas have lower>0 (5000replicates, seed20261108, [.0125,.9875], >=4900valid);
3. mean three-source BA gain C-C0>=.010 AND C-Cs>=.010, >=2/3 positive for EACH;
4. no source C-C0 or C-Cs<-0.020;
5. proxy provider BA C >= C0-.010;
6. valid training-objective drop, cache and shuffle proofs.
The two intervals protect only this frozen pair of proxy claims, not historical
adaptive research. These source-dependent BAs are NOT comparable to 18-class
Kaggle0.668 or LoRA0.520. If failed, reject this fixed output-adapter candidate;
do not retune. Reconsider global representation/training/data hypotheses from
actual failure patterns and pursue a materially different feasible test.

## Stage B and submission (only after Stage A passes)

Official SHA40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707;
folds SHA94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d.
Exactly2,699rows/150prompts/18classes, permanent5folds. Each arm/fold starts
from its SAME external-only checkpoint; fresh AdamW; train exactly2epochs
on fold-training rows ONLY, same settings and train-only class weights.
Cs donors rebuilt using ONLY fold-training prompts; never validation donors.
All validation states are from the unchanged frozen backbone. Final source
readout uses all official rows after validation and follows the same2epochs.

Promotion requires C OOF>=.5567611235396471, paired prompt97.5% CI versus own
LoRA24 hard OOF entirely>0, >=.020 over both C0 and Cs, provider BA no more
than.010 below LoRA, no exact-label recall more than.10 below LoRA, and the
same source/proxy point-gain guards after final official fit. Report5folds,
all18recalls, error partitions and real same-row rescue/regression counts.
Own anchor archive SHAfa675fcf8ac3860984b449748cf067c3da29389a921417e5e1047bb22a7be8ce;
exact id/prompt/model/fold alignment required; its probabilities are absent.
No fabricated LoRA blend. The own-anchor comparison does not stand in for a
paired claim versus V47/Ozertuu0.668; lack of their OOF is not a universal block.

If promoted, build the REAL offline Kaggle Notebook with pinned pretrained
backbone and learned final heads. Test true hidden inference, all18scores,
exactly-once IDs and columns id/model in submission.parquet, offline wheel
compatibility and full runtime. A preview shortcut proves nothing. Freeze
one Notebook version, submit once, retrieve actual Submission ID and official
score when available, compare to0.668. No Public-LB tuning.

## Implementation plan

1. tests/test_conditional_denoising18.py: write/run failing behavioral tests
   for complete word masking, equal word/document weights, actual decoder
   equivalence, full-vocab normalization, source/head isolation, genuine-base
   shuffle, different-prompt donors/unsupported support and fixed aggregation.
2. scripts/conditional_denoising18.py: implement the frozen mask/cache,
   provenance, low-rank conditional decoder, CPU training and stage evaluators;
   reuse existing conservative split, overlap purge and fold/anchor checks.
3. Pass focused and repo checks; commit executable identity before fitting.
   Run a real checkpoint cache-equivalence test and exact compute-path timing.
4. Execute conditional Stage A once after H-R rejects, archive aggregates
   promptly and row-aligned predictions/checkpoints privately. Apply gates.
5. If eligible, execute unchanged Stage B, verify, package and submit. Update
   NOW/RESULT with actual status. Do not describe an unsubmitted notebook as
   submitted or promise background work.

RESTRICTION: frozen MLM body, output-only learning in this test.
WHY: isolate conditional word-choice readout without expensive backbone BPTT.
PIVOT: valid negative controls identify context/readout limitations; separately
contracted current-label attention/representation learning may be justified.
STATUS ACTIVE for this candidate only. CPU backbone learning is user-allowed;
no standalone/local GPU training or paid API remains authorized.
