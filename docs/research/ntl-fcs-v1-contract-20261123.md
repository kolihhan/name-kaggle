# NTL Frozen Causal-Surprisal Fingerprint — FCS-v1, frozen before scoring

Date: 2026-11-23. Branch `research/ntl-frozen-causal-surprisal-20261123`.
This is a **no-target-LLM, pretrained contextual evidence feasibility test**; not a new 18-class autoregressive source-language model and not a Kaggle submission. Original source code and 0.715 winning model are never used for predictions or training.

## Hypotheses

- H1: A pretrained causal LM's **context-conditioned surprise signature** over actual text supplies exact generator-identity features beyond a token-marginal unigram and a context-destroying input control.
- H0: Changes mostly reflect simple token frequencies, length, prompt/topic or the different distribution when context is scrambled. A causal LM not trained specifically on target generators need not distinguish close siblings. The previous class-conditional abstract Markov surrogate scored 0.1893 versus its unigram 0.2445 and a direct Sparse 0.3671; this new mechanism uses **frozen real pretrained token conditional probabilities** rather than counts over ~99 manually constructed style tokens.
- Important algebra: subtracting a single common language model's log likelihood from every target source score does not change an argmax. Here we construct **per-response statistical representations** of contextual uplift and train a fold-local 18-class readout. We make no claim of recovering proprietary Claude/GPT/GLM likelihood.

## Predeclared data/evaluation

- Official TRAIN only: 2,699 rows, 150 grouped Prompt IDs, all 18 labels; frozen five-fold `artifacts/phase1/folds.csv`, exact `scripts.blind_prompt_contrast.load_snapshot` hash verification. These are **spent historical DEV folds** and are NOT a fresh 18-model independent source transfer or an estimate of hidden Public BA.
- Download public *pretrained*, unmodified `distilbert/distilgpt2` revision `f241065e938b44ac52db2c5de82c8bd2fafc76d0`, use `AutoTokenizer` and `AutoModelForCausalLM` in inference/eval mode, FP32 CPU with 2 torch threads. No fitting of foundation model, GPU, private model, new external source, model outputs submitted to Kaggle, credential logging or raw official texts exported.
- For each response tokenize BPE, retain first **127** tokens and prepend GPT-2 EOS as causal BOS. Original continuation score at token t = `log p_base(token_t | BOS, token_<t)` using exactly the real observed previous tokens. Define unigram control `log p_base(token_t | BOS)` from the frozen model's SINGLE BOS next-token distribution. All probabilities are properly normalized over the full 50,257-wordpiece vocabulary and computed by torch `logit(target)-logsumexp(logits_all)`. Truncate independently of label.
- Shuffled context: deterministic SHA256(`20261123|id`) token permutation **within passage**, leaving token multiset/length and BOS fixed, independent of true label/Prompt ID. Score teacher-forced shuffled text under the SAME frozen pretrained LM. Synthetic text may be out of distribution; the no-context baseline is therefore necessary.
- For all arms define identical fixed feature width, including distribution summaries of token log p (mean, std, 10th/25th/50th/75th/90th percentiles), mean by four relative positions, means by five fixed token shapes (word-start, numeric, punct/non-letter, alphabetic, other) with counts, plus length and unique-token ratio. Blocks are computed for **context, no-context, and context uplift=context-minus-no-context**. No supervised feature selection. For A, set context=no-context and uplift zero. For B, use real context. For C, use shuffled context (same bag-of-tokens); for all A/B/C no-context features computed from the same source token multiset.
- Train only the **readout** (StandardScaler + LogisticRegression C=1, max_iter=400, balanced 18-way classes) on the fold's training responses, predict held-out prompt groups. No in-sample readout scoring, no final training on public test, no model ensemble. D is the preexisting fixed Strong Sparse 4-view SVC for same rows/folds as an *independent* competitive reference (not a blend).

## Criteria for a useful mechanism (set before seeing results)

- Main A/B/C evaluation: actual 18-class grouped OOF BA, per-class recall, B−A and B−C paired changed/rescued/regressed rows and 1000-draw prompt-bootstrap CI, 5 fold deltas; D benchmark actual same rows.
- **Mechanism PASS:** B−A >= +0.02, B−C >= +0.02, both bootstrap 95% CI lower bounds >0, B wins >=3/5 folds against BOTH A/C, no class recall drop >0.15 versus A. All must pass.
- **Competitive signal:** B standalone >= D (same 18-class rows), otherwise still only useful if reproducible on a truly unseen source-disjoint benchmark with independent evidence (NOT the spent official OOF).
- If it fails, no retuning length, feature bins, class weights, thresholds, model checkpoint or seeds based on this run. Retire this **frozen one-model surprisal-summary representation**, not all pretrained token likelihood.
- If the pretrained snapshot is inaccessible, report **UNRUN/INFRA** rather than score/model failure.
- Cost: one GitHub Actions CPU job capped at 35 min; inference batch 6, sequence max 128, no GPU and no model training. Only aggregate metrics; full raw texts/IDs/row predictions are NOT logged or uploaded.
- NO official Kaggle submission, NO update to 0.715 Champion.

## Prior/artifact separation

Original score 0.715 is external-source ensemble Public, unpaired with this training-set OOF, not comparable as if same-row. Authentic GLM5.3/Flash next-token likelihood evidence in repo (historical pair 89.3% binary BA, plus small diagnostic 6 rescues/1 regression on 252 external texts) justifies studying real LM scores but is NOT an 18-class result. Existing `conditional-denoising18-v1` preregistered separate masked-model output-adapter experiment is **not executed or overwritten** by this cheaper preflight; its data and scientific gates remain unchanged.
