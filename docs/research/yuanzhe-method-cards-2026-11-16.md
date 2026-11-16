# Yuanzhe Zhou — source-grounded method cards (2026-11-16)
Evidence labels: **SOURCE** direct public text; **INFERENCE** inferred transfer only; **UNKNOWN** not found. This is *not* a claim of a universal personal competition process. Do not attribute team work to Yuanzhe alone. Ranked by relevance to Gemma / NTL.

## 1. Konwinski Prize — 8th (highest relevance to Gemma)
- **a. Problem/metric [SOURCE]:** autonomous fixes of *future* GitHub issues. Official score penalizes wrong patches and slightly penalizes skips; different from Gemma's PASS-rate objective. [overview](https://www.kaggle.com/competitions/konwinski-prize/overview/description).
- **b. Validation [UNKNOWN]:** personal internal CV selection, task distribution and confidence estimates not found in accessible public solution materials.
- **c. Key decisions and ablations [UNKNOWN]:** no attributable 8th-place writeup, full code, or signed ablation located in this pass. Do **not** borrow the 1st/5th-place writeups and label them his.
- **d. Credit [SOURCE]:** Kaggle [public profile](https://www.kaggle.com/yuanzhezhou/competitions) lists yuanzhezhou 8/617, solo listing; no implementation provenance. A public discussion shows an 8th-place participant, not a solution.
- **e. Final-selection rule [UNKNOWN]:** not disclosed.
- **f. Missing:** participant's exact notebook/traces, local folds, inference strategy and final selection criteria.
- **Transfer [INFERENCE]:** measure evidence→patch→objective validation and grader PASS; Gemma's objective must be used (not Konwinski's skip-reward score).

## 2. LLM Prompt Recovery — 5th (highest relevance to NTL)
- **a. Problem/metric [SOURCE]:** recover prompts that transformed given LLM text; leaderboard uses prompt similarity objective (NOT 18-class NTL BA). [joint writeup](https://www.kaggle.com/competitions/llm-prompt-recovery/writeups/z-d-z-5th-place-solution).
- **b. Validation [SOURCE/PARTIAL]:** team compared local experiments and LB outcomes; exact CV split, uncertainty, train/holdout isolation and sample counts per ablation **not disclosed**.
- **c. Team methods [SOURCE]:** synthesize inputs from multiple LLMs (~500k examples after processing); RoBERTa-base/large predicts T5 prompt embeddings; cosine retrieval; Mistral 7B candidate prompt generation; reranking permutation combinations. Token suffix trick showed a ~0.05 local boost that **did not transfer to LB**; they investigated TF-vs-Torch T5 tokenization behavior. Team explicitly says it chose not to overfit 200 public prompts.
- **d. Credit:** writeup explicitly thanks teammates zengzhaoyang and dmitriyab; entire pipeline described as **we**; no credible per-step personal contribution split. List all three authors; do not infer sole inventor.
- **e. Final selection [SOURCE/PARTIAL]:** explicitly declined overfitting 200 public prompts; exact final candidate selection arithmetic/rules **not disclosed**.
- **f. Missing:** raw ablation table, grouped CV mapping, uncertainty, exact candidate cutoff, number of prompt prototypes and attribution of parts.
- **Transfer [INFERENCE]:** test protocol/source/style shift with source- and prompt-disjoint data, keep generation-protocol stratification; a local shortcut without independent transfer cannot be promoted.

## 3. CMI Detect Behavior — 1st (diversity and complementarity)
- **a. Problem/metric [SOURCE]:** wrist-sensor behavior recognition, custom evaluation, not directly NTL or SWE. [joint writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution).
- **b. Validation [SOURCE/PARTIAL]:** a listed contributor describes subject-grouped 5-fold CV. Other team parts' CV setup not fully uniform/disclosed.
- **c. Key decisions [SOURCE]:** ensemble different sensor-feature branches and model families (CNN/RNN/GRU etc), weighted logits, orientation-specific postprocessing; some low standalone performers helped blend. Writeup describes several specific preprocessing/augmentation choices, not controlled universal ablations for each component.
- **d. Credit:** named team yuanzhezhou, ogurtsov, devinanzelmo. Writeup has **Devin's**, **Ogurtsov's**, and **zyz's** sections; `zyz` appears to correspond to Yuanzhe but this mapping is not independently confirmed in the captured text. Do not assign team-wide ensemble or preprocessing exclusively to him.
- **e. Final selection:** final large ensemble described; exact frozen selection criterion/weight sweep policy **not disclosed**.
- **f. Missing:** full OOF error covariance and per-component rescue/regression accounting in accessible writeup.
- **Transfer [INFERENCE]:** NTL should only ensemble model families with *independent* same-row rescue evidence and OOF error complementarity, not stack correlated notebooks or Public-title scores.

## 4. Mitsui & Co baseline — published notebook; NOT a verified competition final strategy
- **a. Problem/metric [SOURCE/PARTIAL]:** commodity-price/time-series forecasting with custom financial metric; [author notebook](https://www.kaggle.com/code/yuanzhezhou/mitsui-co-baseline-train-infer) and [author-attributed article](https://cloud.tencent.com/developer/article/2645634).
- **b. Validation [UNKNOWN]:** source notebook content inaccessible through page crawler; article asserts leak-free baseline, but its exact CV schedule, folds and score correlation NOT independently inspected.
- **c. Decisions/ablations [UNKNOWN]:** no auditable author-provided notebook cell/exact ablation extracted.
- **d. Credit:** public notebook attributed to yuanzhezhou; article attributes authorship to him. No team-specific final solution claims.
- **e. Final selection [UNKNOWN]:** this is a baseline, not a documented winning submission selection.
- **f. Missing:** source cells, actual CV/LB pairs, seeds and final candidate selection.
- **Transfer [INFERENCE]:** explicitly measure OOF/Public pairs without pretending old adaptive-fold OOF predicts new pretrained champion.

## 5. Santa 2024 — team 3rd; Jane Street — historical team 1st
- **Santa a/b/c [SOURCE/PARTIAL]:** optimize word permutations for perplexity; team [writeup](https://www.kaggle.com/competitions/santa-2024/writeups/santamizers-3rd-place-solution) describes simulated annealing, varied move operators, cyclic cooling, different starting permutations, top-k branches. Exact independent validation policy not fully disclosed.
- **Santa d/e/f:** coauthors yuanzhezhou, Steven_Y, Ali/asalhi, Sia; post says much of writeup completed by asalhi. Which experiments Yuanzhe did, exact final-selection protocol and per-operator controlled ablation **unknown**. No personal sole-credit.
- **Jane Street a/b/c [SOURCE/PARTIAL]:** team solution with supervised autoencoder and boosted trees; [Yirun-specific writeup](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s) openly discusses validation-label leakage from representation fit before CV, then avoids it. **This is Yirun's documented contribution, NOT Yuanzhe's**.
- **Jane Street d/e/f:** Yuanzhe is among team authors but his individual implementation and final selection rule not established by this source. In [his public comment](https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/222473), he says the team relied on local validation and observations to reduce overfitting; no source for claimed universal method.
- **Transfer [INFERENCE]:** NTL all learned stages fold-local, including embedding fit when labels or validation text could leak; Gemma issue/task family boundaries kept disjoint. Search diversity only under a fixed experiment budget, unlike tuning many variants against Public LB.

## Small, testable migration table
| Source-grounded principle | NTL hypothesis/protocol | Gemma hypothesis/protocol | Decision rule | Cost |
| --- | --- | --- | --- | --- |
| Prompt Recovery: local boost can fail LB | change protocol/source domain only in a *new* independent 18-label cohort | check patch feedback under unseen repo issues | same-row group-aware 95% CI lower>0 before promotion; else insufficient | metadata CPU low; proper holdout/GPU high |
| CMI: diverse methods may add complementary errors | compare independent family OOF rescue/regression and conditional correlation, no Public weights | compare evidence-channel intervention against V2, not another prompt variant | preregister simple fusion/test; require paired n, CI and actual uptake | GPU high |
| Jane Street: fit before fold can leak | audit every embedding/calibration/route training boundary | no gold/reference patch in model-visible context | zero leakage violations, else invalidate run | CPU low |
| Santa: search candidate diversity | one *orthogonal* representation, fixed-budget evaluation | bounded test-first vs root-cause-first agent only after legal harness gate | preregister paired issue outcomes, regressions and runtime | medium/high |
| Konwinski: unknown personal implementation | N/A | use honest future-issue separation and objective-specific PASS, not skip score | no method claim until source found | CPU initially |

**Scientific status:** All transfers above **UNTESTED** in our repos unless independently linked to a genuine preexisting test. This document is source audit, not experiment outcome.
