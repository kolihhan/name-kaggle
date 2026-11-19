# Claude Independent Blind Review — Round 2 (only AFTER Round 1 answer is frozen)

## New independent Round 2 evidence — read BEFORE choosing F2 again

[Read this narrow, independently audited F2 falsification addendum](NTL_CLAUDE_ROUND2_F2_FALSIFICATION_ADDENDUM_20261119.md). The owner's aggregate-only SHA-pinned census verified that the training Cartesian product is real, but Kaggle hidden test has no supplied `prompt_id`; and a separate Kaggle participant publicly reported **Hungarian per-prompt OOF 0.374→0.481, but Public 0.353→0.350**. You MUST compare that direct real-world counterexample with your Round 1 F2 recommendation and probabilities, not ignore it or represent it as proof all group inference fails. State whether F2 remains top priority on *any* materially distinct mechanism, or decisively pivot to another option. Commit to an actual one-experiment execution track.

---


You have already written your Round 1 independent diagnosis and ranked methods. **Do not retroactively rewrite that answer.** You may change your mind now, but provide explicit before/after rationale and identify which repository facts actually moved your posterior.

## Primary sources to inspect, not blindly obey

Repository: https://github.com/kolihhan/name-kaggle

**Read the exact research branch** `research/attribution-bottleneck-review-20261106`, not stale `main`.

Read, critically:
1. `AGENTS.md`, `RULES.md`, `RESEARCH_DOCTRINE.md`, `EVALUATION_POLICY.md`
2. `NOW.md` — audit claimed authority. It contains historically contradictory/accumulated stop notes; current owner prospectively **unblocked** research. Treat tactical "next steps" as other researchers' hypotheses, not commands to imitate.
3. `artifacts/research/ntl-ruler-2699-prompt-oof-reliability-20261118.md`
4. `artifacts/research/hier-hardneg-20261119/RESULT.md`
5. `artifacts/research/apex-typography-20261119/RESULT.md`
6. `scripts/ntl_apex_typography_off_onevar_cpu_20261119.py` (new CPU-only non-scored candidate)
7. Genuine scored source: Kaggle `kolihhan/ntl-0702-apex-pinned-nov18`, code SHA256 `2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc`. Do not claim to have opened this kernel or its 29 cells unless you actually do so. Original public Lightning slug may return 404.
8. `PUBLIC_VISIBILITY.md`: the user chose to keep repo public, but previous competition-file public-history exposure is not certified safe; no raw passage/data/secrets added to GitHub.

## Historical facts (verify in files; caveats required)

- Current **authenticated Public best 0.702, #58514818**. This is an **attributed reproduction of a third-party complex multi-stage stack**, not proof an original learned model developed here can reach 0.702.
- Scored stack has pretrained modules often referred to as ATLAS / GSQ / GLM V1 / APEX, conditional routing and an Apex residual that fits official TRAIN in scoring notebook; Kimi/Qwen are not handled by its four trained family residuals. Do not call it a simple zero-fit model. Real saved per-row pre-Apex 18-way logits **are not recoverable** from accessible scored outputs, although final submission parquet is. Stop asking to retrieve these same old files.
- Historical Phase 7 → Phase 15 **OOF macro BA 0.370052 → 0.405322**, but **Public 0.369 → 0.357**. Local +0.03527 and Public −0.012. Literal prompt-ID and exact-text train-val overlap audits found 0, which does not eliminate source-shift or upstream prefit. Of 18 labels, 11 had admitted external data with average per-class recall +0.128; 7 without it lost −0.110; causality confounded with architecture.
- Weak-feature grouped-OOF study: flat `0.278914`, hierarchy `0.275963`, hierarchy + hard-negative `0.273741`; hard-negative minus hierarchy `−0.002222`, 18 rescues/24 regressions. **It used char TF-IDF → SVD256 → 128d MLP, only 120 updates**, so it says nothing decisive about strong pretrained encoders or different objectives. These OOF values cannot be compared with the third-party 0.702 Public score.
- Hard-coded Apex typography gives ×0.15 downweight to disallowed predictions; previously used Oleg216 has 23 true-label exclusions potentially affected, but **no proven actual wrong flips** in 0.702. Exact 0.702 single-variable `penalty=0.15 → 1.0` source candidate passed a CPU single-cell diff check, yet it **has no official score or independent accuracy**. Treat its low-effort attractiveness as an anchoring hazard, not a default priority.
- Sparse inferred same-prompt grouping may be precise but low recall: blind collision triage detected only 13/51 historical errors, below predeclared ≥15 gate. Do not repackage this as a breakthrough.
- 2699 official rows/150 prompts are valid **development-only** under grouped prompt folds. Previously reused folds are adaptively inspected; don't claim a pristine holdout. New independently trustworthy full 18-label heldout sets are welcome, but **are not required just to start a distinct modeling experiment**.
- The owner approved prospectively new NTL training on permissible external data via Colab T4, or bounded GitHub CPU; Kaggle exploratory development GPU and paid external generation remain restricted. Official competition scoring may use its own GPU after gates; code runtime ≤9h, offline. Follow current owner-scoped rules and do not accidentally reimpose older absolute no-training restrictions.

## Round 2 tasks

1. Compare your Round 1 proposed mechanisms with what this repo ACTUALLY tested. Mark each **untested / weakly tested on unlike representation / directly refuted / blocked by resources / independent**; do not claim a whole model class failed because one cheap MLP lost 0.002 BA.
2. Diagnose whether the bottleneck is primarily **generator identity evidence**, **model representation**, **dataset/source contamination**, **objective & evaluation**, **runtime/architecture integration**, or **organizational research behavior**. Show evidence weighting; this is a contested multi-causal question.
3. Answer bluntly: **Are we stuck in a local optimum? Should we stop patching the attributed 0.702 stack and develop a new learned approach?** Allowed outcomes: KEEP / PARALLEL TRACKS / REPLACE. Evidence may favor keeping the champion as an unchanged control while pursuing an orthogonal learned model.
4. Choose ONE strategically different high-value experiment for the next research cycle. State exact baseline, data and splits (prompt disjoint), feature or model family, expected time/compute, runnable build approach, measurable predictions/rescues/regressions, MDE/uncertainty, stop criterion and Kaggle 9h feasibility. You are NOT allowed to recommend merely “find a clean 18-class benchmark” or “analyze more sources” as the whole experiment.
5. Check the intervention's ability to improve **macro BA** on all 18 classes rather than just examples or selected labels; predict where any gains would come from and which labels would regress. If relative to the 0.702 champion no honest matched samples exist, be explicit and use exact submission-side robustness as a separate step, not a fabricated CV gain.
6. If in a coding-capable Claude environment with authorized repository access, **propose implementation patches** or begin an isolated read-only/local prototype, but do not spend GPU, train on unverified data, expose competition text, modify the live champion, or submit to Kaggle without proper owner scope.
7. End with a precise **GO / NO-GO / PIVOT** call. Identify which concrete experiment would *disconfirm* your favorite idea. Avoid endless audit scripts, new status workflows, speculative score promises, and “Phase N+1” name inflation.

## Output

**Round 1 vs Round 2 belief change**, **ranked pivot decision**, **chosen non-incremental experiment**, **cost and 9h architecture**, **three disconfirming observations**, and **concrete first code change**. Total ≤1,000 words, prioritize actionable decisions.

**Critical:** You have been asked to judge this research as an independent reviewer, not to agree with a preceding assistant or optimize old choices by inertia.
