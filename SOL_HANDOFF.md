# SOL EXECUTION HANDOFF — Name That LLM (2026-11-17)

> This is an execution handoff for the next model/agent ("Sol"), not a new permission to bypass frozen scientific contracts. **Finish with a legitimate, authenticated Kaggle submission and score receipt once a candidate qualifies. Do not stop at a proposal, audit, or a green CI.** If no candidate qualifies, report the precise scientific blockers and continue with a new, discriminating improvement experiment; do not submit a demonstrably worse model.

## FIRST 10 MINUTES (work immediately, not planning theater)

1. Read `AGENTS.md`, `PUBLIC_VISIBILITY.md`, `README.md`, `NOW.md`, `RULES.md`, `EVALUATION_POLICY.md`, `RESEARCH_DOCTRINE.md` and existing candidate experiment evidence, from **this actual GitHub branch**. The latest signed evidence takes precedence over stale historical headings.
2. Read actual official Kaggle competition Overview, Rules, Code and Submission pages; confirm compliance, competition active, daily quota, accepted entry and 2 final slots where applicable. Do not assume old rules or a notebook title imply the official score.
3. Read public writeups from Kaggle GM **Yuanzhe Zhou (@yuanzhezhou)** for transferable decision patterns, not copy-paste: 
   - 5th LLM Prompt Recovery: https://www.kaggle.com/competitions/llm-prompt-recovery/writeups/z-d-z-5th-place-solution — model how test data were generated; diverse synthetic/data origins (~500k); train a retrieval target, retrieve + rerank; explicitly reject a local +0.05 token trick that does **not** transfer to the LB; avoid 200-public-prompt overfitting.
   - 1st CMI sensor: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution — independent evidence streams, careful selection of invalid data, complementary variants.
   - 3rd Santa: https://www.kaggle.com/competitions/santa-2024/writeups/santamizers-3rd-place-solution — new search neighborhoods and diverse alternatives instead of optimizing one heuristic endlessly.
   These are **specific documented facts**, not a claim Yuanzhe formally prescribed our research loop.

## VERIFIED STARTING STATE

- Competition: https://www.kaggle.com/competitions/name-that-llm, **Featured Code**, awards points/medals; 18-class balanced accuracy; deadline **2027-01-13 23:59 UTC** (verify mutable page again). Offline Kaggle Notebook, `submission.parquet` with exact `id,model`, <=9h CPU/GPU, no online test-time API. Public/free external data only subject to provenance and rules.
- Our own authenticated scored Public best: **0.691**, official submission **#58375806**, attributed frozen publicly sourced Lightning inference-only system, **not an original model invention**. Target **>=0.700**, and beyond toward genuine medal rank. Do not confuse author's “0.694” title or 95.82% historically trained-in data with official LB.
- Frozen 18-model Oleg provenance-sensitive 216 rows: champion **89/216 = 0.412037** correct. **Already spent**; incomplete source/prefit provenance, cannot claim untouched generalization.
- Actual recent negative experiments:
  * [43895516290](https://github.com/kolihhan/name-kaggle/actions/runs/43895516290) blind same-prompt model-collision detection: 13/51 old errors caught, below predeclared 15 minimum; **stop exact configuration**.
  * [43897787642](https://github.com/kolihhan/name-kaggle/actions/runs/43897787642) typography 2/3 fixed features survive 18-label source comparison, but [43899966024](https://github.com/kolihhan/name-kaggle/actions/runs/43899966024) found only 0/51 and 1/127 incumbent errors flagged; **not useful for champion rescoring**.
  * [43902774559](https://github.com/kolihhan/name-kaggle/actions/runs/43902774559) flat TF-IDF+SVC **32/216**, hierarchy **34/216** vs frozen winner **89/216**; **do not promote**. IMPORTANT: that CPU run trained classifiers despite the current strict **NO NEW TRAINING** user boundary; this was a protocol violation, **not precedent**. Ask the owner explicitly before any further fitting; otherwise use already pretrained inference-only mechanisms.
  * [43884199474](https://github.com/kolihhan/name-kaggle/actions/runs/43884199474) text-blind complete prompt grouping at only 5 answers per prompt failed ARI/recall gate.

## DIAGNOSIS AND THE NEXT RESEARCH BET

**Central observed bottleneck:** 0.691 Public and 0.412 Oleg source-transfer; same few historic slices reused repeatedly; cross-model-family residuals dominate incumbent's selected-error pool (27/33 GSQ-selected errors); simple typography, hierarchy, prompt grouping and thin feature tweaks do not add reliable incremental rescues. Dataset/source-identity shift remains a competing explanation. Existing fitted assets' upstream exposure to Oleg/official text is not fully proven.

**H1: attribution is limited by inadequate *conditional generator likelihood* evidence** versus **H2: a text/channel/domain shortcut dominates and makes even good ID signals fail out-of-source**. Identify one **materially different pretrained, inference-only** mechanism that gives genuinely independent evidence versus the frozen champion (e.g. consented, provenance-safe conditional token-likelihood under *correct* candidate model variants, only if existing publicly usable weights and runtime are accessible) and specify a test whose outcomes distinguish H1 from H2. Do **NOT** simply add an LLM vote, one more punctuation rule, another random seed, or optimize 0.691 Public LB.

**Preferred evaluation:** identify or produce legal fully source/prompt-disjoint, all-18 truth corpus with verifiable served-model versions, attribution, license and prompt provenance. Do not just repeat already inspected Oleg; maintain a same-row frozen 0.691 champion control. For a small pilot calculate paired exact-label BA, rescues, regressions, discordance by provider and source, per-model recall and uncertainty. Clearly separate an exploratory small-n signal from promotion. Use only available authenticated private resources, no leaked hidden labels, paid API or new public competition data mirrors. Scientific goal is **material improvement outside spent data**, not another audit.

## MUST END AT SUBMISSION (when authorized and qualified)

1. Preserve a promotion contract *before* reading candidate's validation outcome. Current `EVALUATION_POLICY.md` and `RULES.md` require independent full-18 same-row source/prompt-disjoint evidence, applicable fold OOF and no-new-training; **this message explicitly expresses submission as the end goal, NOT blanket waiver of these existing gates**. If any gate blocks, explain the exact missing evidence rather than faking progress.
2. When a materially distinct candidate PASSES: freeze version, external asset SHA, input membership, label ordering and inference code; verify one offline Notebook run under Kaggle limits using existing repo-supported submission workflow; check official account/entry/quota/no active duplicate.
3. **ACTUALLY submit** through the authenticated Kaggle competition mechanism, not just create `submission.parquet` or say “ready.” Obtain official **submission ID, status, numerical Public balanced accuracy** from authenticated Kaggle. If PENDING, report PENDING only, without inventing a gain. Preserve best existing `0.691` reference unless new authenticated numeric beats it.
4. Record in `NOW.md`, official result/receipt artifact and committed source with exact commit/SHA. Don't repeatedly submit variations to tune LB.

## REPORT FORMAT (keep short)

| Competition | Official best before → after | New Kaggle submission ID/status | Real independent paired result | Decision |
Then **忙了什么 (<=3 short lines)**. Never call CI, package, locally trained or retrospective scores “official improvement”. Prioritize **scoring breakthroughs + official submission**, not repeated research-document commits.
