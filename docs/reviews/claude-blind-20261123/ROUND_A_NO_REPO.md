# Claude blind review — Round A: independent hypothesis before repository exposure
Date: 2026-11-23. Reviewer instructions — read this file ONLY in Round A.

You are an independent principal ML scientist, Kaggle strategist and adversarial scientific reviewer. You did not design this project. Your primary goal is **correctness and discovery**, not agreeing with the owner, ChatGPT, Claude's prior answers, or local repo doctrine.

## Rules of blinding

- **Do not open the linked repository, previous review files, old results, score history, assistants' conversations or existing research proposals until your ROUND_A_ANSWER is frozen.**
- Do not assume this project should stay on the current model architecture, use one particular paper, find a perfect benchmark, or keep doing tiny patches.
- Do not give unconditional approval or deliberate contrarianism. Ground agreement *and* disagreement in testable arguments. Do not invent observed results.
- Write your Round A answer before asking for Round B. Do not revise Round A after seeing more evidence; put belief changes in Round B.

## Problem constraints supplied without solution hints

- Task: Given an AI-generated response passage, attribute it to **one of 18 exact generator-model identities** (several models belong to the same family/provider).
- Scoring metric: unweighted mean of recall over all 18 labels, i.e., balanced accuracy. Any subset score is conditional on the covered labels.
- Official training set: **2,699 labelled texts**, **150 original prompt IDs**; nearly all prompts have one output from each model. Group validation must leave prompt IDs disjoint.
- Scoring test: about **4,000** hidden passages, exact distribution and per-class denominator undisclosed. Only input text and row ID are available; **no test prompt ID is promised**. Public leaderboard is roughly 40% of hidden test; Public and Private class distributions can differ.
- Submission is a Kaggle **offline code rerun** with approximately **9 hours** runtime. Existing model assets and data must be legally accessible and allowed in the competition. Do not assume original generators' exact weights, logprobs, or serving endpoints are obtainable.
- Resources: owner seeks practical improvements under a tight inference budget and favors genuinely differentiated scientific work over repeated rule patches. No paid target-model API spending, exposure of private competition data, training on hidden labels, or new official submission is authorized in this blind-review task. You may evaluate whether *future* bounded lawful training is scientifically justified, but do not execute it.

## Independent tasks — write before any repo access

1. Model the latent data-generating process: distinguish **model identity**, **prompt/topic**, **serving version**, **system wrapper**, **decoding setup**, **response length**, **collection provider**, **source selection**, and **inference budget**. Which are identifiable from output text alone? Which can create false certainty?
2. Propose **at least three genuinely different competing explanations** for a likely plateau in 18-way attribution, and one observable falsifier for each. Separate intrinsic Bayes ambiguity from bad representation, calibration, data shift, and operations.
3. Rank **up to four research directions** (including "keep existing system and do nothing" if warranted). Avoid solutions that only rename n-gram styles, add a weak expert, tune a threshold, or blindly use Hungarian matching. Name required evidence and time/GPU cost, and say what could make each infeasible.
4. Describe the **minimum meaningful experiment** that could distinguish the top two explanations, respecting prompt grouping and without pretending historic CV is a clean test. If a strong incumbent cannot be honestly scored on the same rows, offer an explicitly scoped mechanism falsifier and distinguish it from promotion evidence.
5. Choose a **tentative strategic stance**: retain architecture, parallel new representation, replace architecture, or stop. Give the strongest argument *against* your own choice.
6. Explicitly state **what you do not know** and what strong evidence could reverse your ranking.

## Required output: ROUND_A_ANSWER.md

- Hypothesis ranking with a FACT / INFERENCE / UNKNOWN distinction
- Strongest research direction, one concrete falsifier, and cost envelope
- Two strongest adverse interpretations of your own favored approach
- Specific information required from Round B
- Provisional verdict (not a prediction of a new Public score)

**Stop after Round A.** Only then open STAGE_B_EVIDENCE.md. Do not browse neighboring review files now.
