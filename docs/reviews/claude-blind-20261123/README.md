# NTL — Independent Claude Blind Review (2026-11-23)

**STATUS: REVIEW PACKET PREPARED; CLAUDE HAS NOT BEEN INVOKED.** This folder contains instructions and research evidence, **not** an independent Claude review answer. Do not claim that writing these documents makes a review complete.

## Why a third review?

A prior Claude review existed on 2026-11-19 for a different 0.702 scored incumbent. Since then the owner's externally sourced Public score reached an authenticated 0.715, extensive 2026-11-23 GLM, ATLAS, source, and instrumentation experiments completed, and the local research process itself became a candidate bottleneck. A new independent judgment must not simply inherit previous recommendations.

## The three locks (strict order)

| Stage | Show Claude | Never show yet | Required output |
| --- | --- | --- | --- |
| **A. Truly blind to repo** | Copy/paste ONLY [ROUND_A_NO_REPO.md](ROUND_A_NO_REPO.md) into a **new empty Claude conversation**. No repository URL in the initial message. | This README, Stage B/C, old reviews, NOW, score stories, today's assistant analysis | Round A independent method ranking/falsifiers; freeze as ROUND_A_ANSWER |
| **B. Actual evidence** | Only after A is saved, give [STAGE_B_EVIDENCE.md](STAGE_B_EVIDENCE.md) and linked source branch | Old Claude review and Stage C | Evidence-supported verification, revised rank, concrete action; freeze as ROUND_B_ANSWER |
| **C. Adversarial calibration** | Only after B is saved, give [STAGE_C_PRIOR_REVIEWS.md](STAGE_C_PRIOR_REVIEWS.md) and previous review links | Nothing new beyond bounded source access | ROUND_C_FINAL_REVIEW with GO/PARALLEL/PIVOT/STOP and falsifier |

**No look-ahead, no rewriting an earlier round.** If Claude reads repo or the prior reviews before A is frozen, mark the session **not blind** and restart in a new conversation. Do not let the previous assistant feed its favorite solution as a system message.

## Scope of source access

All numeric claims have GitHub Actions / actual Kaggle score/source lineage references in B. We have not uploaded ChatGPT's full private chat history, users' API tokens, or raw official competition passages into this packet. Source files in public Git history previously raised license/security questions: do not reproduce or quote whole official Kaggle train/test or generated passages in a review, and do not trigger secret-bearing workflows.

No new model training, Kaggle GPU, official Kaggle submission, paid API provider requests or model changes are part of this read-only reviewer task. An explicit user-authorized experiment may follow later if evidence supports one.

## How to handle review results

- Save each Claude-authored answer **as-is**, preserving model/date and whether it had web/repo access; do not invent or backfill missing content.
- A correction from ChatGPT or the owner must go in a **separate reply**, not silently edit Claude's prior answer.
- If claims cannot be corroborated in actual code/logs/source manifests, mark **INCONCLUSIVE**, not verified.
- The agent has not run because no Claude-capable connected invocation is available in this ChatGPT session. These files are a ready-to-send handoff for Claude web/Claude Code, not a claim that an external reviewer responded.

## Research objective

Improve genuinely generalizing balanced accuracy on **18 exact models** and distinguish this from inherited highscore Notebook reproductions, same-dataset historical dev gains, fragile Public-LB selection, or engineering compliance theater. Preserve precise cost and negative results. A useful review may recommend stopping a bad line.
