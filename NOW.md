# NOW — Name That LLM (2026-11-19 14:25 MYT)

## Authenticated score
- **Official Kaggle Public BA: 0.702**, submission **#58514818**, +0.011 vs old 0.691 #58375806. Target >=0.720, gap +0.018. [Verified scored ledger](https://github.com/kolihhan/name-kaggle/actions/runs/44446729620). Attributed upstream Lightning 0.702 implementation, **not an original breakthrough; Private score unknown**. No new Kaggle submission or GPU run this cycle.

## One substantive research decision — Gemma-4 cousin expert NOT ready
- [Top 100 ranked public Kaggle Notebook source scan #44469878828](https://github.com/kolihhan/name-kaggle/actions/runs/44469878828): 100/100 public sources pulled; **only 5** with a novel model mount vs our scored 0.702, all Kirderf V17a/c/d/e/V18a and all use **Gemma 4 12B Instruct**. Their Notebook rank 89–94 is NOT an official numerical Public BA.
- [Actual code-path #44470522542](https://github.com/kolihhan/name-kaggle/actions/runs/44470522542) and [stage context #44470982731](https://github.com/kolihhan/name-kaggle/actions/runs/44470982731) verify a new executable COUSIN conditional token-likelihood stage, Gemma4 + Qwen3-14B, with `cousin_score.py`, fitted `cousin_head.json` prerequisite and `COUSIN_DEADLINE_H=8.80`. Not a mere model-mount decoration, **but runtime execution and score gain are not proven**.
- [Five Kaggle kernel statuses #44471239606](https://github.com/kolihhan/name-kaggle/actions/runs/44471239606): ALL **CANCEL_ACKNOWLEDGED**, none COMPLETE. Historical output `submission.parquet` exists, but is NOT proof COUSIN stage ran successfully on hidden rerun.
- [Prerequisite check #44471611805](https://github.com/kolihhan/name-kaggle/actions/runs/44471611805): no `cousin_head.json` found in the visible files of 26 dataset and 9 kernel input references (one dataset file listing was paginated and not exhaustively checked). **Cannot prove absent in every possible file**, but no mounted prefit head was verified. The scored 0.702 Notebook logs were readable; aggregate signatures show 0 clear error/OOM/skip signals, which does not establish correctness of all stages.
- **Decision: DO NOT submit the canceled/unverified Gemma public fork; no blind extra Kaggle GPU; do not claim >=0.720.** One worthwhile next step would be an original, properly fitted and disjointly evaluated new expert plus the required head. No head/no clean transfer -> stop this lane instead of further code-auditing.

## Other retired directions
- 363 purported fresh GPT proxy examples: **355/363 normalized exact text duplicates** in current OpenPool. [CPU paired overlap #44465398268](https://github.com/kolihhan/name-kaggle/actions/runs/44465398268). Reject as new independent data.
- Strict source/protocol/license filtering of 231,036-row new OpenPool version still yields **8,338 exact-matching rows, 13/18 classes**, zero qualified samples for five GPT sibling labels. [CPU gate #44451612939](https://github.com/kolihhan/name-kaggle/actions/runs/44451612939).
- Nov17 trained ModernBERT source-adversarial C1 vs C0: validation BA **0.0544 vs 0.2144**, rejected. [Colab run #44116912517](https://github.com/kolihhan/name-kaggle/actions/runs/44116912517).

## Rules
No stronger inference model substitution, no paid APIs, no unauthorized Kaggle exploratory GPU, no adaptive leaderboard tuning, no new weak-proxy iteration. Preserve all data privacy/license checks and exact source pins. Full previous state: [NOW@feac668](https://github.com/kolihhan/name-kaggle/blob/feac668216981d5247f9d68b33471c1ecad45abe/NOW.md); [rules](RULES.md).
