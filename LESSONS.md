# LESSONS — NTL (2026-11-16)
Canonical score provenance: [NOW](NOW.md) and [VALIDATION](VALIDATION.md). Categories: 已证实 / 已证伪 / 证据不足 / 未测试. Do not imply a future transfer test has run.

| Item | Status | Grounded evidence / scope |
| --- | --- | --- |
| Current Public best is 0.691, not 0.694 claimed in a title | **已证实** | #58375806, [authenticated Action 43743764085](https://github.com/kolihhan/name-kaggle/actions/runs/43743764085); official score, not Private |
| Oleg sensitive-topic-only external stress test gives 0.412 | **已证实, diagnostic only** | 89/216 [Action 43804320931](https://github.com/kolihhan/name-kaggle/actions/runs/43804320931); no independent Kaggle-distribution claim |
| Oleg non-sensitive cohort can certify a full 18-class matched holdout | **已证伪 for THIS source** | 427 valid cells only 10/18 labels and four complete 10-label shared prompts [Action 43808061989](https://github.com/kolihhan/name-kaggle/actions/runs/43808061989) |
| External zayd8890 collected corpus qualifies as independent 18-class benchmark | **证据不足 — 未准入** | 18 static aliases, actual confirmed per-label prompt counts 0 verified; OpenRouter collector omits served-ID, DVC bytes not openly auditable, license/prefit overlap unknown. [Static source audit](artifacts/research/2026-11-16-pro-source-zayd8890-metadata-admission.md); **n=0 scored, CI N/A** |
| Free Kaggle Benchmarks proxy dataset covers all 18 labels for an independent benchmark | **已证伪 for this source; not admitted** | [Source data card](https://www.kaggle.com/datasets/zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations) names GPT/Gemini only (max 9/18 competition labels). [Audit](artifacts/research/2026-11-16-kaggle-proxy-source-eligibility.md), evaluated **n=0**, accuracy CI N/A; other conditions unknown |
| 0.691 proves genuine hidden-domain performance gain beyond Public or mechanism originality | **证据不足** | no comparable OOF or Private; no Public tuning |
| Zhou team Prompt Recovery lesson — suffix shortcut not reliably transferring | **已证实 as reported team observation, untested here** | [joint original writeup](https://www.kaggle.com/competitions/llm-prompt-recovery/writeups/z-d-z-5th-place-solution) |
| 540 equal-class responses can robustly detect a +0.009 BA gain | **已证伪 under optimistic paired independence sensitivity assumptions** | CPU power approximation: if paired discordance 10–30%, nominal 80%-power delta at N=540 is ~0.038–0.066, larger with clustering. See [VALIDATION.md](VALIDATION.md). NOT empirical Kaggle accuracy |
| Source-disjoint new generator/protocol test, family-diverse representation, complementary ensemble | **未测试** | See [IDEAS](IDEAS.md) and method card |

**Decision audit (round 5/5 COMPLETED):** [10+ previously made decisions reviewed](artifacts/research/2026-11-16-round5-small-n-decision-audit.md); general conclusions from underpowered n=3/4 Gemma tasks and narrow n=12/class Oleg or 12-label NTL sources **REOPENED**, while exact preregistered candidate stop decisions remain. The previous general anti-small-model/encoder/protocol elimination is not justified. No new model/GPU/submission from this audit.
