# NTL CPU-only routing headroom and likelihood feasibility — 2026-11-26

No new GPU, model inference, training, Kaggle submission, or scheduler. Owner asked CPU-first research. Latest verified recorded Public BA 0.718 (#58755304), external reproduction; 31B v4 #58935812 last recorded pending, not independently rechecked in this turn.

## Executed new aggregate reanalysis
Parsed the authenticated archived job 139037878216 from [run #45565106275](https://github.com/kolihhan/name-kaggle/actions/runs/45565106275), specifically NTL_FABLE_V2_REAL_FULL18_BOTTLENECK JSON. No raw competition text copied. Spent publisher-labelled diagnostic, 252 rows/14 per class, 151 correct, 101 wrong. These are not hidden or independent results.

| Specific true→predicted edges | Wrong rows potentially rescued | Currently correct rows with the implicated predicted labels |
|---|---:|---:|
| Gemini 3.7→3.8 (8), 3.6→3.1 (4) | 12 | 19 |
| Kimi→Luna (5), GPT5.5 (3), Sol (2) | 10 | 30 |
| GLM full→Opus5 (3), GLM Flash→Opus5 (5) | 8 | 13 |
| Total (distinct destination labels) | 30 | 62 |

Oracle rescue bound 30/252 = 0.11905 on this panel, unrealizable at inference. To illustrate a +0.042 BA design scale, ceil(0.042*252)=11 net rows on this balanced panel. At 40%, 50%, 60%, 70% targeted rescue, the corresponding rescue counts are 12,15,18,21 and maximum allowable regressions are 1,4,7,10. At 50%, the destination-correct regression ceiling is 4/62=6.45%; other regressions would reduce this allowance. This is not an estimated attainable gain nor a target fitted to these spent rows. E5's 4 rescues / 7 regressions illustrates why pairwise discrimination alone is inadequate.

## First priority: quantify routing ceiling before model selection
The existing scripts/ntl_gemma4_31b_ensemble_stage_20261126.py builds a single 18-label reference prompt and restricts hidden decisions to top-2 rows where the current final prediction is one of the top two logits; selects lowest margins, at most 600, wall-clock limit 1700 s. Thus even a perfect expert cannot rescue truth labels excluded from its top-two route. Its source uses final-logits.npz; archived 252 outputs contain only hard predictions, not matching logits. Do not fabricate top-k coverage. Current source is the base stage, not a claim that all generated v4 notebook contents are identical.

CPU next test on a legally qualified, aligned, labeled cache: verify row IDs, exact label order and source version, then calculate oracle net BA for top-2/top-3/top-5 and a fixed runtime-limited prefix; distinguish missed truth outside top-k, missed eligible rows from budget, expert error and regression. No reranking training, no change to hidden submission. Report class-weighted oracle rather than raw accuracy if class counts differ. A 31B expert cannot compensate for a routing ceiling lower than the targeted gain.

## Conditional style likelihood feasibility
EMNLP 2026 paper https://aclanthology.org/2026.findings-emnlp.14/ motivates testing reference-conditioned passage likelihood rather than direct label output. It does not establish competition gains. New source verification: llama.cpp server docs https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md describe n_probs on GENERATED tokens; issue https://github.com/ggml-org/llama.cpp/issues/27174 documents missing prompt/echo probabilities. Native perplexity implementation https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp uses llama_get_logits_ith for likelihood, suggesting a native C API path. Exact pinned 31B runtime not tested; do not interpret chat-completion generated probabilities as teacher-forced passage probabilities.

CPU-first next engineering gate: construct fixed tokens prefix_A + target and prefix_B + target with exactly identical target token IDs, request logits for positions preceding each target token, gather log-softmax target entries, reduce mean difference and compare shuffled-reference and neutral-prefix controls. Check off-by-one, batch reset, UTF-8 boundary, masks, truncation and unchanged target token count with synthetic logits before any 31B inference. Use reference prompts disjoint from target; no threshold fitting on the spent panel. Candidate only if lawful independent same-row 18-way predictions demonstrate net gains and runtime suffices. Native C API route needs actual pinned build validation.

## Blockers
No new authentic independent full-18 cache and no verified exact 31B teacher-forced runtime. The prvsiyan panel shares upstream provenance/173 exact Gemini3.7 texts with current Open Pool; not independent against 0.718. Current latest official Kaggle score not live-verified. No actual new candidate performance claim. Decision: DIAGNOSE routing coverage and likelihood extraction; no GPU or submission until evidence gate passes.
