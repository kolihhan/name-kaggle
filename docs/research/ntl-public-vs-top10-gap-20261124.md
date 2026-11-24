# NTL competition positioning: published notebooks are **not** top leaderboard solutions
Date: 2026-11-24 (MYT). This report distinguishes **independently verified Kaggle notebook source** from public scoreboard and author-claimed title score.

## Verified competitive context
- [Public leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard) remains live; ~40% public test score, 60% private.
- [Independent daily snapshot (23 Nov 2026, 18:00 Eastern)](https://autoresearch-arena.github.io/contests/name-that-llm/): reported leader **YK 0.809**; top 10 cutoff **0.774**, 15th **0.761**. **These are dated figures, not an authenticated live Kaggle snapshot**; repeat before making claims for the present day.
- Account's verified best **0.718** (submission `58755304`), reproducing an outside publicly shared Notebook, not a new winning method.
- At the Nov23 snapshot, the alleged best public code Notebook is **0.721** by title; gap to 0.809 ~**8.8 pp**. Account gap 9.1 pp. Last 0.774 top10 margin from account is 5.6 pp.
- Contest final notebook close is **2027-01-13 23:59 UTC** per official [timeline](https://www.kaggle.com/competitions/name-that-llm/overview/abstract), so top competitors are not required to reveal private submission source until contest closes. **No independently authenticated top-team full scoring pipeline found**, and no extrapolated winning architecture should be presented as a fact.

## Actual PUBLIC source archaeology, authenticated read only
[GitHub Actions #45420401563](https://github.com/kolihhan/name-kaggle/actions/runs/45420401563) pulled exactly these public Kaggle kernel sources (no training or Kaggle submission):
1. [zayd88903/0-721-name-that-llm](https://www.kaggle.com/code/zayd88903/0-721-name-that-llm), 29 code cells, ~250.7k code chars, 9 model/26 dataset/10 kernel references.
2. [areebbb/0-721-preserve-glm-to-fable-corrections](https://www.kaggle.com/code/areebbb/0-721-preserve-glm-to-fable-corrections), same number of code cells/chars, same nine model source refs, 26 datasets, ten kernel refs and all 13 written code modules matched Zayd's corresponding modules by SHA. Full ipynb SHA differs, so don't say byte-identical.
3. [alexchilton/name-that-llm-0-719-what-moved-the-board](https://www.kaggle.com/code/alexchilton/name-that-llm-0-719-what-moved-the-board), 31 code cells ~256.3k chars; the same 9 model, 26 dataset, 10 kernel reference counts and 12/13 module source SHAs match Areeb/Zayd, one differing source cell.
- Combined high-level mechanics: heavily inherited neural/multimodel stack with Qwen-3-base / Qwen2.5 variants, Llama 3.1 8B, embedding features, native GLM-5.3/Flash likelihood, style/sibling residual, sinkhorn balancing, Fable-guard logic. These are metadata/code-presence observations, **not proof each module runs or improves the Public score**. Some notebook titles claim 0.721 but this read-only audit did NOT independently associate title score with an official scored submission.
- Current public frontier appears to be incremental variants of a shared donor pipeline. These cannot explain leader 0.809 from available code. **Stop treating replication of 0.721 as research toward a 0.80 winner**.

## Competitor data clues, not documented winning solutions
- [150 reconstructed official training prompts](https://www.kaggle.com/pjmathematician/discussion): Claude inferred labels/prompts from 17–18 same-prompt outputs. Reconstructions are estimates, not original prompts. Author reports broad task distribution (pop science, vivid narrative and analogy, ~40–250 words), but reconstruction confidence is model self-rated and not independent validation.
- [External 7,753 responses / 10 classes](https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations) with per-row API route/provider/served model metadata, but generated prompts, routing and protocol vary. Don't falsely assume label means same exact model/settings as the competition. Even 7,753 rows only cover 10 labels and cannot by themselves solve 18-way exact attribution.
- [Kaggle community thread on prompt uniqueness test](https://www.kaggle.com/competitions/name-that-llm/discussion/751802): matching 1 response per source per inferred test prompt raised *local* BA ~0.374→0.481 but decreased Public 0.353→0.350. Evidence of train/test protocol shift; do not enforce a 1-per-source Hungarian constraint on hidden test.
- Unseen source-specific generation protocol is a plausible bottleneck. Claims like 'top teams certainly use private data' are **speculative** without their code.

## 3 best next investigations, ordered and not conflated with experiments
1. **Incumbent 0.718 per-class error headroom**: gather actual same-row 18-way source-pinned error matrix or a strong full18 external panel where champion was genuinely run, not the 0.718 3-row fast-commit preview. Calculate the GLM, Gemini, OpenAI Luna/Sol/Terra, Claude siblings' confusion contributions and compare potential BA gains vs runtime. Current earlier frozen selftests never produced a full 0.718 genuine labelled matrix.
2. **Generator/source protocol fidelity**: score the 18-model public datasets by verified canonical model ID, prompt reconstruction fidelity, text length and preprocessing parity, fallback provider, deduplication, licensing, prompt independence. Separate 'more rows' from 'genuine data matching official generative settings'. No small model training. Avoid arbitrarily blending external datasets.
3. **Strong native generator signals**: source-conditioned likelihood where original or related family LLMs are truly available, with a mixed-family gate and exact same-row rescue/regression against incumbent. GLM-only reroute has been studied and did not establish hidden improvement. Do not reimplement the same GLM threshold experiment.

## Scientific stop/scope decisions
- `KNOWN`: verified ~0.718 own-account score, Nov23 0.809 leader snapshot, public 0.721 notebooks share 13 code-module sources. There is a >5.6pp top-10 gap, not a 0.002 tweaking problem.
- `UNKNOWN`: top teams' training data, model size, code, validation split and true Private scores. Do not invent 'top solution' nor claim score from Notebook title as official verified.
- `PARK`: small-model fine-tuning; user explicitly declined it for NTL.
- `NEXT`: rank/verify truly competitive source strategies, seek original mechanism that creates new identifying evidence at 18 classes, not a reused public ensemble wrapper; independent evaluation and Kaggle 9-hour ceiling gate required before any official submission.
