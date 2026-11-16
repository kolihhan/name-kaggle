# PRO 2026-11-16 — Oleg external prose-only matched-model experiment: feasibility falsified, no new GPU

## Question
Can the actual own scored 0.691 Lightning V2 18-class model be tested on a **reliable non-sensitive-topic same-model control**, correcting the earlier real 216-row Oleg sensitive_topics-only result **0.412**? This is a distinct diagnosis from tuning on the already-inspected 216 examples.

## Authenticated source evidence

- [Full 18-class external short and all-view eligibility #43805155243](https://github.com/kolihhan/name-kaggle/actions/runs/43805155243): **zero qualified non-sensitive Gemini 3.8 Flash** even under weaker model identity evidence. The corpus does **not** support a trustworthy non-sensitive *18-class* benchmark. 
- [Real 40–250-word, source-certified prose cohort #43806583149](https://github.com/kolihhan/name-kaggle/actions/runs/43806583149): **427** valid labeled cells from only **10 of 18** model classes with ≥12 unique prompts each (strong `request_and_returned_model`, known license, text-only suitable, no observed truncation, exact prompt).
- [Real pairwise prompt-group census #43808061989](https://github.com/kolihhan/name-kaggle/actions/runs/43808061989), SUCCESS: the 427 cells occupy just **75** prompt groups. Of these, only **4** have examples from **all 10** qualifying classes; 50 groups contain ≥5 classes. Thus an all-ten same-prompt cohort can have at most **4 fully matched prompts** → 40 responses, high variance and adaptation risk. 120 globally unique source prompt groups are not available. Most of the 427 examples share underlying prompts, and treating repeated prompts as 120 independent test items would inflate precision.
- Qualifying 10 labels: `claude-fable-5`, `claude-opus-4-8`, `claude-sonnet-5`, `glm-5.3-flash`, `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`, `kimi-k3`. Note Kimi-K3 only **14** unique qualified prompts; one filtering change could deplete the cohort.

## Scientific decision

**Do not launch another Kaggle GPU just to report a new 10-class number from 4 shared prose prompts.** A 10-label metric would also be non-comparable to full 18-class Kaggle Balanced Accuracy, and this is only the same Oleg source family as the spent 216 sensitive-topic benchmark.

For any next external model attribution test, source criteria must be preregistered: all 18 exact labels with true *served* identity evidence, source/prompt disjointness and source licensing, non-sensitive-topic/domain coverage (not a single theme), 40–250-word outputs when relevant, known lineage overlap, a meaningful number of **unique prompt groups** per class, and no reuse for tuning. If no such source exists, prioritize designing an actual new independent sampling channel rather than manufacturing apparent gains by relaxing label certainty.

**No new training, GPU run, leaderboard submission, calibration sweep, or adjustment to frozen own 0.691 champion.** The decision to decline weak evidence is the result.
