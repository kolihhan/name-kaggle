# Name That LLM — Phase 1: Honest Validation Baseline

## Dataset audit
- Official train rows: 2,699
- Prompt groups: 150
- Candidate models: 18
- Missing values: 0
- Duplicate full rows: 0
- Duplicate texts: 0
- Prompt structure: 149 prompts have 18 rows; 1 prompt has 17 rows
- Missing pair: `p002 × claude-fable-5`
- Frozen source SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`

## Validation contract
- 5 folds
- Group key: `prompt_id`
- Canonical row order before fold assignment: `id` ascending
- Every prompt lives in exactly one validation fold
- 30 prompt groups per fold
- Prompt overlap between train/validation: 0 in every fold
- Fold sizes: 540, 540, 540, 540, 539 rows
- Permanent mapping: `folds.csv`

## Honest OOF baselines
| Experiment | Balanced accuracy |
|---|---:|
| Word TF-IDF (1–2) + LinearSVC | 0.26298 |
| Char TF-IDF (3–5) + LinearSVC | 0.30226 |
| Word + char TF-IDF + LinearSVC | 0.30151 |
| Hierarchical lab→model char TF-IDF | 0.30340 |
| Char TF-IDF (1–5), balanced LinearSVC | **0.31930** |

## Error profile from char (3–5)
- Total OOF errors: 1,883
- Same-lab errors: 899 (47.7%)
- Cross-lab errors: 984 (52.3%)
- Therefore sibling-model confusion is important but not the whole problem.

## Decision
Do not spend more time on hierarchy or small sparse-feature tweaks. Keep `0.31930` as the current local classical baseline and move to the next evidence-backed axis: public external generations + encoder/LoRA, while keeping these exact frozen folds for every experiment on the official train set.
