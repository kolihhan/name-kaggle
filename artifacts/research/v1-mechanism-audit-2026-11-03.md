# V1 Mechanism Audit — Verified Post-Hoc Evidence

Date: 2026-11-03 MYT

## Status

**POST-HOC DIAGNOSTIC ONLY.** This report explains already-consumed OOF evidence. It is not candidate-promotion evidence, does not change the frozen DCT v3 contract, does not authorize a blend/router, and does not authorize a Kaggle submission.

## Reproducibility

Public GitHub Actions run: `41588641409` — **SUCCESS**.

Evidence artifact:

- name: `v1-mechanism-audit`
- artifact ID: `13208529100`
- digest: `sha256:eb9f52a383dc68a10bdf364f8bd48b7ae98def59bac305cfbebd904973c28db6`

The workflow downloaded the immutable completed artifacts:

- Gate-B evidence artifact `13125872633`;
- Phase14 sparse artifact `13067907516`.

It then ran `tests/test_analyze_v1_mechanism.py`, executed `scripts/analyze_v1_mechanism.py`, verified the known reference balanced-accuracy values exactly, and uploaded JSON/Markdown reports.

## Reproduced anchors

- v1 plain OOF BA: `0.5467611235396471`
- v1 centered OOF BA: `0.5478498632861049`
- Phase14 sparse OOF BA: `0.4178100919711658`
- official OOF rows: `2,699`

## What the remaining v1 errors look like

Plain makes `1,223` row-level errors.

- same-family errors: `701` (`57.318%`)
- cross-family errors: `522` (`42.682%`)

The largest plain confusion pairs are dominated by sibling/version ambiguity:

1. `gemini-3.7-flash -> gemini-3.8-flash`: `50`
2. `gpt-5.6-sol -> gpt-5.6-terra`: `40`
3. `gemini-3.8-flash -> gemini-3.7-flash`: `37`
4. `glm-5.3 -> glm-5.3-flash`: `35`
5. `glm-5.3 -> claude-fable-5`: `34`
6. `gemini-3.7-flash -> gemini-3.6-flash`: `31`

**Interpretation:** v1 has learned substantial provider/model fingerprint signal, but a majority of its remaining mistakes are not gross provider mistakes. The hard region is nearby model versions plus a smaller set of genuinely cross-family ambiguous cases.

## Length is a real descriptive bottleneck, not only label mix

The simple global quartile view shows accuracy increasing strongly with text length, but label mix could confound that comparison. The audit therefore also splits each true model at its own median word count.

Result:

- evaluable labels: `18 / 18`
- labels where the longer half has higher plain accuracy: `16 / 18`
- macro long-minus-short plain-accuracy delta: `+0.1351122618`

Only `gemini-3.8-flash` is meaningfully negative (`-0.02525`); `gpt-5.6-terra` is flat. Several hard classes show large gains with more text, including `gpt-5.5`, `claude-opus-4-8`, `kimi-k3`, and `gemini-3.7-flash`.

**Interpretation:** the short-text weakness is broad across labels, so it is not explained only by some difficult model classes happening to write shorter answers. More response evidence materially helps exact-model attribution.

## Why centered / sparse can help without being stronger overall

### Centered vs plain

- rescues plain errors: `353`
- breaks plain-correct rows: `350`
- net row correctness: `+3`
- rescued same-family plain errors: `207`
- rescued cross-family plain errors: `146`

Centered is therefore almost exactly neutral in aggregate while being highly non-identical row by row. Its value is complementary information, not a globally superior decision rule.

### Sparse vs plain

- rescues plain errors: `293`
- breaks plain-correct rows: `641`
- net row correctness: `-348`
- rescued same-family plain errors: `193`
- rescued cross-family plain errors: `100`

Sparse is much weaker overall, but still independently solves hundreds of LoRA failures, especially same-family failures. This is the mechanism behind the large oracle gap; it does **not** by itself justify overriding plain predictions.

## Agreement structure explains the observed ensemble potential

Mutually exclusive hard-prediction patterns:

| Pattern | Rows | Plain acc | Centered acc | Sparse acc |
|---|---:|---:|---:|---:|
| all three agree | 829 | 0.8191 | 0.8191 | 0.8191 |
| plain = centered != sparse | 736 | 0.6073 | 0.6073 | 0.1440 |
| plain = sparse != centered | 358 | 0.4358 | 0.2933 | 0.4358 |
| centered = sparse != plain | 216 | 0.2176 | 0.4769 | 0.4769 |
| all distinct | 560 | 0.2625 | 0.2589 | 0.1500 |

This makes the earlier three-way-majority diagnostic understandable: agreement between independently biased views often carries information. But these conditional accuracies were measured on already-consumed OOF and must **not** be turned into a new router or threshold table.

## What this says about DCT temporal-fingerprint v3

The current frozen v3 changes mean pooling into fixed DC+AC1 temporal pooling. That mechanism is well aligned with one observed limitation: v1 currently collapses the whole response-token trajectory into a mean, while exact-model evidence becomes markedly stronger as more response text is available.

However, this audit also gives a useful explanatory warning:

- short responses are the hardest region;
- a temporal AC feature has less trajectory to exploit when text is short;
- therefore a v3 gain driven only by long passages would not mean the short-text bottleneck has been solved.

This is **not** a new PASS condition. The v3 contract was frozen before this audit and remains unchanged. The point is mechanistic interpretation after the already-predeclared transfer gate runs.

## What not to do next

Do not:

- derive a router from the agreement table;
- tune centered/sparse weights on these OOF rows;
- resurrect prompt-cluster Hungarian constraints from train structure;
- tune DCT modes after seeing v3;
- add more external rows merely because the pool is larger.

Current public evidence also contains a cautionary example where prompt clustering recovered train prompt groups extremely well and Hungarian assignment produced a large OOF gain, yet failed to improve the real leaderboard. Structural train assumptions can therefore create convincing but non-transferable gains.

## Breakthrough hypotheses if v3 eventually FAILs

These are **research hypotheses only**, not frozen candidates yet.

### 1. Centered-to-plain privileged-information distillation

Use prompt groups only during training. Let the centered head act as privileged training information, but distill its prompt-conditioned evidence into the deployable plain head. Hidden-test inference remains one passage at a time and requires no `prompt_id`.

Why this is materially different from rejected v2: v2 pulled exact-model representations toward cross-prompt prototypes. Distillation would instead transfer the already-observed same-prompt nuisance-cancellation signal directly into the deployable head.

### 2. Fixed local-statistics pooling for short text

If v3 shows that AC1 helps mainly on long passages, test a fixed representation that preserves mean plus local dispersion/segment information rather than adding more DCT modes. This targets weak evidence density in short passages without introducing learned attention or a mode sweep.

### 3. Explicit sibling-separation objective

The dominant remaining plain error mode is same-family confusion. A future candidate could emphasize within-family hard negatives while keeping exact-model classification primary. This must be a new frozen mechanism, not a post-hoc family router derived from these OOF errors.

## Anti-overfit process recommendation after v3

Fresh source holdouts are better than reusing official folds, but repeatedly spending a new source subset for every idea can still meta-overfit the finite public source pool.

After v3 resolves, do not launch a sequence of v4/v5/v6 candidates each consuming another three sources. Before the next candidate, freeze a transfer-budget policy: development evidence versus a reserved final source-shift firewall, or obtain a materially independent pre-competition corpus. A candidate should earn access to the reserved transfer evidence through mechanism-level tests first.
