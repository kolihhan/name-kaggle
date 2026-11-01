# Domain separability transfer audit — result

Date: 2026-11-01

## Verdict

**The simple “Phase 15 failed because its external corpus has more obvious source/format fingerprints” explanation is not supported.**

After controlling for model identity and holding out prompt groups deterministically, the known-transfer Phase 7 external corpus was **easier** to distinguish from official competition responses than the rejected Phase 15 corpus under both fixed classifiers.

This result should be read together with Phase 18 and Phase 19:

- Phase 19 showed that Phase 15's local advantage over Phase 7 survives all three frozen alternate prompt-group assignments.
- Phase 18 showed that Phase 15 external responses are materially closer to official competition responses than the Phase 7 external control under a common nearest-neighbor audit.
- This audit shows that Phase 15 is also **less domain-separable** from official responses under simple lexical and style/format classifiers.

The evidence therefore points away from one-split luck and away from a crude obvious-format-fingerprint story. The stronger remaining hypothesis is **overmatching to the official-train/public-corpus distribution**: Phase 15 learns signal that generalizes across held-out groups inside the 2,699 official rows, but the hidden prompt/topic/construction distribution differs enough that the gain does not transfer.

That is still an inference, not proof of the exact hidden shift.

## Frozen setup

The audit was frozen in `artifacts/domain-separability/CONTRACT.md` before results were observed.

For each shared model label independently:

- classify `official` vs `external`;
- deterministic prompt-group holdout with salt `ntl-domain-separability-v1`;
- no prompt group crosses train/test inside a domain/label;
- style/format classifier: fixed `StandardScaler + LogisticRegression(C=1.0, class_weight=balanced)`;
- lexical classifier: fixed word TF-IDF 1–2, lowercase, `min_df=2`, max 20,000 + the same logistic regression;
- no hyperparameter search, threshold tuning, source filtering, or competition-model change.

Phase 7 comparison covers its 8 shared external labels. Phase 15 comparison covers its 11 terms-clean external labels.

## Results

| Comparison | Style/format domain BA | Lexical domain BA |
|---|---:|---:|
| **Official vs Phase 7 external control** | **`0.7365883583`** | **`0.7552909188`** |
| **Official vs Phase 15 external** | **`0.6627327934`** | **`0.5810843766`** |

Across-label sample SD:

- Phase 7 style: `0.0413121228`; lexical: `0.0560075513`.
- Phase 15 style: `0.0530087005`; lexical: `0.0494759553`.

Ranges:

- Phase 7 style: `0.65747–0.78485`; lexical: `0.67342–0.82527`.
- Phase 15 style: `0.59067–0.77668`; lexical: `0.49549–0.65562`.

Phase 15 lexical separation is close to chance for some labels, including `gpt-5.6-sol` at `0.49549`.

## Per-label Phase 15 domain BA

| Label | Style | Lexical |
|---|---:|---:|
| claude-fable-5 | `0.59067` | `0.59333` |
| claude-opus-4-6 | `0.63736` | `0.62546` |
| claude-opus-4-8 | `0.60035` | `0.62148` |
| claude-opus-5 | `0.62383` | `0.60470` |
| claude-sonnet-5 | `0.65238` | `0.60339` |
| gpt-5.5 | `0.69401` | `0.54139` |
| gpt-5.6-luna | `0.77668` | `0.55665` |
| gpt-5.6-sol | `0.69612` | `0.49549` |
| gpt-5.6-terra | `0.68876` | `0.51312` |
| gpt-6-astra | `0.68716` | `0.58131` |
| qwen3.8-max | `0.64273` | `0.65562` |

## Style signals

Both external corpora still differ measurably from official responses in simple style statistics. The strongest recurring signals include period rate, digit rate, uppercase rate, token-length/whitespace structure, and non-ASCII rate.

But these signals are **not stronger overall for Phase 15 than for Phase 7**. Therefore their existence alone cannot explain why Phase 15 transfers worse.

## Evidence classification

### FACT

- Phase 15 beats Phase 7 on the permanent grouped OOF and all three Phase 19 shadow assignments.
- Phase 15 loses to Phase 7 on the real Public leaderboard (`0.357` vs `0.369`).
- Phase 18 found Phase 15 external text materially closer to official responses than Phase 7 external under common cross-model nearest-neighbor statistics.
- This audit finds Phase 15 external text less separable from official responses than Phase 7 external under both frozen model-controlled domain classifiers.

### INFERENCE

The current evidence is most consistent with **official-distribution overmatching / external-data transfer mismatch**, not with a single bad fold assignment and not with a simple obvious-source-fingerprint explanation.

In other words, the Phase 15 corpus appears to make the classifier better at the distribution represented by the official 2,699 training responses, including across held-out official prompt groups, without making it better on the hidden distribution.

### UNKNOWN

The exact mismatch cannot be identified from the available labels. Plausible dimensions still include hidden prompt/topic composition, generation settings, response-length distribution, provider/source construction, temporal effects, and other latent dataset-construction differences.

The Public leaderboard covers only part of the hidden test, so it is not an oracle for the final Private distribution either.

## Consequence

Do **not** respond by removing Phase 15 sources, adjusting source weights, tuning SVC parameters, or optimizing a domain-classification threshold. Those would be post-hoc choices made after observing both OOF and Public LB.

The next modeling work should be structurally less dependent on naïvely concatenating a large external corpus with official fold-training rows. Two-stage representation learning and complementary honest-OOF models remain reasonable directions to brainstorm, but only after a future evaluation policy is frozen.

## Evidence

- workflow run: `41285318226` — SUCCESS
- artifact: `domain-separability-transfer-audit`
- artifact ID: `13074629702`
- artifact digest: `sha256:182f177cfb4d83613dece9dcd684955002c58541ae15f8f010ce832dcf696652`
- exact Phase 15 corpus rebuilt from the pinned raw pool: 26,025 rows / 2,801 prompt IDs
- Kaggle submission authorized by this audit: **false**
