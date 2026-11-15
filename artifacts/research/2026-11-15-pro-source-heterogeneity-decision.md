# PRO mechanism study: sibling fingerprint vs source/protocol artifact

Status: **executed, descriptive negative control** (2026-11-15). Not a new classifier, independent holdout, or official BA result.

**Hypothesis H1:** GPT-5.6 Sol text is inherently longer than GPT-5.6 Luna for the same prompt across sources. Predict stable Sol-longer rates.
**Hypothesis H2:** pairwise length is dominated by prompt source, requested length, generation harness/protocol and other confounding. Predict major source-dependent changes.

## Executed evidence

Read-only CPU [Action #43682442089](https://github.com/kolihhan/name-kaggle/actions/runs/43682442089) SUCCESS. Source [script](../../scripts/ntl_pro_source_heterogeneity.py), [3 synthetic tests](../../tests/test_ntl_pro_source_heterogeneity.py). Frozen previously authenticated pair counts:

| Existing source | Sol longer than Luna | Percentage |
| --- | ---: | ---: |
| Official competition train, 150 matched prompt pairs | 75/150 | 50.00% |
| SWEeper agentic snippets, matched source/prompt | 195/196 | 99.49% |
| Proxy primary, matched prompt and setting | 20/47 | 42.55% |

Posthoc exact two-sided Fisher comparison SWEeper vs official: p=2.74e-32; SWEeper vs proxy: p=4.71e-22. These figures quantify descriptive source heterogeneity, not causality or fresh validation. In proxy, the two models were asked for different target lengths for all 47 pairs, and the observed length ordering matched the requested target ordering on 47/47 pairs. Source, task, protocol and backend revision remain entangled.

## Decision

- **Reject** universal source-blind Sol-longer length heuristic and per-source fixed-label thresholds. Do not tune a blender to these spent data.
- **Do not conclude** intrinsic generator signals are nonexistent; this does not test all representations, and no balanced accuracy was measured.
- **Next competing representation test:** compare a frozen champion with one length-normalized conditional-likelihood or prompt/protocol-robust sibling residual candidate on the *same* externally source/prompt-disjoint rows, stratified by model siblings. Exact 18-class promotion requires verified support of all 18 classes plus prefit exposure audit; partial class evidence is only exploratory.
- **Falsifier:** absent net sibling rescues on independent matched rows, retire that specific representation. If positive, corroborate on a second unused source before allocating one official submission. Respect 9h limit.
- **Current official best stays 0.683.** Lightning's 1200-text test is a throughput study only, not attribution accuracy.

Provenance: [official matched length](ntl-official-sibling-length-negative-control-20261112.md), [cross-source/protocol audit](generator-vs-channel-identification-decision-20261106.md), [existing 84-row complementarity](ntl-areeb8-vs-wemm-paired84/RESULT_20261114.md), [independent validation gate](ntl-18class-independent-validation-gate-20261111.md). No GPU, training or official submission from this research.
