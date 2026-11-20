# Phase 7 → Phase 15 source diff and the OOF→Public gap — 2026-11-20

## Verdict

**Completed source/data-lineage review; no new model fit or submission.** The prior claim that k=2 resampling of fixed OOF predictions refutes sparse-group inference was too strong: it only increases sampling variability of a fixed row-level scorer. More importantly, **both submitted Phase 7 and Phase 15 implementations make each test-row prediction without Hungarian, prompt centering, matching, or using other hidden-test responses. Therefore full-prompt-group dependence is not a credible explanation of this exact Phase 7→15 reversal.**

There is **not enough evidence to assign the precise 0.0483 Phase15 OOF/Public gap** to any single mechanism. This code review identifies incompatible source coverage plus architecture change, adaptive OOF model selection, and possibly source/serving mismatch as rival explanations. Exact-response prompt overlap has already been audited and not found; semantic/content-level overlap remains unresolved.

## Exact observed discrepancy

| | Phase 7 | Phase 15 | Phase15−Phase7 |
|---|---:|---:|---:|
| Historical official TRAIN grouped OOF 18-class BA | 0.370052 | 0.405322 | +0.035270 |
| Authenticated Public BA | 0.369 | 0.357 | −0.012 |
| OOF−Public difference | 0.001052 | **0.048322** | +0.047270 |

The last row is a **difference of generalization gaps**, not a measured source-data leakage effect. Public is rounded and can vary on re-execution, so these point estimates should not be assigned stronger precision than the score receipts support.

## Actual submitted implementation changes (not a single-variable ablation)

| Dimension | Phase 7 | Phase 15 |
|---|---|---|
| Architecture | Original 18-label sparse anchor + separately trained 8-label specialist, gated on current-row **anchor predicted class** | A single 18-label `LinearSVC` with four TF-IDF views (char, word, shape, function-word) |
| External corpus | Pinned 2,503 rows of `abdullah1afify/ntl-generations` | Pinned `nishantdahal/ntl-handwriting-blend` pool: 75,783 raw → 43,781 conservative → **26,025** terms-clean |
| External class support | **8**: Gemini (excluding 3.7) and GPT | **11**: Claude 5, GPT 5, Qwen 1 |
| Specifically missing external class support | 10 (anchor remains for most of them) | **7**: all 4 Gemini, both GLM, Kimi |
| Per-fold training | 4/5 official fit + complete 2,503 external in specialist only; original 18-class anchor fits official training only | 4/5 official fit + complete 26,025 external, **all through one global feature vocabulary and classifier** |
| Prediction structure | Per-row anchor→specialist gate; not pairwise or group-based | Per-row 18-class SVM; not pairwise or group-based |
| Runtime | CPU, no GPU for inference | CPU, no GPU for inference |

Source evidence:
- [Phase 7 exact OOF source](../../scripts/run_gated_specialist.py) and [Phase 7 first scored run](../../artifacts/submission/FIRST_SUBMISSION.md);
- [Phase 15 OOF source](../../scripts/run_phase15_terms_clean_sparse.py), [TF-IDF fitting source](../../scripts/run_phase14_public_sparse.py), [official scoring inference source](../../scripts/phase15_submission.py), [Notebook bundle](../../scripts/build_phase15_kaggle_kernel.py), and [Phase 15 official receipt](../../artifacts/submission/SECOND_SUBMISSION.md).

**Important:** A source-level diff across these two systems cannot causally attribute their difference to “adding external data” because *architecture, source data, training populations and gating change together*. Likewise Phase 7 and Phase 15 exact OOF differences cannot prove an inference-group error because neither system depends on the hidden-test group.

## Quantified class-support tradeoff already in original authentic outputs

Both original OOF runs used the exact same official 2,699 IDs, same 150 grouped folds, same true labels; old outputs were joined by ID and re-rated in [the original paired audit](../ntl-ruler-2699-prompt-oof-reliability-20261118.md).

- **11 classes represented in Phase15 external data**: unweighted average recall delta **+0.128017**.
- **7 classes without external Phase15 support**: average recall delta **−0.110476**.
- Overall macro delta: `(11×0.128017 + 7×(−0.110476))/18 ≈ +0.035270`.

Largest losses:

| Model | Phase7 recall | Phase15 recall | Delta |
|---|---:|---:|---:|
| Kimi K3 | 0.2400 | **0.0467** | −0.1933 |
| GLM 5.3 Flash | 0.2000 | **0.0133** | −0.1867 |
| GLM 5.3 | 0.1800 | **0.0200** | −0.1600 |
| Gemini 3.8 Flash | 0.4800 | 0.3533 | −0.1267 |

Largest gains:

| Model | Phase7 recall | Phase15 recall | Delta |
|---|---:|---:|---:|
| Claude Sonnet 5 | 0.3800 | **0.6733** | +0.2933 |
| Claude Fable 5 | 0.1409 | **0.3691** | +0.2282 |
| Claude Opus 4.8 | 0.2067 | **0.4333** | +0.2267 |

Per-class source: [Phase 7 pinned aggregate](../../artifacts/phase7/gated_specialist.json), [Phase15 pinned aggregate](../../artifacts/phase15/terms_clean_sparse.json).

**Mechanism candidate, not causal proof:** Phase15 may strongly optimize decisions for externally represented labels at the expense of missing/host-mismatched labels. `class_weight="balanced"` is used; calling this *just a prior/class-count bias* would itself be an unjustified simplification. TF-IDF vocabulary, normalization, discriminative boundaries, source-specific lexical signatures and training data may all change.

Phase14→15 **does** hold architecture fixed and drop only provider-terms-blocked external classes: OOF goes **0.417810→0.405322** (−0.012488). No authenticated Phase14 Public receipt exists in the cited records; do not use this contrast to explain Phase15's Public gap.

## Leak hypothesis: already tested, narrower than “not leaking”

[Actual Phase15 external-vs-official audit, Actions #44213134245](https://github.com/kolihhan/name-kaggle/actions/runs/44213134245) confirmed:
- GroupKFold: **0** official train/heldout prompt-ID intersection in all five folds.
- Same-normalized-response hashes across official train/heldout: **0**.
- All 26,025 external examples enter every fit fold; **0 exact normalized response matches with the heldout official rows**.
- **0 literal external prompt-ID/prompt-key string equals heldout official prompt-ID**; however different source ID namespaces make this weak evidence for shared or disjoint *semantic* prompts.
- No external rows are scored as validation rows, but **being only in training is not proof that their prompt bank is independently sourced**.

Hence the strong allegation of *documented identical-sample or literal prompt-ID leakage* is not established. **Remaining unknown:** paraphrases, host prompt reconstruction in external data, shared templates, source aliases, generated-model version/serving differences. These require actual prompt-generation provenance, not another random fold-ID comparison.

## Selection bias, metric variance, and limits

- Both Phase7→15 OOF estimates were evaluated under projectwide repeatedly-used 150 official prompt groups and triggered model choices. A previously measured paired historical OOF +0.03527 had 95% prompt-bootstrap CI [+0.016761,+0.053469], but bootstrapping reused selection data does **not** estimate true hidden generalization bias.
- The opposing assertion that OOF-selection optimism can contribute “only 0.01–0.02” is **unsupported** without the effective number and dependence of adaptive model/configuration trials and their errors.
- Phase7 OOF≈Public (0.37005 vs 0.369) on one draw can be coincidence. Phase15's single-Public 0.357 can itself contain scoring variability. Neither point reveals the hidden labels, class proportions, or per-class Public errors. We cannot causally decompose the 0.048 gap from two Public score receipts.
- [k=2 retrospective on fixed OOF](../ntl-sparse-prompt-k2-retrospective-20261120/RESULT.md) was a **low-information methodological negative control**: for these row-independent predictions, mean BA remains essentially constant with k; any success at reproducing the Public ranking comes from finite-sample variance. It does **not** refute a prospective sparse-group *re-inference* for a group-dependent method (not present in either Phase7 or Phase15).

## Outcome / next actual decision

1. **REJECT group-inference-size as the explanation for Phase7→15**: neither scored architecture contains group-coupled test inference.
2. **NO EVIDENCE for exact-duplicate prompt leakage**, but source alias, paraphrase/semantic overlap and serving compatibility remain possible.
3. **Highest priority mechanism**: Phase15's *source-class support/representation tradeoff* combined with a CV target that systematically over-rewards host-mismatched external examples. This is a falsifiable hypothesis, not a confirmed explanation of Public collapse.
4. Phase15 was already rejected for competition promotion. **Do not launch another Phase15 weight/label/threshold sweep on the same 150 prompt folds**. If future source-matched data become legally and provenance-admissible, prospectively compare a fixed model recipe with/without the one external-source change on *unseen, source-disjoint* prompts, and test class-specific regression. Otherwise stop this expired lane.
5. 0.702 publicly trained/pretrained full-ensemble OOF remains unavailable with clean fold lineage. **No claim of conditional complementarity to 0.702** from historical Phase7/15 or the 347-row own LoRA24 vs old 0.679 study. That study shows 24 rescues vs 66 regressions and **negative unconditional transfer** on two old sources, not a valid candidate on the 0.702 pipeline.

**Computation for this report:** authenticated source reads and arithmetic on original pinned aggregate JSON only. No new GPU, Kaggle submission, dataset export, or newly trained/OFFICIAL score. 