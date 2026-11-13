# GLM 5.3 vs Flash likelihood diagnostic

The old 0.679 baseline missed the exact sibling within the same provider on 71 of 87 errors in one spent 12-class panel. That panel has no true GLM rows. The scored 0.683 Areeb champion uses surrogate LMs, whereas [Padilla's public version 6](https://www.kaggle.com/code/estopadilla/0-689-real-glm-5-3-likelihood-on-2xt4) adds likelihood under the two actual open-weight GLM models and publishes a frozen 54-feature multinomial head. The author's 0.689 is reported for their own full stack. Its preprocessing, head and baseline are different from ours.

Ranked directions:

1. **Direct model likelihood.** It introduces observed-token evidence from both possible generators. First use [the author's cached model outputs](https://www.kaggle.com/code/estopadilla/ntl-glm-likelihood-train/output), fetched with Kaggle's explicit `v3` output selector, to test whether the raw likelihood ratio is predictive even without a classifier. This is a CPU diagnostic on the 150 already-used official training prompts.
2. **Conditional prior and prompt grouping.** A potentially cheap change. Prior transport tied the earlier 0.679 incumbent, source shifts weakened prompt-based heuristics, and no new residual evidence warrants another change to global decision rules.
3. **Additional frozen ensemble member.** WeMM's paired test is negative. Many public models share the incumbent's weight families. Reopen only with new model-specific evidence.

## Frozen CPU diagnostic

- Join version 3 Flash and GLM-5.3 feature outputs to the existing 2,699-row training index by exact ID. Fail on missing or duplicate IDs or unequal token counts between arms. Use only GLM truth pairs with one output from each model for the same prompt.
- Compare mean log probability under GLM-5.3 minus under Flash. Use sign at zero for pointwise binary classification, with ties split equally. The primary statistic is paired ranking within the true prompt. The ranking across different prompts supplies a topic-matching comparison. Measure both single-model log probabilities with fixed orientation and the contrast in bits per character as controls. None of these diagnostics estimate or fit any parameters.
- Test whether the pair ordering is above 50% using a one-sided exact sign test over non-tied prompts. A positive ranking with poor zero-threshold classification motivates the published frozen readout, whose official-train performance would be in-sample. A negative paired ranking eliminates the simple mean likelihood contrast on this dataset. These data are spent regardless of result. No tuning or promotion from this check.
- For transfer, a separate prospective experiment must use the actual 0.683 champion on the same unseen, source/protocol-qualified rows. Do not use the old 84-row panel, where both GLM labels have zero true support. Check model provenance, real inference cost and the applicable frozen evaluation and authorization gates before any official entry.
