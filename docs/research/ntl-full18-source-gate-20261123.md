# NTL Full-18 prospective source gate — 2026-11-23

## Goal and reasons for this one bounded intake
Find a **genuinely independent, authentic 18-exact-model-label source panel** before testing a new attribution architecture. Current account's officially scored strong Champion is **0.715 Public BA**, but the publicly retrievable winning Kaggle Notebook's saved `submission.parquet` contains **3 preview rows**, not its hidden full-test predictions; post-GLM final 18-way logits are not available for a real same-row champion-vs-new-candidate external comparison.

Avoid repeated 300 GLM historical token tests, small-model substitutes, QLoRA, paid model-provider calls, and public-leaderboard tuning.

## Already verified obstacles

- [2026-11-18 provenance admission #44212521956](https://github.com/kolihhan/name-kaggle/actions/runs/44212521956):
  - `liuyihuileo/ntl-external-training-candidates-18` 3,946-row balanced original-article subset, **18 source-declared model labels**, but only **13** exact model labels had at least one nonempty string-equal upstream `gen_served_model` in that subset (1,381 exactly matching rows overall). This is not a first-party API verification. Qwen official snapshot equality and upstream Open Pool dependency independence are unproven. The 32,542 filtered source candidate article counts are *training-source counts*, not clean validation counts.
  - `keisukefujiwara/nlm-extra-gen` 3,840 responses across **12** model labels / 320 prompt IDs; no served-model response field; cannot be a complete 18-way held-out panel.
- [Full18 frozen historical 0.691 model on Oleg216 #43804320931](https://github.com/kolihhan/name-kaggle/actions/runs/43804320931) yielded **0.412037** on 12 examples per class, all one `sensitive_topics` source domain. Already spent; cannot use for new candidate selection or stand in for present 0.715.
- [0.715 public source preview availability #45139584293](https://github.com/kolihhan/name-kaggle/actions/runs/45139584293): no full 4k hidden prediction vectors; 3-row preview is not a same-row validation cohort.

## Live catalog intake contract

- Run `scripts/ntl_full18_source_intake_20261123.py` (GitHub CPU runner, **Kaggle public metadata only**).
- Three search terms, first two 100-item pages each, dedupe by exact `owner/slug`.
- Exclude only from *new-source follow-up* the previously audited 12 dataset refs; don't mislabel other datasets as unpublished or independent.
- Inspect metadata names for up to **12** distinct not-previously-audited sources updated since 2026-11-18 (or with unknown parseable update date). Explicitly mark unknown dates rather than treating them as new.
- No raw text/response download, GPT API, GPU, model inference, training, calibration or competition submission.
- Dataset title, 18-label claim, file names, generated prompt guesses and site-claimed scores **cannot** pass the full evidence gate.

## Subsequent genuine full18 admission requirements (ALL, not tuned after outcomes)

1. Legal access/license and source ID/date/revision documented.
2. Real responses with all **18 exact** competition label names and actually served-model identity evidence (requested proxy route alone insufficient).
3. Independent original prompts and record/response IDs with request/served/protocol metadata; source/version confounding marked and ideally controlled.
4. Source, prompt, duplicate, near-duplicate and upstream-training-pool intersection audited against **current 0.715 frozen prefit input assets**, not just official train.
5. New held-out selection is frozen **before** looking at model outcomes; no cohort reused from previous score-search; no Public leaderboard adjustments.
6. Reproducible same-row **frozen exact 0.715** and one preregistered differentiated candidate produce all 18 labels, with matched-runtime and resource budget; each result is a BA, class/family recall, source diagnostics, paired rescues/regressions and uncertainty by truly independent prompt groups.

**If any gate fails:** label `NOT_ADMITTED`, preserve Champion; do not manufacture a 0.715-vs-new pair from unrelated outputs or score a source-trained incumbent as unseen. Source is not permanently banned for training research; this decision concerns **independent validation** only.

## Competing mechanisms for the later candidate (not yet approved or scored)

- **Source/protocol confounding:** current highscore discriminators learn producer wrapper, generation channel and formatting correlations; predictions collapse on new domains even with strong embeddings. Falsifier: comparable attribution under disjoint channels / controlled prompt-target distributions.
- **Intrinsic sibling/source identity ambiguity:** even controlling for wrappers, exact model versions share styles; complementary source-conditioned likelihood/decision provenance should improve within-family corrections *without* reopening family false positives. Falsifier: no same-row paired rescues once ordinary strong evidence, protocol, runtime and class balance are held fixed.

The catalog script does not resolve these mechanisms and cannot establish research novelty by itself. If no genuinely eligible source exists, close with an explicit constraint rather than invent a 18-way accuracy gain.
