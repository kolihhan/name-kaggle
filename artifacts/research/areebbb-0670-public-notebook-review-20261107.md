# Areebbb public 0.670 notebook — exact-source research review (2026-11-07)

**Status:** READ-ONLY SOURCE REVIEW COMPLETE; official score and independent transfer **not verified**. No model run, GPU training, Kaggle submission, or leaderboard tuning during this review.

**Question:** Does [Areebbb's `0.670 | THIS ACTUALLY WORKS`](https://www.kaggle.com/code/areebbb/0-670-this-actually-works) provide a genuinely different attribution mechanism, a reliable reproducible public baseline, or a leaderboard-only blend?

## Primary evidence

- [Source pull and cell-level manifest](https://github.com/kolihhan/name-kaggle/actions/runs/42359768508) — public Kaggle notebook `areebbb/0-670-this-actually-works`, Kaggle numeric notebook ID `143033312`, SHA-256 `1f8b4818891bcd90ca8c53e98f20fe2c3e6dce7b6f75e387916ae780c1cf054e`, 9 executable + 11 Markdown cells. Successful authenticated **source retrieval only**. Metadata: 20 external dataset refs, 8 upstream notebook refs, 8 model refs, GPU enabled, internet disabled.
- [Read-only public predecessor diff](https://github.com/kolihhan/name-kaggle/actions/runs/42359952735) — pulled `areebbb/ntl-we-back-0-666`, SHA-256 `bd13ddaa14d84ddd0e8afc4eb01d60d69a097faa0fc66e0f6f954d88b07e166e`. The two have 9 executable cells each; exact unified code diff has 33 diff lines, 25 changed/header lines. Neither notebook was executed in this review.
- Author's Notebook text **claims Public LB 0.670**. This review retrieved published source/author listing, but the public Kaggle listing API response did **not** expose an authoritative best-score field. Treat 0.670 as **author reported, not independently authenticated by our account**. Do not equate author's score with our own previous officially verified 0.668.

## Actual pipeline

1. Load existing publicly attached LoRA/encoder weights and corpus assets. Main ensemble includes Qwen3-14B, Qwen3-8B, Llama3.1-8B, Qwen2.5-14B, encoder head, `hf_cls` DeBERTa, and additional complementary models.
2. Text-feature readouts include fitted Open-Pool SVC, pre-fit SVC, TF-IDF, distort/style components, shared-prompt Gaussian, sibling residual, pairwise verifier, and Atlas with Sinkhorn-style balancing.
3. Original public V5 SVC absent: startup code checks `/kaggle/input/**/ntl-svc-prefit-v5/metadata.json` and otherwise uses public `ntl-svc-prefit-pool-v4n`. The paper's claimed 'private tuned V5 weights' are therefore **applied to a fallback public V4n SVC**, not the original private fitted V5. The command line uses `--ours_svc_w 6.0` and `--svc_w 4.0`, `--top34_w 0.15`, `--atlas_w 0.20` / `--atlas_margin 0.32`, `--sinkhorn 0.30`. This is a legitimate public variant but not a byte-identical reproduction of the private V5 system.
4. The extra public DeBERTa classifier is from `nuridudak/ntl-two-stage-encoders/desklib`, inserted with `w=2.2`, max length 384. It uses an 18-label sequence classification head and maps the checkpoint's `id2label` to the ensemble label order. It is an already trained model; the pipeline still performs some lightweight fitting/transformations at scoring time.
5. On the 3-row public fixture, `FAST_COMMIT` skips the heavy pipeline and emits a placeholder `gpt-5.5` Parquet. The hidden larger rerun uses the full inference; a green 3-row Notebook commit would not establish full runtime success.

## Direct Areebbb 0.670 vs Areebbb earlier 0.666 source comparison

Only four small source regions changed, chiefly:

- The earlier `ntl-we-back-0-666` requires a separately attached **ModernBERT** head `ours_mbert` and blends that member at weight 1.4 with public DeBERTa at weight 0.8.
- The analyzed `0.670` removes ModernBERT entirely and raises the existing public DeBERTa weight to **2.2**.
- The new loader no longer passes the `attn_implementation='sdpa'` keyword used by ModernBERT; the in-sample 40-row classifier sanity check becomes conditional on `len(test) < 50`.

Thus the delta between these **two** versions is better described as **replacing ModernBERT+DeBERTa split weight with DeBERTa-only**, NOT introducing DeBERTa for the first time, NOT a new architecture, and NOT a proven causal +0.004 improvement. The 0.670/0.666 numbers are in the authors' notebook titles; this review did not independently retrieve official submission scores or any heldout per-row predictions.

## Claims to keep separate

**Verified source facts**
- Public notebook source and its public predecessor are readable by Kaggle CLI.
- Readout, member weights and fallback paths above are in the exact source.
- The author credits upstream contributors; most learning/backbones and specialist procedures are inherited.
- The DeBERTa 40-row check the author cites (28/40 correct) is on official training passages the classifier saw during its training. That is sanity evidence **only**, not honest validation.

**Author claims, not independently verified**
- Public score of `0.670`, repeatability of score from every public fork, and claims that earlier upstream notebooks intentionally embedded `Decoy Fallback` / `Decoy Camouflage` weights.
- The original upstream Ozertuu snapshots are described as subsequently private. This review did not fetch exact original versions to verify quoted comments, intent, training recipe, or score.

**Unknown**
- Honest prompt/source-disjoint 18-label balanced accuracy and per-class recall, whether DeBERTa contributes novel conditional correct predictions after V47, source/protocol/version robustness, actual hidden test runtime on a fresh fork, whether attached public weights trained with any unauthorized overlap.
- The risk of Public LB selection bias given repeated public variants. Public leaderboard alone is not promotion evidence under `EVALUATION_POLICY.md`.

## Research verdict and next falsification

**Engineering/competition:** Useful public reproducibility reference; worth inspecting as a competitor baseline. The public-source manifest alone is not proof that the full hidden run will fit within Kaggle's nine-hour limit.

**Scientific:** No demonstrated new model-attribution capability. The likely practical lever is relative utility of pretrained/fine-tuned **DeBERTa versus an added ModernBERT** within a large ensemble. That is a hypothesis, not a causal effect estimate, and the author's higher title number cannot independently establish gain.

**Smallest meaningful next test:** If a genuinely independent, exact-model/version and prompt/source-disjoint all-18-label cohort can be verified against the attached training sources, obtain row-aligned predictions/logits for the frozen **DeBERTa-only versus ModernBERT+DeBERTa** variants. Precommit primary BA, class-level recall, provider-vs-sibling confusion and rescue/regression counts. Compare without Public-LB selection. If this cohort cannot be verified or we cannot access valid component outputs, mark the causal/transfer question **not identifiable** rather than running a sequence of leaderboard weight variations. No new training or submission authorized by this review.

**Source-of-truth:** `RESEARCH_DOCTRINE.md`, `EVALUATION_POLICY.md`, and `NOW.md` remain controlling; this review is not a new promotion contract.
