# NTL 31B zero-finetune — v3 real GPU outcome and next discriminating research

**Recorded:** 2026-11-26. **Scientific conclusion:** Gemma 4 31B has **actually loaded and decoded** on Kaggle T4×2, but no official scored submission from v1–v3. This note is evidence-based; it does not project Public BA above 0.718.

## v3 observed (actual execution, not a source-code-only smoke test)

Kaggle notebook `kolihhan/ntl-gemma4-31b-ensemble-no-train-20261126`, private version **3**, verified through the [GH run #45771094308](https://github.com/kolihhan/name-kaggle/actions/runs/45771094308) and [decoded actual Kaggle server logs #45779703827](https://github.com/kolihhan/name-kaggle/actions/runs/45779703827).

- Model: unchanged Gemma 4 31B-it GGUF Q4_K_M, **18,323,726,560 bytes**.
- Runtime: actual **CUDA 13, T4-compatible llama.cpp**, two-card layer split. First successful load `model loaded`, local endpoint `127.0.0.1:19031`.
- Real **three** /v1/chat/completions inferences: first prompt evaluation ~7.5s / 1,337 prompt tokens; next two prompt eval ~2.1s/147 and ~2.3s/194. Generated 28 tokens each at **10.21, 10.65, 10.66 tokens/s**; server output record exists.
- Critical failure: all three responses **failed exact official-label parsing**, so the Notebook terminated with `RuntimeError: no real valid 31B predictions` and no official Kaggle Submission ID. This is **a reasoning-channel/format protocol failure candidate**, not proof of wrong model labels; raw assistant content was not persisted, so specific contents and the precise parsing cause cannot be claimed.
- v4 holds the same weights/prompt task/v52 baseline and original low-margin top-two route; **changes only inference protocol**: server-supported constrained JSON schema with label enum `[incumbent, alternative]`, `enable_thinking=false`, max 96 tokens, strict JSON decode. [v4 real run](https://github.com/kolihhan/name-kaggle/actions/runs/45779905866). No v4 or official score claimed in this note.

## Time-budget implication

Original 31B stage budget **1,700 seconds** after the much heavier v52 v1/stack. Measured warm stage (two responses) **4.6–4.9s each** with 28 generated tokens, so **600 rows would take roughly 2,800–2,950 seconds plus model load** if the same average persists. Since server prefill + decode varies by text and actual v4 constrained decoding may change token counts, this is a **first-order warning, not a benchmark of all hidden rows**. Expect a smaller subset than 600 to be processed before the hard 1,700s stop. The stage's existing early-stop handles this, but the effective change count may be too small to move 18-way BA by +0.036–0.042. Hence never promise 0.76 from 31B merely because it has more parameters.

## Next research design: 31B conditional style likelihood (no fine-tuning)

*Research basis:* [Miralles-González et al. EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.14/) reports that **one-shot style-transfer conditional token log-probabilities** outperform unsupervised direct prompts on their authorship-verification benchmarks. It does *not* report a score on this competition or guarantee any 18-class Public improvement. Source-conditioned [ATLAS (arXiv:2610.19583)](https://arxiv.org/abs/2610.19583) learns source-specific likelihood readouts and already contributes related evidence to the 0.718/0.724 ensemble lineage; importing ATLAS is not novel and may require training.

A genuinely distinct next experiment would freeze **the same 31B weights** and compare, under identical prompts and test examples:
- (A) v4 constrained direct top-two attribution, plus prompt-order randomization as a control.
- (B) inference-only *teacher-forced per-token* text likelihood after conditioning on one true-labeled source style reference per candidate, scored on the *same unknown text tokens*. Use per-token mean and a neutral-prefix negative control.
- (C) swapped/shuffled source reference to falsify label-protocol shortcuts. Score 18-way BA and paired source-wise rescues/regressions; group by prompt to prevent topic leakage.

**Feasibility gate:** Verify true **prompt-token log-probabilities** in the exact pinned runtime; generated label-token logprobs are not the same thing. Do not spend a full Kaggle scoring slot or claim score uplift based on model scale or small binary-only samples. Establish an independent or explicitly DEV-labeled same-row 18-class truth set and a within-Kaggle runtime estimate first.

**Keep user constraints:** no further small-encoder research as the primary lane, no new training for the current submission, never silently re-enable the canceled scheduler, no second score submission for this v4 once a unique receipt exists.


## v4 genuine Kaggle acceptance: verified UPDATE on 2026-11-26 16:08 MYT

- v4 corrected only Gemma4 reasoning/response formatting via **llama.cpp JSON-schema enum** and `chat_template_kwargs.enable_thinking=false`, retaining Gemma 4 31B Q4 weights, frozen Lightning v52, no new model training.
- Authentic Kaggle v4 private preview **COMPLETE**. Actual retrieved `ntl31_proof.json` records `loaded_and_inferred=true`, `inference_calls=3`, `valid_calls=3`, `preview=true`, `n_test=3`, `time_s=130`, `gguf_bytes=18323726560`. The 3-row preview `changed=0` is intentional; it has no labeled-truth evidence and does NOT estimate hidden score.
- Authenticated official single competition code submission **#58935812**, created `2026-11-26 08:06:26 UTC`, description `Gemma4 31B Q4 zero-finetune top2 rival replacing GSQ27B on Lightning V52 once 20261126`.
- Official score state at last audited check: **`SubmissionStatus.PENDING`**, Public BA empty. Do not report a 0.76 breakthrough, successful hidden 4,000-row completion, or an improvement over 0.718 until Kaggle returns a numeric Public BA.
- Evidence: [read-only private 31B proof + unique official receipt #45780567819](https://github.com/kolihhan/name-kaggle/actions/runs/45780567819); [single launch & official request #45779905866](https://github.com/kolihhan/name-kaggle/actions/runs/45779905866).
- The source's `final-logits.npz` is indeed `ids,labels,logits` with a single 18-column matrix. Confirmed against the frozen donor source in [GH Action #45780220459](https://github.com/kolihhan/name-kaggle/actions/runs/45780220459). Hidden runtime path and effective 31B improvement remain unknown pending scoring.
- Do **not** resubmit this candidate or re-enable the user-cancelled scheduler. Next experiment must be new scientifically discriminating evidence, not replaying the same Public-LB hyperparameters.
