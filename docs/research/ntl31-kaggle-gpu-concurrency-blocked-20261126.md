# Name That LLM Gemma4 31B + Ensemble — actual 2026-11-26 push state

**Outcome: PREPARED / BLOCKED_BY_KAGGLE_GPU_CONCURRENCY, NO official Kaggle submission.**

The user explicitly requested no newly trained model, a real Gemma4 31B + existing strong Ensemble official code score, aiming for Public BA >=0.760. We built and attempted the one-shot Kaggle path rather than running a small-model training experiment.

- Implementation: `scripts/ntl_gemma4_31b_ensemble_stage_20261126.py` and `scripts/ntl_gemma4_31b_ensemble_submit_once_20261126.py`.
- One-shot code-submission workflow: [GitHub Actions #45766283484](https://github.com/kolihhan/name-kaggle/actions/runs/45766283484).
- Frozen source: Lightning v52 independently scored 0.724, real pinned Kaggle source SHA `b4c9715e32d27463a399f8715684c8403f057c83f1f77e0039ff0da2427df343`; candidate built SHA `17769a31e1e8474e9729128fd62ed1e01937fa14a412b0033003192d225f9bc4` (BEFORE title-only metadata alignment update).
- Model: Gemma 4 31B instruct GGUF Q4_K_M, **18,323,726,560 bytes**, Kaggle dataset `batuhankska/gemma-4-31b-it-q4-k-m`; T4 SM75 llama-server `nixty2/llama-cpp-turing-sm75`. Verified manifest in [read-only Action #45765548592](https://github.com/kolihhan/name-kaggle/actions/runs/45765548592) / [#45765840657](https://github.com/kolihhan/name-kaggle/actions/runs/45765840657).
- CPU Python preflight and static source pin **PASSED**. Exact v52 source verified. Defined intervention disables 27B GSQ (share 0.035 -> 0.0), lowers stage time allowance 7h -> 6.25h, appends a real 31B in-context top-two scorer on up to 600 uncertain rows. Inherited small inference-time classification heads remain; no additional fitting or weight training was added.
- Authenticated Kaggle: own champion `#58755304` COMPLETE Public BA 0.718; 5 allowed submissions, no active competition scoring job.
- **Kaggle `kernels push` returned exit status 0 but explicitly printed `Kernel push error: Maximum batch GPU session count of 2 reached.`** The fail-closed script correctly rejected this pseudo-success; no actual new Kaggle Notebook version/preview, 31B GPU load, competition submission ID, hidden score or GPU bill was initiated for this candidate.
- User's other two active GPU batch sessions were not modified or cancelled. Owner-facing candidate notebook title was aligned with its intended Kaggle slug after the rejected push; this was metadata cleanup only.
- A dedicated hourly *conditional* automation was created to check for GPU session availability and resume this **single** source-pinned submission attempt when safe. It must stop after the first official submission ID and must not repeatedly push/submit or rerun retired Confidence Fusion, nor touch other running GPU jobs. Until Kaggle has free batch capacity the status remains **BLOCKED**, not scored.
- Never report 0.76 or an official submission ID without an authenticated COMPLETE numeric receipt. **Public Best unchanged 0.718.**

[Prospective candidate contract](ntl31-gemma4-inference-only-one-shot-contract-20261126.md). Scoring/runtimes remain unknown. The requirement that actual 31B model load and at least one real response complete in Kaggle GPU preview is still unverified.
