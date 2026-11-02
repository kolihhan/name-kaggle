# NOW

Updated: 2026-11-02

## Competition state

- **Best submitted Kaggle system remains Phase 7 gated specialist:** Public LB `0.369`, submission `57838845`.
- **Phase 15 is rejected:** local OOF `0.4053218990802883`, Public LB `0.357`, submission `57861337`.
- **Best verified local candidate is now `two-stage-lora24-v1`: Gate B PASS.**
- Gate-B primary plain-head OOF balanced accuracy: **`0.5467611235396471`**.
- Gate-B centered-head diagnostic OOF: `0.5478498632861049`; diagnostic only under v1 and not eligible for post-hoc blending.
- Phase-7 comparison OOF: `0.3700521998508576`; Gate-B delta: **`+0.17670892368878954`**.
- Gate B wins **5 / 5 folds** versus Phase 7.
- Frozen 5,000-resample prompt bootstrap for GateB-minus-Phase7: 95% CI **`[+0.15743723589361175, +0.19594425801640553]`**.
- Class-recall safety veto: **not triggered**.
- Exact Gate-B result: `artifacts/candidates/two-stage-lora24-v1/GATE_B_RESULT.md`.

## `two-stage-lora24-v1` lineage

- Gate A PASS: A0 held-out public-source BA `0.3942430082` → A1 LoRA BA `0.4657129266`, delta `+0.0714699184`; all three held-out-source deltas positive.
- Model: `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, first 24 layers, LoRA r16/alpha32/dropout0.05, pooled layers `[2,4,6,8,12,16,20,24]`.
- Exact Stage-1 A1 checkpoint: artifact `13094069155`, digest `sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66`.
- Locked checkpoint Kaggle dataset: `kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint`, version `1`, manifest `c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0`.
- Gate-B scientific run already existed and completed successfully: run `41398543856`.
- Gate-B evidence artifact: `13125872633`, digest `sha256:fa675fcf8ac3860984b449748cf067c3da29389a921417e5e1047bb22a7be8ce`.
- Gate-B OOF: exactly `2699` official rows / `150` prompts / `18` labels; zero competition-test rows; zero public Stage-1 rows mixed into Stage 2; no Public-LB tuning; each fold starts fresh from the exact Stage-1 checkpoint.
- The one-shot Gate-B result boundary is now **consumed**. Do not rerun Gate B because of score.

## Evidence interpretation

- Gate B is **strong-but-adaptive/supporting evidence**, not independent hidden-distribution proof: the permanent official rows have been consumed repeatedly across the project.
- Gate A gives separate source-level transfer evidence for the LoRA mechanism, but neither Gate A nor Gate B authorizes a competition submission automatically.
- `two-stage-lora24-v1` is now eligible for a separate submission promotion/preflight.
- A v1 final fit must preserve the frozen identity: exact Stage-1 checkpoint → two Stage-2 epochs on all 2,699 official rows → **plain head only** → offline competition-test inference. No centered blend, sparse router, new backbone, new data, Sinkhorn/Hungarian, or result-driven calibration may be added inside v1.

## Post-Gate-B optimization evidence

Read-only analysis of the existing honest Phase-14 OOF against Gate-B OOF found substantial complementarity:

- Phase-14 four-view SVC OOF BA: `0.4178100919711658`.
- Gate-B plain vs Phase-14 hard-prediction disagreement: **`0.5602074842534271`**.
- Gate-B-only correct rows: `641`.
- Phase14-only correct rows: **`293`**.
- Either-component oracle BA: **`0.6552970420084514`**.
- Gate-B plain vs centered disagreement: `0.4201556131900704`; plain-only correct `350`, centered-only correct `353`; two-head oracle BA `0.6775466070096942`.
- A diagnostic three-way hard majority (`plain`, `centered`, Phase14 SVC; all-distinct tie -> plain) scores **`0.5674869500372856`**, +`0.020725826497638433` over plain, improves all 5 folds, and has same-OOF prompt-bootstrap 95% CI `[+0.01111111, +0.03037037]`.
- **This majority result is adaptive diagnostic evidence only.** The rule was examined after v1 OOF was visible, so it is not a fresh promotion result and must not be silently folded into v1.

Public-method research already in the repo independently supports the same broad complementarity pattern: historical Qwen3-8B and Qwen3-14B two-stage LoRAs were each reported around `0.569` grouped CV while disagreeing on about 22% of passages; the public 8B+14B+n-gram+centered/balancing ensemble was reported around `0.593`. See `artifacts/research/public-two-stage-lora-audit-2026-11-01.md`.

## Current execution policy

- **Do not use GitHub Actions for new work** until the user changes this instruction; quota is exhausted. Reading already-existing run/artifact evidence is allowed.
- Repository visibility is currently **private**. `PUBLIC_VISIBILITY.md` remains a separate historical/public-safety cleanup record and must not be treated as evidence that past exposure was erased.
- Do not create another Kaggle submission until the submission promotion/preflight for `two-stage-lora24-v1` is frozen and verified.

## Next single action

Build and verify a **direct, non-GitHub-Actions submission preflight/final-fit path for `two-stage-lora24-v1`**, preserving the frozen Gate-B Stage-2 configuration and plain-head-only v1 inference. Do not submit until that preflight is complete and submission is explicitly authorized.
