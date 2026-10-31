# Phase 5 — Honest multiview style baseline result

Date: 2026-10-31

## Result

The fixed official-only multiview style model materially improves the frozen CPU anchor.

| Candidate | OOF balanced accuracy | Δ vs Phase 1 |
|---|---:|---:|
| Phase 1 char sanity | 0.31929 | ~0 |
| Phase 5 multiview style SVM | **0.36004** | **+0.04074** |

Phase 1 reproduction error was only `7.46e-06`, so the improvement is measured on a consistent validation setup.

Multiview fold scores:

- fold 0: 0.36852
- fold 1: 0.33148
- fold 2: 0.36481
- fold 3: 0.38148
- fold 4: 0.35390

## Class-level effect

Recall improved for **13 / 18** labels relative to the exact Phase 1 sanity baseline.

Largest gains:

- `gpt-5.5`: +0.1533
- `qwen3.8-max`: +0.1400
- `claude-opus-4-6`: +0.1133
- `gpt-6-astra`: +0.1133
- `glm-5.3-flash`: +0.0733
- `gemini-3.8-flash`: +0.0667

Largest regressions:

- `claude-opus-4-8`: -0.0533
- `claude-opus-5`: -0.0400
- `kimi-k3`: -0.0200

## Decision

**Promote Phase 5 as the new CPU anchor.**

The result closes most of the obvious gap between the original char-only baseline and stronger public prompt-grouped style baselines without changing the permanent folds, adding external data, or tuning parameters against this OOF result.

The public ~0.374–0.376 grouped-CV results are useful context but are not a target to tune against because their exact fold mapping may differ from ours.

## Next experiment

Re-run the already-audited 2,503-row GPT/Gemini external-data ablation using this stronger multiview representation:

- A: official-only Phase 5 anchor;
- B: official training folds + all 2,503 external rows;
- C: official training folds + the class-balanced external slice;
- validation remains official rows only on the permanent folds.

This answers whether the Phase 2 sparse-data uplift survives after the CPU representation becomes substantially stronger.

The GitHub Actions artifact for this run has digest:
`sha256:43b2c6d35743adbc82b8071697349329d13a3ee1b0ae4880f84a59c72d9abbc0`.
It retains the full multiview OOF CSV.
