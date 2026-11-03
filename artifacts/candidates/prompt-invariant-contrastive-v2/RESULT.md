# prompt-invariant-contrastive-v2 — Paired Transfer Result

Date: 2026-11-03 MYT

## Verdict

**REJECT — frozen paired source-transfer gate failed.**

The treatment has a small positive point estimate, and all three frozen source-level deltas are non-negative, but the predeclared prompt-group bootstrap confidence interval crosses zero. Under the frozen contract this candidate stops here: no official OOF, no retuning, and no Kaggle submission.

## Frozen experiment

- candidate: `prompt-invariant-contrastive-v2`
- control: v1 Stage-1 A1 objective (`plain CE + centered CE`)
- treatment: control + cross-prompt prototype CE
- contrastive weight: `1.0`
- temperature: `0.2`
- prompt-bucket seed: `20261101`
- base: `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- pool SHA-256: `37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec`
- training rows: `24,099`
- heldout sources: `lucasboesen`, `jadkutbi`, `maniceet`
- heldout rows: `10,745`
- eligible scored rows: `10,723`
- eligible labels: `11`
- prompt-id overlap: `0`
- prompt-key overlap: `0`
- official competition rows scored: `0`
- Public-LB tuning: `false`

## Primary result

| Metric | v1 | v2 | Delta |
|---|---:|---:|---:|
| Pooled balanced accuracy | `0.4102773263` | `0.4129626992` | `+0.0026853729` |

Frozen 5,000-resample prompt-group bootstrap, seed `20261102`:

- 95% CI: `[-0.0006323912, +0.0069285212]`
- required lower bound: `> 0`
- observed lower bound: `-0.0006323912`

**Failure reason:** `prompt_bootstrap_lower_bound_not_positive`.

## Source-level deltas

| Source | v1 BA | v2 BA | Delta |
|---|---:|---:|---:|
| `jadkutbi` | `0.5034189540` | `0.5136205872` | `+0.0102016332` |
| `lucasboesen` | `0.5299789830` | `0.5306388704` | `+0.0006598874` |
| `maniceet` | `0.3632653889` | `0.3636437620` | `+0.0003783732` |

- mean source delta: `+0.0037466313`
- negative-source count: `0 / 3`
- safety-veto labels: none

These positives are supporting evidence that the mechanism is not obviously destructive, but they do not override the predeclared bootstrap gate.

## Training diagnostics

Both conditions used:

- `8,483` triplet microbatches
- `1,414` optimizer steps
- `0` skipped GradScaler steps
- exact one Stage-1 epoch

Treatment contrastive diagnostics:

- eligible anchors: `24,058 / 24,099`
- coverage: `0.9982986846`
- mean contrastive loss per triplet: `2.8699201359`
- extra backbone forwards: `0`
- prototype updates only after successful optimizer step: `true`

## Scientific execution and evidence

The scientific Kaggle execution completed under:

- kernel: `kolihhan/ntl-v2-transfer-a-b`

The original orchestration workflow failed because it polled a stale/wrong kernel ref after the scientific kernel had already completed. The run was **not retrained**. A read-only recovery attached to the existing completed kernel and preserved its outputs.

Recovered evidence:

- GitHub Actions run: `41568080226`
- artifact: `v2-paired-transfer-existing-evidence`
- artifact ID: `13199153657`
- artifact digest: `sha256:54f31cc985bfddf3642eb57c5920cb45bc29a9845752ab2936ab3e94ea14db9b`
- files: `paired_transfer_result.json`, `paired_transfer_predictions.csv`

## Decision

Per the frozen contract:

- reject `prompt-invariant-contrastive-v2`;
- do **not** tune contrastive weight, temperature, prototype scheme, source set, label subset, or threshold using this result;
- do **not** run official OOF to rescue it;
- do **not** submit it to Kaggle.

The small positive point estimate is useful mechanism evidence only. The next candidate must change the mechanism materially. The next research direction should test token-trajectory evidence rather than retuning prompt-invariant contrastive learning.
