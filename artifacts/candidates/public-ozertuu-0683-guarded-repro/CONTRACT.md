# Ozertuu 0.683-title public Notebook — frozen guarded reproduction contract

Date: 2026-11-11 MYT. **PRE-FLIGHT ONLY, not yet a scored submission.**

## Scientific scope / baseline

- Existing verified own Kaggle Public BA: **0.668** (58022340 / 58045266).
- Public source: `ozertuu/0-683-0-682-qwen3-8-27b-gsq-15-mix-0-52`. The `0.683` number is **author's Notebook title claim**, not our verified Kaggle score. No improvement may be claimed until our official ledger returns Public Score.
- The scientific act is **external public reproduction**, not original model innovation and not a distinct new data/architecture hypothesis. Do not tune on Public LB.
- Exact public Notebook byte SHA-256: `14d48eb60e4201cd749e0db7f6b9fae89796d4cbb10a4f2f7006300f267535c6`, established by read-only action [42871709117](https://github.com/kolihhan/name-kaggle/actions/runs/42871709117).
- Upstream Notebook status COMPLETE; internet disabled; T4 x2; `submission.parquet` exists in upstream preview. This **does not prove full hidden re-run meets 9h**.
- Exactly 20 data, 8 Notebook and 6 model sources in upstream pulled metadata. Original input slugs retained; no new data/weights committed to GitHub.

## Hard authorization/resource/safety boundary

User has previously prohibited additional training just to submit. Inference-only reuse of already-fitted public assets is the ceiling of this candidate.

The generated own Kaggle Notebook MUST be private, with original credits plus explicit attribution to [Ozertuu's public source](https://www.kaggle.com/code/ozertuu/0-683-0-682-qwen3-8-27b-gsq-15-mix-0-52), as well as original notices kept intact.

Fail closed if ANY of the following changes:

1. source SHA and public metadata, external attachment counts or original code path;
2. supervisory `gauss.fit` branch activates during an exception/OOM fallback;
3. `AtlasModel(..., training=True)` activates; or
4. `core_worker --train-final` activates.

Allow only original unlabeled prompt grouping/TF-IDF fit-transform during inference. Do **not** train LoRA, fit a new supervised Gaussian, update model weights, use Public-LB tuning, download official competition files onto GitHub runners or commit training text/secrets.

`scripts/ntl_public_ozertuu_0683_one_shot.py --source-preflight` builds a hash-pinned, guarded, private Notebook only on ephemeral runner disk; it never pushes or submits. The promotion entry point without flags is **not authorized until remaining license and execution checks pass**. Preserve source-preflight provenance in GitHub Actions logs.

## External rights / contest rules

Kaggle Code Competition allows openly and freely available pretrained assets, offline T4 inference up to 9 hours. However public source `kernel-metadata.json` has **no explicit license field**, and accessible page HTML exposed no license declaration. Do not claim Apache-2 merely because some other notebooks carry that license. Kaggle's [official Notebook guide](https://www.kaggle.com/docs/notebooks) expressly permits **forking any public Notebook to experiment with edits**. This project therefore allows **one private, attributed platform-native fork** for a no-training competition reproduction, without public redistribution or asserting that its SPDX license is Apache-2. The upstream-specific license identifier is still unknown and must not be invented. If a model/dataset reports restricted access or a separate license acceptance requirement, abort rather than bypass it. Do not make public derivative distributions.

## Reproduction gate

1. Confirm hash, metadata, exactly one notebook, source guard insertion counts (one Gaussian, one Atlas, one core-worker).
2. Confirm named credits and no supervised fit active; compile/test generated source scripts and guard behavior where safe.
3. Rely only on Kaggle's documented ability to privately fork a public Notebook; retain all original notices and source credit. A specific open-source license for this source was not exposed by the API and must not be invented. Stop on any dependency-license or permission error.
4. Push **one** own private frozen Notebook under the existing no-training authorization, then inspect its preview status and `submission.parquet` without bypassing no-training guards.
5. If preview passes, submit this **single pre-frozen** private Notebook version once to obtain an authenticated hidden rerun. This is the required whole-size 4,000-row / 9h test, not an independently proven runtime feasibility claim. Report accepted submission ref and eventual official score. No duplicate submission, repeated tuning, retraining or escape hatch.

## Public GitHub safety

[PUBLIC_VISIBILITY.md](../../../PUBLIC_VISIBILITY.md) permits narrowly scoped, no-data GitHub Actions while public. Historical official data exposure is unresolved. Never put raw official data, secrets, external raw source or licensed weights into new commits or public artifacts. This preflight uses ephemeral `/tmp` only.

## Research status

**No scored improvement, no original breakthrough.** This is preparation of a guarded external baseline to determine whether a published 0.683 claim can transfer to the user's account.
