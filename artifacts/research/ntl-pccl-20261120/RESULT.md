# NTL PC-CL v0 — ACTUAL paired QLoRA training result, 2026-11-20

## Scientific decision: reject the tested v0 recipe; no claim that all Contrastive Learning fails

**Authenticated run:** [GHA #44683814370, completed SUCCESS](https://github.com/kolihhan/name-kaggle/actions/runs/44683814370), aggregate artifact `ntl-pccl-real-17b-paired-aggregate-44683814370` (no raw passages, IDs, per-row outputs, model weights or checkpoint). This is **genuine owner Colab T4 QLoRA optimization**, not static mock testing and not official Kaggle hidden scoring.

Two initial executions [#44682500480](https://github.com/kolihhan/name-kaggle/actions/runs/44682500480) / [#44683032195](https://github.com/kolihhan/name-kaggle/actions/runs/44683032195) stopped before starting any training due the *old pinned model revision lacking parsed license metadata*. The [CPU license+weight provenance audit #44683679240](https://github.com/kolihhan/name-kaggle/actions/runs/44683679240) verified the **old pinned and current Apache-2.0 card revisions contain identical `model.safetensors` SHA256 `6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841`**. Fixed a metadata guard, **not** model weights, data, training budget, objective or source split. Third run completed both arms and closed/released T4.

## Verified same-row results

Source-disjoint external-publisher-only cohort pinned from Kaggle open pool SHA256 `7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492`. Source/time/protocol: [preregistered contract](PREREG.md), [real CPU feasibility #44681721339](https://github.com/kolihhan/name-kaggle/actions/runs/44681721339). **8,347 train texts (8 publishers), 1,800 validation (3 disjoint publishers; 100 examples for each of 18 labels)**. Prompt contents/IDs and normalized text duplicates separated across train/valid. Publisher route attestations are not independent original-API proofs.

| Real metric | C0 ordinary 18-way CE + QLoRA | C1 same CE + paired 2×2 InfoNCE |
|---|---:|---:|
| Pinned backbone | Qwen3-1.7B-Base, r8 4-bit NF4 | identical |
| Optimizer steps | **160** | **160** |
| Total response presentations | 640 | **identical 640** |
| Trainable parameters | 8,753,152 | identical |
| 18-label source-disjoint BA | **0.078889** | **0.071667** |
| Valid correct / 1800 | **142** | **129** |
| Last 40-step mean training loss | 3.688179 | 3.705957 |
| Wall time incl. 1,800-row eval | 298.43 sec | 295.31 sec |

**C1−C0 = −0.007222 BA**, 1,500-draw source-stratified prompt-cluster bootstrap 95% CI **[−0.018839, +0.004112]** (includes 0; *not* uncertainty over unseen sources). Same-row **62 rescued / 75 regressed**, net −13. No official Public BA gained; owner best remains **0.702**.

Crucial per-label BA shifts (all labels have 100 validation examples):
- `qwen3.8-max` **0.21→0.54 (+0.33)**;
- `gpt-5.6-terra` **0.34→0.06 (−0.28)**;
- `glm-5.3` **0.31→0.15 (−0.16)**;
- `gemini-3.7-flash` **0.07→0.01 (−0.06)**.
- Six of the 18 labels have **zero recall under both** C0 and C1; other labels remain near chance. Large compensating class movements mean a claim of global generator-invariant learning would be false.

## Real design weaknesses and why the scope of the conclusion is narrow

1. **C0 is barely above random 18-class BA 1/18≈0.05556**. Neither 160-step training arm is a competitive attribution model. A result on such an underfitted baseline has little power to identify the upper bound of generator-level contrastive representation.
2. Only **13/18 labels** have real 2-prompt×2-model rectangles; five receive ordinary CE only. This is not a fully 18-way contrastive study.
3. Only **12/120 (10%)** contrastive rectangles pair the *same model* across different sources; the rest contrast prompts *within a training source*. So **source-invariance is tested only indirectly via the source-disjoint validation**, never guaranteed by objective design.
4. **Potential bad curriculum:** first 120 batches all rectangles; **last 40 all ordinary label-balanced CE**. The last-40 loss increased, so a distribution-switch/forgetting problem is plausible. Because both arms share the exact schedule, it does not create a confound *between arms*, but it limits how much the test distinguishes the proposed mechanism from insufficient balanced exposure.
5. **Only 640 training text presentations per arm**, even though the train pool contains 8,347 rows. It is a bounded smoke-size mechanism trial, NOT a fit-to-convergence test.
6. Source-disjoint does not mean independent generator route verification. The three validation publishers are uneven in label support; the macro score is exactly balanced across labels overall, but source effects remain a possible factor.
7. Neither model was scored on the 0.702 champion's same-row labels. **No complementarity or Private BA conclusion**.

These limitations were present in preregistration or deterministic batch schedule, not discovered after selecting a better loss weight. **No weight/temperature/seed tuning** is authorized based on these results.

## 2026-11-20 15:53 MYT follow-up — **ACTUAL post-hoc CPU reference, no new QLoRA**

[GitHub Actions #44698095061](https://github.com/kolihhan/name-kaggle/actions/runs/44698095061) **completed SUCCESS** (artifact `ntl-pccl-cpu-same640-vs-full8347-44698095061`). Existing fixed [reference script](../../../scripts/ntl_pccl_source_holdout_svm_sanity_20261120.py) was run once, with no hyperparameter search, external new data, GPU, or Kaggle submission. Original SHA-pinned external pool was rematerialized ephemerally and its `train.parquet`, `valid.parquet` and `batches.json` were **fingerprint-checked** before training. All raw texts and data copies deleted from the public Action work area.

| True evaluated model | Training exposures | Distinct texts if available | Source-disjoint 18-label BA | Correct of 1,800 |
|---|---:|---:|---:|---:|
| Qwen3-1.7B CE QLoRA (previous matched T4) | 640 | ≤640 | 0.078889 | 142 |
| Qwen3-1.7B CE+PC-CL QLoRA (previous matched T4) | same 640 | same | 0.071667 | 129 |
| **TF-IDF char+word LinearSVC on exactly same batch presentations** | **640** | **532** | **0.122778** | **221** |
| **Same frozen TF-IDF+LinearSVC on complete allowed train** | **8,347** | — | **0.216111** | **389** |

This is a reference comparison, **not a causal attribution of the 0.137 BA gap to training steps**: moving from QLoRA to SVM changes representation, training method, and optimization; moving SVM640→SVM8347 changes sample coverage. It cannot isolate the effect of more QLoRA steps or confirm that source shift is the only bottleneck. Nevertheless **the data are not inherently random-label/wholly unlearnable**: even a simple supervised method extracts transferable source-disjoint signal. On the **same very limited 640 samples** the SVM already beats the 160-step QLoRA CE, so QLoRA baseline is **undertrained/under-exposed or otherwise ineffective in this pilot regime**.

Source-specific SVM full train macro Recall among available labels: `alperen5252525/ntl-vertex-gemini-proxy-gpt-generations` **0.318979** (4 classes); `arxitekton/ntl-claude-gen-v1` **0.205209** (16 classes); `kenandemirci/ntl-glm-gemini-generations` **0.178836** (5 classes). These are NOT full 18-label BA per-source and have unequal class supports. Contrast with SVM640 0.136583 / 0.133915 / 0.218254 on the same respective sources. Only three published validation sources exist, so none certifies the actual competition host.

### Updated model-class decision (required research-doctrine pivot)

1. **Keep PC-CL v0 rejected**: actual paired 160-step C1−C0 was **−0.007222**, CI crosses zero. No loss-weight, temperature or seed sweep on this spent heldout.
2. **Retire the incorrect inference that any contrastive method is disproved**: the CE control is ~0.079 BA, only 640 presentations/532 unique texts, with 13 contrastive-eligible labels and 10% cross-source matched pairs. Under these conditions the pilot is a training-regime falsification, not a capacity-limit measurement.
3. **Global bottleneck remains unresolved**: inefficient/imbalanced QLoRA exposure, source/served-model mismatch, and weak 18-way identity signal could coexist. SVM full transfer 0.216111 suggests real but limited source-distinct learnability. The old 0.702 Public champion remains on entirely different hidden data; its complementarity with this experiment is UNKNOWN.
4. **Next hypothesis when a materially new test budget is justified:** one **CE-only, fully class-covered** 1.7B QLoRA reference with fixed longer balanced training and an unseen *new source* evaluation, compared to the **0.216111 SVM full-data reference** for basic training sufficiency; only revisit prompt-controlled contrastive if CE can itself learn competitive signal. This is a *prospective possible new study*, NOT an authorized immediately repeated run or frozen numeric improvement. Do not count 0.216 against 0.702 as paired evidence.
5. Per the user's **two-informative-tests/pivot agreement**, stop the current PC-CL v0 lane after **the actual CE vs CE+PC-CL paired GPU test and this CPU SVM exposure check**; reconsider data/representation rather than issuing another V1/V2 training tweak on the same split.


## Recommendation after the real negative

- Per predeclared rule `C1−C0 ≤0`: **REJECT PC-CL v0** on this cohort. Do not run Qwen3-8B PC-CL, add a shuffled third training arm, or submit to Kaggle on this evidence.
- **Before spending another T4**, run **one frozen CPU supervised reference** on the exact same source-disjoint 8,347/1,800 cohort (e.g. char+word TF-IDF LinearSVC, unaltered training split) to distinguish: (A) this pool/route split has very poor cross-source transfer for all classifiers, versus (B) 160-step 1.7B QLoRA is drastically undertrained/poorly sampled. Label this *post hoc diagnostic* (not fresh validation), and don't tune against it.
- If the CPU anchor transfers well, return to **training exposure / balanced interleaving and hard-negative coverage** as first-order bottlenecks. If even a reasonable fully-fitted sparse supervised anchor is at chance, the source and label provenance mismatch dominates and more LoRA/GPU is unlikely to help.
- Revisit originality after deciding whether the core problem is representation, under-optimization or unavailable label/protocol parity. Two future informative tests maximum per architecture class, not repeated prompt/weight sweeps.
- Current official competitor target remains 0.702 vs 0.761 top10 at last verified Nov20. This is **not a Kaggle Public-score improvement**, and no official submission was made.

## Reproduction / status

- [Source frozen train/valid & rectangles](../../../scripts/ntl_pccl_18source_prepare_20261120.py)
- [Actual GPU QLoRA trainer](../../../scripts/ntl_pccl_17b_real_qlora_colab_20261120.py)
- [Colab wrapper](../../../scripts/ntl_pccl_real_colab_remote_once_20261120.py)
- [SUCCESS official job and aggregate artifact](https://github.com/kolihhan/name-kaggle/actions/runs/44683814370)
