# NTL 0.718 source contamination and Kimi/GPT frozen encoder probe — 2026-11-26 MYT

## Current decision
**Do not use the 4,302-row prvsiyan 18-label panel as independent validation against the pretrained 0.718 stack.** Source audit shows the incumbent directly mounts prvsiyan and actively reads the upstream `crtproduction/name-that-llm-open-pool` table. Current Kaggle Public Best remains **0.718** (#58755304). Confidence-Fusion official submission workflows stay permanently disabled. **No official Kaggle submissions, model training or GPU inference** in this research.

## Evidence: frozen scorer lineage
[Scored source notebook evidence](ntl-0715-vs-0718-real-lineage-causal-boundary-20261124.md) ties the owner's 0.718 to Evgen v36. Read-only [#45743661629](https://github.com/kolihhan/name-kaggle/actions/runs/45743661629) recovered the same 9-model, 26-dataset, 10-kernel dependency inventory from the frozen 0.718 source lineage (retired 252-run notebook). That inventory **directly includes** `prvsiyan/prvsiyans-extended-name-that-llm-dataset` and `crtproduction/name-that-llm-open-pool`. The code reads `ntl_open_pool.parquet`, filters `flag_train_prompt_regen`, and provides examples to a text-classifier component. Other prefit components have opaque upstream training histories. A mounted source does not itself prove its contents were directly used, nor prove the exact Nov23 scoring-version snapshot.

## ACTUAL text-hash overlap with the currently published upstream pool
[Successful read-only CPU run #45743815844](https://github.com/kolihhan/name-kaggle/actions/runs/45743815844) downloaded real 5,070-row `prvsiyan/passages_clean.csv`, restricted to `split in {new,new2}` (4,302 rows / 18 classes × 239), and the current `crtproduction/ntl_open_pool.parquet` (238,750 rows). Casefolded, whitespace-normalized SHA256 exact-response hashes were compared without publishing any text.

- **Gemini 3.7 Flash: 173 of 239 texts already in the current upstream pool**; all 173 have `source_dataset=prvsiyan/...` and are not marked `train_prompt_regen`.
- **Gemini 3.8: 0/239**, **Kimi K3: 0/239**, **GPT 5.5/Luna/Sol: 0/239** (same check), and 0 in every other new-prompt class.
- Total 173/4,302 current exact matches. This is a **real, asymmetric overlap**, not only a generic risk.
- **Time caveat:** The Open Pool is mutable; currently published 238,750 rows are not demonstrated to be the exact snapshot mounted for the 2026-11-23 official 0.718 score. This result cannot retrospectively claim exactly 173 training leaks in that version. Further, absent exact repeats, semantic duplicates and previously trained checkpoint exposure remain possible.

## Next distinct mechanism: strong pretrained encoder (not a tiny fitted classifier)
One frozen proof-of-signal with MIT-licensed **intfloat/e5-large-v2**, a **335M-parameter**, 1,024-D encoder released before the 2026 dataset. This is a frozen *representation audit*, not a production model or an 18-class candidate.

Protocol, source code: [scripts/ntl_kimi_e5_frozen_transfer_20261126.py](../../scripts/ntl_kimi_e5_frozen_transfer_20261126.py).
Run: [#45744290403](https://github.com/kolihhan/name-kaggle/actions/runs/45744290403).

- Kimi Moonshot route vs GPT Luna OpenAI first-party route only; exclude `train_regen` and opaque Kaggle Proxy.
- **70** grouped official train prompt pairs for nearest-centroid readout, **40** disjoint official prompt pairs for a control, **80** external new-prompt pairs for source transfer, **380** embedded passages total.
- Exactly matching selected official/external text hashes are disallowed. Pin pretrained E5 revision `d94960066016d2064f2a3a2dbc65107a2b48230d`, run CPU only, no gradient updates.
- Compare external binary balanced accuracy, Kimi and GPT separate recalls, paired rank, matched-prompt bootstrap CI, 200 shuffled training-label controls, and single-feature word-length reference.
- **NOT an honest paired BA comparison against the full 0.718 incumbent**, which has pretrained data dependencies and no verified same-row 18-class logits for these texts.
- First execution [#45744110426](https://github.com/kolihhan/name-kaggle/actions/runs/45744110426) verified the real dataset (380 texts, no exact cross-source overlap) but failed on the **checkpoint loader**, because this SHA-pinned model contains `pytorch_model.bin`, not `model.safetensors`. Fixed one file-format argument without changing inputs, split or evaluation; the corrected run is #45744290403.

## Scientific decision gate
- If the frozen 335M embedding representation cannot beat the length-only reference and shuffled control on external Kimi/GPT pairs, reject **generic semantic embeddings as a Kimi fingerprint**; do not spend a larger GPU on that premise.
- If it materially beats both, the next step is *not* a submission. Verify complete training provenance, acquire actual same-row incumbent outputs on lawful independent rows, and measure **18-class rescues/regressions** to see whether the new signal is complementary.
- Gemini 3.7 vs 3.8 stays out of this first frozen representation test due to asymmetric exact-source overlap.
- Never tune thresholds on this spent research panel and present them as independent.
