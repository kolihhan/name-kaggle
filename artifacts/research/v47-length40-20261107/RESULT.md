# V47 fixed-window SVC result

Updated 2026-11-08 MYT. Official submission `58078151` is **COMPLETE, Public BA 0.663**, with Private score blank. The completed score was visible in [authenticated history 42385772029](https://github.com/kolihhan/name-kaggle/actions/runs/42385772029) and freshly reverified at 2026-11-07 16:56:59 UTC / 2026-11-08 00:56:59 MYT in [read-only history 42393822063](https://github.com/kolihhan/name-kaggle/actions/runs/42393822063). This supersedes the earlier 07:11 UTC PENDING observation. No re-submission or model change.

## Frozen solution

The selected hypothesis is that equal-length evidence windows can reduce a collection-length shortcut in V47's externally fitted SVC. Score contiguous 40-word windows with the existing SVC, average their raw margins, then apply the original log-softmax and weight 12. Pass passages of at most 40 words through directly. The window starts cover the full passage from its first word to its last word. The other votes and downstream blending follow V47.

The 40-word budget comes from the competition's published lower passage-length bound. It was frozen before candidate evaluation. The earlier external pilot demonstrates a requested-length shortcut in that particular corpus. Its membership in V47's fitting pool remains unverified. [Frozen contract](CONTRACT.md)

## Exact versions

| Item | Value |
| --- | --- |
| Control | Lightning V47, Kaggle script version `365000313` |
| Verified control submission | `58008463`, Public `0.666` |
| Private candidate | `kolihhan/ntl-v47-length40`, version `2` |
| Accepted candidate submission | `58078151`, 2026-11-07 at 03:54:38 UTC |
| Candidate code SHA-256 | `b686a03a1f792ca722cae07a289c44ea2c5314a9eacd627773a5d68c1865d75c` |
| Candidate Notebook SHA-256 | `9942051bd6f4ad778e419f0441e9263b960114d888ac021ddbcb462bfd86f63f` |
| Original hub attachment | dataset v18, numeric ID `21594398` |

The Kaggle save was read back and its executable code hash checked before submission. Fourteen dataset version selectors were recovered against the archived numeric IDs. All eight original notebook weight attachments were independently checked through Kaggle output URLs against the archived kernel version IDs. Five Qwen model attachments retain their original version-1 selectors. Two attached external text corpora remain unpinned and are outside the inference path.

## Verification

| Check | Observed result |
| --- | --- |
| Notebook validation and compilation | All 37 code cells passed |
| Archived source metadata | Same 30 serialized source IDs |
| Targeted regression tests | 3 passed |
| Mount compatibility | Canonical and legacy mounts passed, incomplete cache rejected |
| Complete executable diff | Seven changed cells, patch applied byte-exactly to original cells |
| Real fitted SVC CPU check | Passed on the v18 state fitted on 190,588 pool rows |
| Synthetic repeated 40-word block | Original maximum margin drift `0.27140673584030006`, treatment `0.0` |
| Real fitted short-passage control | All three core sparse votes matched their original scoring formulas |
| Full Kaggle fixture inference | COMPLETE on version 2 with preview shortcut bypassed |
| Fixture repetition control | Original maximum margin drift `0.21825205772124678`, treatment `0.0` |
| Submission output | 3 rows, columns `id` and `model`, ordered IDs checked, 18-label vocabulary checked |
| Official submission count for this variant | One accepted submission, `58078151` |

The full fixture run exercised the original GPU ensemble and final Parquet writer. The repeated-block checks establish the intended invariance with real fitted weights. The completed Public score did not show a gain. [Full Kaggle preflight and submission runner](https://github.com/kolihhan/name-kaggle/actions/runs/42331397544) · [Independent fitted-SVC check](https://github.com/kolihhan/name-kaggle/actions/runs/42336678550) · [Authenticated full-preflight and accepted-submission evidence](https://github.com/kolihhan/name-kaggle/actions/runs/42336749310)

## Executable changes

| Cell | Change |
| --- | --- |
| 2 | Disable the three-row preview shortcut and resolve the original hub mount with both required caches present |
| 4 | Bind existing hub paths to the resolved mount |
| 10 | Add the fixed-window scorer to the SVC4 vote |
| 12 | Fail closed if the inherited Gaussian overflow branch would refit supervised parameters |
| 44 and 45 | Read the original numeric cache parts through the resolved mount |
| 46 | Run the complete inference path, check real-weight invariance and output schema, and expose a failed stage's traceback |

Notebook outputs and execution metadata were cleared for delivery. No new supervised model fitting is part of this candidate. The active algorithm is the single frozen SVC treatment.

The first private version failed before competition submission because the inherited hub-version hint selected v17 and the short mount path missed the numeric cache. Kaggle dataset metadata independently identified v18 as original source ID `21594398`. Version 2 corrects those operational bindings and retains the frozen treatment. [Attachment version evidence](https://github.com/kolihhan/name-kaggle/actions/runs/42330857139) · [Original notebook weight identities](https://github.com/kolihhan/name-kaggle/actions/runs/42332529741)

## Decision

The official score is `0.663`, or `-0.003` against the predeclared V47 control `0.666`. Under the original rule (follow up at >=`0.671`, close at <=`0.661`), this is **INCONCLUSIVE**, not a promotion or a statistically established loss. It did not beat the control. Keep the variant frozen; do not retune windows or weights toward this Public score.

This single diagnostic follows the user's latest instruction to build a solution and submit it for verification. Leakage-safe grouped OOF for the V47 ensemble is unavailable. A validated research promotion requires independent transfer evidence in addition to this frozen official comparison.
