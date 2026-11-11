# Name That LLM — authenticated public leaderboard top-20 snapshot

**As observed:** 2026-11-11 16:08:44 UTC / 2026-11-12 00:08 MYT.
**Source:** Kaggle authenticated `kaggle competitions leaderboard name-that-llm --show -v -q`, read-only; [Actions #43055333132](https://github.com/kolihhan/name-kaggle/actions/runs/43055333132), job `130643141833`, SUCCESS.
**Scope:** Public leaderboard ranks at this point in time only; approximately 40% of hidden test. Final Private standings may differ. The leaderboard lists teams/scores, **not methods or reproducible code**.

| Rank | Team | Public BA |
|---:|---|---:|
| 1 | James Liebel | 0.765 |
| 2 | You’re Absolutely Right! | 0.756 |
| 3 | Le Viet | 0.752 |
| 4 | Meni David Gottesman | 0.752 |
| 5 | kulshekhar | 0.748 |
| 6 | Team Aillis | 0.747 |
| 7 | Hod Carriers | 0.747 |
| 8 | Lian Matsuo | 0.747 |
| 9 | ymg_aq | 0.737 |
| 10 | Søren Ravn Andersen | 0.737 |
| 11 | 吃白饭的大肥鱼 | 0.737 |
| 12 | Harveen Singh Chadha | 0.736 |
| 13 | Liaaar | 0.736 |
| 14 | Nishant Dahal | 0.735 |
| 15 | LLM Hunter | 0.733 |
| 16 | DiamondH | 0.733 |
| 17 | NguyenThanhNhan | 0.732 |
| 18 | t_shiga | 0.730 |
| 19 | Keanan | 0.729 |
| 20 | Smart Manoj | 0.727 |

## Competitive gap

- #1: James Liebel, **0.765** (team ID 17148682); last winning submission on board dated 2026-11-11 01:01:28 UTC.
- #10: **0.737**; #20: **0.727**.
- Our best verified **0.679** (ref 58210835, original public Ozertuu reproduction), **0.086** absolute BA below current #1, **0.048** below #20, at snapshot time.
- MGT-v1 ref 58233227 **PENDING**, no numeric Public BA as of the same authenticated run. Lightning ref 58167751 also **PENDING**.

## Evidence limitations / research guard

- No verified public Notebook, code hash, or methodology for #1 was established by this leaderboard response. Do not claim to possess or replicate James Liebel's method.
- Do not optimize model weights or parameter thresholds against these Public LB scores. Compare any substantial candidate against the frozen 0.679 control on a genuinely independent, source/protocol-qualified, same-row 18-class benchmark before promotion. The [independent-data issue](https://github.com/kolihhan/name-kaggle/issues/1) remains open.
- Snapshot excludes all official passage data; it contains public team labels, dates, ranks, scores only.
