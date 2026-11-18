# NTL · Phase A 盲评｜2026-11-18

[事实] 独立评阅以代码快照 `d9d144c`、赛事资料、训练原表和下列参赛者原文为依据。原表 2,699 段、150 个 prompt；149 组各有 18 个不同模型，余下一组有 17 个。隐藏集约 4,000 段，公榜约占 40%，测试只给 id 和文本，主办方明确两榜类份额不同。[数据](https://www.kaggle.com/competitions/name-that-llm/data) · [评分](https://www.kaggle.com/competitions/name-that-llm/overview/abstract) · [公榜](https://www.kaggle.com/competitions/name-that-llm/leaderboard)

[判断] BA 是 18 项召回率的平均。修对类 c 的一条加 1/(18 n_c)。公榜约 1,600 行，若各类近均衡、召回均为 .691、行独立，则单模型 SE≈.01155，2SE≈.0231，+.029 约需 47 个等权净修正。一般近似方差为 Σ[r_c(1−r_c)/n_c]/18²；同题相关时须按 prompt 成组。同一公榜两个模型的差还依赖哪些样本一对一错。[未核实] 隐藏精确 N、各类 n_c、配对 SE，以及领先者的数据和配方。

| 竞赛与获奖团队原文 | [事实] 可迁移的部件 |
|---|---|
| [SemEval 2024 Task 8B · AISPACE](https://aclanthology.org/2024.semeval-1.212/) | 多生成器归属，类加权损失与多模型软投票 |
| [IberLEF 2023 AuTexTification · Drocks](https://ceur-ws.org/Vol-3496/autextification-paper14.pdf) | 英西生成器归属冠军，多底座概率作为二层输入 |
| [Kaggle Detect AI Generated Text · 冠军](https://github.com/rbiswasfc/llm-detect-ai) | 二分类，监督对比表征、近邻检索与多模型融合 |
| [COLING 2025 GenAI Detection · Advacheck](https://aclanthology.org/2025.genaidetect-1.26/) | 英文检测，训练时加领域辅助任务，推理用二分类头 |
| [PAN 2018 跨域归属 · EACH-USP](https://downloads.webis.de/pan/publications/papers/custodio_2018.pdf) | 融合字符、变形字符、词三种独立表示 |
| [PAN 2015 作者验证 · Bagnall](https://downloads.webis.de/pan/publications/papers/bagnall_2015.pdf) | 共享字符模型与作者专属预测头，多次运行合并，四个语言组中两个第一 |

[事实] 每赛只计一次。合并独立系统或运行 5/6，微调预训练 Transformer 4/6；最接近的两个多生成器赛都融合概率 2/2。作者条件字符预测与监督对比近邻各 1/6。[判断] 频数表示做法在所选文献出现，不能据此推算 NTL 增益。

| [判断] 单项改动，主观 P(Public≥.720) | 证据 → 最便宜的反证与预算估计 |
|---|---|
| 同题双向对比损失 **15%** | 149 个完整组让每类在同题的其他作者中找自己的回答。保留行级交叉熵，加入列级交叉熵，缺项屏蔽；推理逐行，不需要隐藏 prompt。相同特征/底座、行数和优化步数，比较 5 折新 prompt 的 BA、类内救回/改错；先跑廉价 CPU 线性头，必要时 T4 单折约 1–3 小时。 |
| 作者条件的字符/标点序列似然 **5%** | PAN 的条件预测提供不同读法。先用折内平滑计数，保持词表、长度归一化、类别训练样本一致，与忽略顺序的一元模型配对；150 prompt/2,699 行，估计 5–30 CPU 分钟。 |
| 跨表示的二层读出 **8%** | 两个最近的归属赛均有概率融合。先查独立成员能修对多少基础模型错误；即使全选对也补不了 .029 就不以此冲目标。有不同且合格的折外分数后，按 prompt 嵌套拟合二层；约 10–30 CPU 分钟，不含重建成员。 |

[判断] BA 在校准后可用 p(c|x)/训练先验作决策；官方训练各类为 149–150，单靠这项修正可用空间取决于近乎并列的行。阈值在内层拟合、外层锁定；只在单项配对通过时考虑提交。当前强基线的各部件缺少可归因的同包消融；不能由单个 Public 值拆出贡献。验证 CV↔LB 要为多个预注册、可复现的独立变体配上“同一实现的 OOF 与 Public”，报告秩相关与区间；单个配对点无法标定。Lightning 官方训练预测与 Oleg216 均未用于判断。
