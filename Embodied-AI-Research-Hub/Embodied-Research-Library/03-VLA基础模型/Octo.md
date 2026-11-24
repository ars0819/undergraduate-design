---
citekey: Octo
arxiv: '2405.12213'
checked: 2026-10-24
priority: 扩展12
source_level: 摘要与目录导读
tags: [具身智能, 论文学习]
---

# Octo：开源通用机器人策略

## 1. 英文标题与中文译名

Octo: An Open-Source Generalist Robot Policy

Octo：开源通用机器人策略（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Octo Model Team; Ghosh, Dibya; Walke, Homer; Pertsch, Karl; Black, Kevin; Mees, Oier; Dasari, Sudeep; Hejna, Joey; Kreiman, Tobias; Xu, Charles; Luo, Jianlan; Tan, You Liang; Chen, Lawrence Yunliang; Sanketi, Pannag; Vuong, Quan; Xiao, Ted; Sadigh, Dorsa; Finn, Chelsea; Levine, Sergey。首次提交：2024/05/20。RSS 2024；正式会议论文。

[arXiv](https://arxiv.org/abs/2405.12213) · [正文HTML](https://arxiv.org/html/2405.12213) · [公开PDF](https://arxiv.org/pdf/2405.12213) · [正式来源](https://roboticsproceedings.org/rss20/p090.html)

正式DOI：10.15607/RSS.2024.XX.090；arXiv DOI：10.48550/arXiv.2405.12213。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要与目录级。所列章节供后续精读定位，不表示已经精读这些正文段落。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

Octo 把跨机器人预训练看作新任务学习的起点。模型通过可适配的输入 Token 化和输出动作头处理不同传感器与动作空间，并能依据语言或目标图像完成任务。它用大量跨具身轨迹学习共享表示，再用较少新平台数据微调。它虽然也使用 Transformer 和语言条件，但不应简单等同于以大型互联网 VLM 为主干的所有 VLA。对你的学习，Octo 的价值在于展示输入输出接口如何设计，以及为什么数据格式相同仍需要为新机器人调整动作表示。

## 4. 通俗背景

预训练通用策略怎样适应新传感器和动作空间？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

预训练通用策略怎样适应新传感器和动作空间？

## 6. 已有方法的不足

固定输入输出结构限制跨平台适配；每个机器人从零训练效率低。

## 7. 核心方法

模块化tokenizer、Transformer共享主干与可适配动作readout。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。JAX工具链与新平台接口增加学习成本。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

图像/语言或目标图→共享表示→动作生成头。

## 10. 公式、变量和推导思路

本轮未逐式核验其扩散目标；先对照III-A与III-C，不把OpenVLA的离散损失直接套用。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

III-A至III-E，IV-B新领域微调、IV-C设计消融。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

约800k跨具身轨迹、9机器人评估（摘要）；具体设备以III-D为准。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

展示少量平台内数据可适配；“数小时消费GPU”需绑定具体设置，非12GB保证。

## 14. 局限及待检验研究空白

JAX工具链与新平台接口增加学习成本；泛化依赖数据。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

OXE→Octo；与OpenVLA提供不同开放通用策略设计。

## 16. 本科毕设与博士价值

接口思想值得借鉴，本科不必同时维护多套训练框架。

## 17. 阅读前置知识

Transformer、扩散策略、传感器与动作空间。

## 18. 精读与略读建议

优先精读：III-A至III-E，IV-B新领域微调、IV-C设计消融。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **Octo是LLM吗？** 不是简单的大语言模型。
2. **输入空间固定吗？** 强调适配性。
3. **800k是OpenVLA训练集吗？** 混合不同。
4. **消费GPU等于12GB保证吗？** 不是。
5. **先读哪里？** III架构和IV-B。

## 20. 卡住时补什么

Transformer、扩散策略、传感器与动作空间。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- Octo: An Open-Source Generalist Robot Policy
- Abstract
- I Introduction
- II Related Work
- III The Octo Model
- III-A Architecture
- Task and observation tokenizers
- Transformer backbone and readout heads
- III-B Training data
- III-C Training objective
- III-D Training Details
- III-E Model Checkpoints & Code
- IV Experiments
- Evaluation setups
- Comparisons
- IV-A Octo Controls Multiple Robots Out-of-the-Box
- IV-B Octo Enables Data-Efficient Learning in New Domains
- IV-C Design Decisions for Generalist Robot Policy Training
- Model architecture
- Training data
- Training objective
- Model scale
- V Discussion and Future Work
- References
- Appendix A Contributions
- Appendix B Octo Code Example
- Appendix C Data mixture
- Appendix D Training Hyperparameters
- Appendix E Things that Worked and Did Not Work (Yet)
- Things we found improved performance
- Things that did not work (yet)
- Appendix F Experimental Setups
- F-A Zero-Shot Evaluations
- WidowX BridgeV2
- UR5
- RT-1 Robot
- F-B Model Ablations
- F-C Finetuning Evaluations
- CMU Baking
- Stanford Coffee
- Berkeley Peg Insertion
- Berkeley Pick Up
- Berkeley Coke
- Berkeley Bimanual

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://octo-models.github.io/
- 代码：https://github.com/octo-models/octo
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：https://huggingface.co/rail-berkeley（官方项目链接的组织入口；具体checkpoint未核）
- 权重核验深度：官方入口已定位；未下载、未加载验证
- arXiv首次/更新：2024/05/20 / 2024/05/26；当前HTML版本v2。
- 出版页面标注日期：2024/07/15；会议年份与论文集上线日期分开记录。
- 正式版题名：Octo: An Open-Source Generalist Robot Policy
- 正式版作者：Dibya Ghosh; Homer Rich Walke; Karl Pertsch; Kevin Black; Oier Mees; Sudeep Dasari; Joey Hejna; Tobias Kreiman; Charles Xu; Jianlan Luo; You Liang Tan; Lawrence Yunliang Chen; Quan Vuong; Ted Xiao; Pannag R Sanketi; Dorsa Sadigh; Chelsea Finn; Sergey Levine

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
