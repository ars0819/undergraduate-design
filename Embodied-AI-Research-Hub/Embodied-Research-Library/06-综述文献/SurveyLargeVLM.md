---
citekey: SurveyLargeVLM
arxiv: '2508.13073'
checked: 2026-10-24
priority: 扩展12
source_level: 摘要与目录导读
tags: [具身智能, 论文学习]
---

# 基于大型视觉语言模型的机器人操作VLA综述

## 1. 英文标题与中文译名

Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey

基于大型视觉语言模型的机器人操作VLA综述（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文（综述）。作者：Shao, Rui; Li, Wei; Zhang, Lingsen; Zhang, Renshan; Liu, Zhiyang; Chen, Ran; Nie, Liqiang。首次提交：2025/08/18。arXiv；正式状态未核；预印本已核；正式状态待核。

[arXiv](https://arxiv.org/abs/2508.13073) · [正文HTML](https://arxiv.org/html/2508.13073) · [公开PDF](https://arxiv.org/pdf/2508.13073)

正式DOI：本轮未核得，不猜测；arXiv DOI：10.48550/arXiv.2508.13073。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要与目录级。所列章节供后续精读定位，不表示已经精读这些正文段落。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

这篇综述聚焦以大型视觉语言模型为基础的机器人操作策略，按整体式和分层式结构梳理研究。整体式中也可能有不同子系统，分层式则通过中间表示更明确地区分规划与执行。作者进一步讨论强化学习、免训练优化、人类视频学习和世界模型等交叉主题。你可以用它比较 ForeAct、OpenVLA 与轻量动作专家分别在系统中承担什么角色。分类名称本身不证明哪一种更好；具体选择仍取决于延迟预算、任务反馈速度、数据和评估方式。

## 4. 通俗背景

大型VLM如何以不同结构接入机器人操作？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

大型VLM如何以不同结构接入机器人操作？

## 6. 已有方法的不足

术语和结构分类不统一，跨方向研究碎片化。

## 7. 核心方法

整体式与分层式分类，讨论RL、免训练优化、视频与世界模型。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。分类不能覆盖所有混合架构。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

综述，不提供一个统一策略接口。

## 10. 公式、变量和推导思路

无新统一损失要推导；具体方法公式回原文。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

II背景、III整体式、IV分层式及后续交叉方向。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

汇总不同机器人、数据与基准，非统一实测。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

不从综述表格推断算法在相同硬件下的排行。

## 14. 局限及待检验研究空白

分类不能覆盖所有混合架构；更新范围受版本限制。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

与ForeAct分层视觉规划、OpenVLA端到端动作预测对照。

## 16. 本科毕设与博士价值

帮助本科选结构范围，避免把所有方向同时做。

## 17. 阅读前置知识

VLM、动作专家、层级规划。

## 18. 精读与略读建议

优先精读：II背景、III整体式、IV分层式及后续交叉方向。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **双系统就是分层吗？** 按文中定义区分。
2. **结构决定性能吗？** 还看数据与协议。
3. **为什么看综述？** 定位问题与来源。
4. **能直接抄表比较吗？** 先核对原实验。
5. **代码仓库是模型吗？** 此处主要是文献索引。

## 20. 卡住时补什么

VLM、动作专家、层级规划。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey
- Abstract
- Index Terms:
- I Introduction
- II Background
- II-A Evolution of Vision-Language Models
- II-B Advancements in Robotic Manipulation
- II-C Relevant Surveys and Taxonomies
- III Monolithic Models
- III-A Single-system Models
- III-A 1 Classic Paradigm: Autoregressive Decoding
- III-A 2 Model Performance Enhancement
- III-A 3 Inference Efficiency Optimization
- III-B Dual-system Models
- III-B 1 Cascade-based Methods
- III-B 2 Parallel-based Methods
- IV Hierarchical Models
- IV-A Planner-Only
- IV-A 1 Program-based Methods
- IV-A 2 Keypoint-based Methods
- IV-A 3 Subtask-based Methods
- IV-B Planner+Policy
- IV-B 1 Keypoint-based Methods
- IV-B 2 Subtask-based Methods
- IV-C Comparison: Monolithic vs. Hierarchical
- V Other Advanced Field
- V-A Reinforcement Learning-based Methods
- V-B Training-Free Methods
- V-C Learning from Human Videos
- V-D World Model-based VLA
- VI Characteristics of VLA Models
- VI-A Multimodal Fusion
- VI-B Instruction Following
- VI-C Multi-Dimensional Generalization
- VII Conclusion
- References
- Appendix A Experimental Evaluation
- A-A Benchmark Performance Analysis
- A-A 1 Evaluation on Diverse Tasks
- A-A 2 Evaluation on Generalization Performance
- A-A 3 Evaluation on Real-World Custom Tasks
- Appendix B Datasets and Benchmarks
- B-A Real-world Robot Datasets
- B-B Simulation Datasets and Benchmarks
- B-C Human Behavior Datasets
- B-D Embodied Datasets and Benchmarks
- Appendix C Outlook
- C-A Future Directions
- C-B Discussion
- Appendix D Practical Model-Selection Guidelines

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
- 代码：未核得独立官方实现
- 代码核验深度：本轮未核得独立官方实现；不等于未开源
- 权重：未核得独立权重，或不适用
- 权重核验深度：本轮未核得独立权重；综述/算子/数据工作可能不适用
- arXiv首次/更新：2025/08/18 / 2026/08/29；当前HTML版本v3。
- 出版页面标注日期：未提取；会议年份与论文集上线日期分开记录。
- 正式版题名：Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey
- 正式版作者：未从出版元数据抽取；上方保留arXiv作者

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
