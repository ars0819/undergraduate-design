---
citekey: SurveyEmbodied
arxiv: '2405.14093'
checked: 2026-10-24
priority: 扩展12
source_level: 摘要与目录导读
tags: [具身智能, 论文学习]
---

# 具身人工智能视觉语言动作模型综述

## 1. 英文标题与中文译名

A Survey on Vision-Language-Action Models for Embodied AI

具身人工智能视觉语言动作模型综述（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文（综述）。作者：Ma, Yueen; Song, Zixing; Zhuang, Yuzheng; Hao, Jianye; King, Irwin。首次提交：2024/05/23。IEEE TNNLS (2026)，arXiv列Early Access；正式期刊论文，卷页需出版社二次核验。

[arXiv](https://arxiv.org/abs/2405.14093) · [正文HTML](https://arxiv.org/html/2405.14093) · [公开PDF](https://arxiv.org/pdf/2405.14093)

正式DOI：10.1109/TNNLS.2025.3650584；arXiv DOI：10.48550/arXiv.2405.14093。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要与目录级。所列章节供后续精读定位，不表示已经精读这些正文段落。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

这篇综述把视觉语言动作领域整理为部件、低层策略和高层规划等相互联系的研究线索。它提醒你，具身智能并不只是在图像前面加一句指令：视觉表示、世界模型、策略学习和规划都可能成为研究对象。学习时可以先用它建立术语地图，再回到具体方法论文检查细节。由于综述从2024年预印本持续更新到2026年期刊版本，必须记录所读版本，不能把早期网页内容当成当前论文的全部结论。对初学者，分类的作用是提出问题，不是把所有论文硬塞进唯一类别。

## 4. 通俗背景

如何组织快速增长的VLA组件、策略和规划研究？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

如何组织快速增长的VLA组件、策略和规划研究？

## 6. 已有方法的不足

术语跨度大，相关领域常割裂。

## 7. 核心方法

按组件、低层控制策略与高层任务规划分类，汇总资源和挑战。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。版本更新和分类边界可能改变，不能代替方法复核。。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

综述无统一模型输入输出。

## 10. 公式、变量和推导思路

无本卡需推导的新目标；引用公式应回到对应原论文。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

II背景、III分类、数据/模拟器/基准相关章节；实际完整目录附卡。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

非统一重测；条目来自多来源文献。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

不产生可直接比较的统一系统成功率。

## 14. 局限及待检验研究空白

版本更新和分类边界可能改变，不能代替方法复核。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

与RealWorld综述互补；前者分类地图，后者部署全链。

## 16. 本科毕设与博士价值

作为概念补查索引。

## 17. 阅读前置知识

VLA、模仿学习、强化学习、规划。

## 18. 精读与略读建议

优先精读：II背景、III分类、数据/模拟器/基准相关章节；实际完整目录附卡。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **综述能代替原论文吗？** 不能。
2. **三条线互斥吗？** 可交叉。
3. **首发年份与期刊年份相同吗？** 不同。
4. **需要背全部分类吗？** 不需要。
5. **该如何使用？** 带问题查章节。

## 20. 卡住时补什么

VLA、模仿学习、强化学习、规划。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- A Survey on Vision-Language-Action Models for Embodied AI
- Abstract
- Index Terms:
- I Introduction
- I-A Related Work
- I-B Contributions
- II Background
- III Vision-Language-Action Models
- III-A Components of VLA
- III-A 1 Reinforcement Learning
- III-A 2 Pretrained Visual Representations
- III-A 3 Video Representations
- III-A 4 Dynamics Learning
- III-A 5 World Models
- III-A 6 LLM-Induced World Models
- III-A 7 Visual World Models
- III-A 8 Reasoning
- III-A 9 Policy Steering
- III-A 10 Strengths and Limitations
- PVRs
- Forward & Inverse Dynamics
- World Models & Reasoning
- III-B Low-Level Control Policies
- III-B 1 Non-Transformer Control Policies
- III-B 2 Transformer-Based Control Policies
- III-B 3 Control Policies for Multimodal Instructions
- III-B 4 Control Policies with 3D Vision
- III-B 5 Diffusion-Based Control Policies
- III-B 6 Diffusion-Based Control Policies with 3D Vision
- III-B 7 Control Policies for Motion Planning
- III-B 8 Control Policies with Point-Based Actions
- III-B 9 Large VLA
- III-B 10 Strengths and Limitations
- Architectures
- Action Types and Their Training Objectives
- RT Series
- LVLA vs. Generalized VLA
- Scaling Laws
- IV Task Planners
- IV-A Monolithic Task Planners
- IV-A 1 End-to-End Task Planners
- IV-A 2 End-to-End Task Planners with 3D Vision
- IV-A 3 Grounded Task Planners
- IV-B Modular Task Planners
- IV-B 1 Language-Based Task Planners
- IV-B 2 Code-Based Task Planners
- IV-C Strengths and Limitations
- V Datasets and Benchmarks
- V-A Real-World Robot Datasets & Benchmarks
- V-B Simulators, Simulated Robot Datasets & Benchmarks
- V-C Automated Dataset Collection
- V-D Human Datasets
- V-E Task Planning Benchmarks
- V-F Embodied Question Answering (EQA) Benchmarks
- VI Challenges and Future Directions
- Safety First
- Datasets & Benchmarks
- Foundation Models & Generalization
- Multimodality
- Framework for Long-Horizon Tasks
- Real-Time Responsiveness
- Multiagent Systems
- Ethical and Societal Implications
- Applications
- VII Conclusion
- Acknowledgments
- References
- -A Background
- -A 1 Unimodal Models
- -A 2 Vision-Language Models
- -B Background (Extended Version): Unimodal Models
- -B 1 Computer Vision
- Convolutional Neural Network
- Vision Transformer
- Vision in 3D
- -B 2 Natural Language Processing
- Early Developments
- Recurrent Neural Network & Convolutional Neural Network
- Transformer & Large Language Model
- -B 3 Reinforcement Learning
- Deep Reinforcement Learning
- Robotics
- -B 4 Graph
- Graph and Vision
- Graph and Language
- -C Background (Extended Version): Vision-Language Models
- -C 1 Self-Supervised Pretraining
- -C 2 Contrastive Pretraining
- -C 3 Large Multimodal Model
- -D Supplementary Related Work
- -E Additional VLA-Related Work
- -E 1 Components of VLA: Pretraining
- -E 2 Components of VLA: World Models
- -E 3 Components of VLA: Imitation Learning
- -E 4 Subsequent Surveys on VLA
- -E 5 Latest Developments of VLA
- -E 6 Beyond VLA

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://github.com/yueen-ma/Awesome-VLA
- 代码：未核得独立官方实现
- 代码核验深度：本轮未核得独立官方实现；不等于未开源
- 权重：未核得独立权重，或不适用
- 权重核验深度：本轮未核得独立权重；综述/算子/数据工作可能不适用
- arXiv首次/更新：2024/05/23 / 2026/05/01；当前HTML版本v8。
- 出版页面标注日期：未提取；会议年份与论文集上线日期分开记录。
- 正式版题名：A Survey on Vision-Language-Action Models for Embodied AI
- 正式版作者：未从出版元数据抽取；上方保留arXiv作者

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
