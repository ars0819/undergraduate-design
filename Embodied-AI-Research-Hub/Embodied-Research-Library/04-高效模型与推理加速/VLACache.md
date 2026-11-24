---
citekey: VLACache
arxiv: '2502.02175'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# VLA-Cache：通过自适应Token缓存实现高效视觉语言动作操作

## 1. 英文标题与中文译名

VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching

VLA-Cache：通过自适应Token缓存实现高效视觉语言动作操作（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Xu, Siyu; Wang, Yunke; Xia, Chenghao; Zhu, Dihao; Huang, Tao; Xu, Chang。首次提交：2025/02/04。NeurIPS 2025；正式会议论文。

[arXiv](https://arxiv.org/abs/2502.02175) · [正文HTML](https://arxiv.org/html/2502.02175) · [公开PDF](https://arxiv.org/pdf/2502.02175) · [正式来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f062da1973ac9ac61fc6d44dd7fa309f-Abstract-Conference.html)

正式DOI：10.52202/085713-5484；arXiv DOI：10.48550/arXiv.2502.02175。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

机器人相邻时刻的画面通常只有局部发生变化，若每次都重算所有视觉相关表示，就会浪费计算。VLA-Cache 用图像块相似度找出静态区域，再排除任务相关、即便外观变化小也应重新计算的区域，复用其他区域在语言模型中的键值表示。它还按层的注意力情况调整复用比例。关键不只是“缓存更多”，而是在节省计算与保持最新任务信息之间取舍。轮式机器人的相机移动会破坏相同像素位置的静态假设，因此迁移到 OpenFleX 时，移动阶段的缓存失效应成为独立实验。

## 4. 通俗背景

如何跨控制时刻复用静态视觉Token且保留动作相关信息？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

如何跨控制时刻复用静态视觉Token且保留动作相关信息？

## 6. 已有方法的不足

标准KV缓存只省同一次自回归解码；静态块全部复用会损伤精细控制。

## 7. 核心方法

相邻图像patch余弦相似度 + 任务相关过滤 + 逐层自适应KV复用。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。移动相机、光照与接触小变化可令缓存过期。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

当前图像、前帧表示、文本与缓存→选取重算Token→动作；缓存必须跨时刻管理。

## 10. 公式、变量和推导思路

原文式(4) Sim(P_t,P_(t−1))=点积/(两者范数乘积)。式(5)阈值加Top-k挑静态块；相似不代表语义无关。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

§3.1–3.4；图2整体；表1逐步加入静态选择、任务过滤、层适应的消融。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

OpenVLA等基策略；LIBERO和SIMPLER及真实操作。具体速度表绑定GPU/输入和缓存策略。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

当前正文表1：84.4%/51.56ms→仅静态74.2%/31.03ms→完整83.8%/32.22ms（LIBERO Spatial）。不把这些跨设备套用。

## 14. 局限及待检验研究空白

移动相机、光照与接触小变化可令缓存过期；视觉前端是否仍重算需查实现。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

Transformer KV→跨帧VLA缓存；与EfficientVLA去噪步缓存的时间轴不同。

## 16. 本科毕设与博士价值

很适合做静止/移动相机下缓存失效与刷新实验；需真实闭环评估才可宣称控制改进。

## 17. 阅读前置知识

余弦相似、attention、KV缓存、张量索引与mask。

## 18. 精读与略读建议

优先精读：§3.1–3.4；图2整体；表1逐步加入静态选择、任务过滤、层适应的消融。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **缓存的是全部原图吗？** 核心复用KV表示。
2. **静态必可复用吗？** 任务相关块仍须更新。
3. **底盘运动有何影响？** 同位置patch不再对应同物体。
4. **只降FLOPs够吗？** 还需计时和成功率。
5. **缓存与删Token相同吗？** 不是。

## 20. 卡住时补什么

余弦相似、attention、KV缓存、张量索引与mask。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching
- Abstract
- 1 Introduction
- 2 Related Work
- 3 Methodology
- 3.1 KV Cache for VLA Token Reusing
- 3.2 Temporal Redundancy in Robotic Perception
- 3.3 Retaining Task-Relevant Information
- 3.4 Layer Adaptive Token Reusing
- 4 Implementations
- 4.1 Cross-Frame Visual Token Caching
- 4.2 Theoretical Analysis of Computational Complexity
- 5 Experiment
- 5.1 Experiment Setup
- 5.2 Evaluation Benchmark
- 5.3 Results on Simulation Environment
- Main Results on LIBERO.
- Main Results on SIMPLER
- 5.4 Results on Real Robot
- 6 Conclusion
- Acknowledgments
- References
- Appendix A Limitations
- Appendix B Impact Statement
- Appendix C Complexity Analysis Details
- Appendix D Inference Detail of VLA-Cache
- Inference Procedure.
- Token Reuse Mechanism.
- Experimental Settings.
- Appendix E More Experiment Results
- E.1 Simulation Experiment
- LIBERO Task Definitions.
- SIMPLER Task Definitions.
- Implementation Details.
- E.2 Additional Simulation Results
- Result of Subtask on LIBERO Spatial Task Suit.
- Visualization Results.
- E.3 Additional Ablations and Comparisons
- Attention vs. object–mask proxies for task relevance.
- Sensitivity to static-token budget k k and relevance threshold τ \tau .
- Applicability to diffusion-based and alternative VLA architectures.
- E.4 Real Robot Experiment
- Robot Setup.
- Data Collection.
- Data Preprocessing.
- Task Definitions and Success Criteria.
- Trial Protocol and Randomization.
- Results with Counts and Rates.
- Visualization Results.

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://vla-cache.github.io/
- 代码：https://github.com/siyuhsu/vla-cache
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：未核得独立权重，或不适用
- 权重核验深度：本轮未核得独立权重；综述/算子/数据工作可能不适用
- arXiv首次/更新：2025/02/04 / 2025/10/21；当前HTML版本v2。
- 出版页面标注日期：2026-04-23；会议年份与论文集上线日期分开记录。
- 正式版题名：VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching
- 正式版作者：Xu, Siyu; Wang, Yunke; Xia, Chenghao; Zhu, Dihao; Huang, Tao; Xu, Chang

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
