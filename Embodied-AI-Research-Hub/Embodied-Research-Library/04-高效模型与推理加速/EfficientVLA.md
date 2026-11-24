---
citekey: EfficientVLA
arxiv: '2506.10100'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# EfficientVLA：视觉语言动作模型的免训练加速与压缩

## 1. 英文标题与中文译名

EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models

EfficientVLA：视觉语言动作模型的免训练加速与压缩（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Yang, Yantai; Wang, Yuhao; Wen, Zichen; Zhongwei, Luo; Zou, Chang; Zhang, Zhipeng; Wen, Chuan; Zhang, Linfeng。首次提交：2025/06/11。NeurIPS 2025；正式会议论文。

[arXiv](https://arxiv.org/abs/2506.10100) · [正文HTML](https://arxiv.org/html/2506.10100) · [公开PDF](https://arxiv.org/pdf/2506.10100) · [正式来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a2ef31a1e45908901adc0ca853a8faf-Abstract-Conference.html)

正式DOI：10.52202/085713-1365；arXiv DOI：10.48550/arXiv.2506.10100。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

EfficientVLA 没有把所有时间都归因于一个模块，而是联合检查语言主干、视觉 Token 和迭代动作头。它根据层输入输出的相似性删去变化较小的语言层，兼顾任务相关性与多样性挑选视觉 Token，再复用动作头去噪过程中变化不大的中间特征。它说明多种冗余可以同时存在，但组合优化也使结果解释更困难：成功率下降可能来自层删减、视觉遗漏或动作近似。对你的学习，最有价值的是学会画模块延迟占比，并通过单独消融判断每个优化到底带来什么。

## 4. 通俗背景

如何同时压缩VLM与扩散动作头中的不同冗余？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

如何同时压缩VLM与扩散动作头中的不同冗余？

## 6. 已有方法的不足

单模块优化可能遇到其他瓶颈；随意删层/Token可能损害动作语义。

## 7. 核心方法

非连续语言层剪枝 + 任务相关/多样性视觉选择 + 迭代动作头中间量缓存。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。层相似可能遗漏任务关键的小变化。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

多模态输入→压缩VLM条件→缓存加速扩散动作头→动作块。

## 10. 公式、变量和推导思路

原文式(1)重要性为1减去层输入输出平均余弦相似度。越相似，定义的重要性越低；这是启发式，不是控制价值的严格因果测量。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

§3.2、§3.3、动作头缓存节；图2三模块框架；§4主结果与消融。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

以CogACT扩散型VLA为主，SIMPLER基准；不是纯自回归OpenVLA的原样即插即用。设备和计时细节需要§4绑定。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

摘要报告CogACT上1.93倍推理提速、FLOPs剩28.9%、成功率下降0.6%；原摘要百分号口径需结合表格，不把它扩展成任务完成时间。

## 14. 局限及待检验研究空白

层相似可能遗漏任务关键的小变化；组合误差叠加；免训练不代表无需校准和工程适配。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

AWQ压比特、VLA-Cache跨帧复用、EfficientVLA跨层/Token/去噪步优化。

## 16. 本科毕设与博士价值

适合作为多模块分析范例；本科只挑一项变量做受控实验。

## 17. 阅读前置知识

Transformer残差、余弦相似、扩散动作头、消融设计。

## 18. 精读与略读建议

优先精读：§3.2、§3.3、动作头缓存节；图2三模块框架；§4主结果与消融。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **删层凭什么？** 输入输出相似度启发式。
2. **免训练等于零数据吗？** 不代表无统计/校准。
3. **缓存跨哪条时间轴？** 动作头去噪迭代。
4. **可直接套到全部VLA？** 需结构兼容。
5. **组合倍率能相加吗？** 不能。

## 20. 卡住时补什么

Transformer残差、余弦相似、扩散动作头、消融设计。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models
- Abstract
- 1 Introduction
- 2 Related Work
- 3 Method
- 3.1 Preliminaries: Vision-Language-Action Models
- 3.2 Vision-Language Model Pruning
- 3.2.1 Layer Redundancy Analysis
- 3.2.2 Importance-Driven Non-Contiguous Layer Pruning
- 3.3 Task-Relevance and Diversity-Driven Visual Token Pruning
- 3.3.1 Quantifying Task Relevance
- 3.3.2 Selection of Key Task-Relevant Tokens
- 3.3.3 Augmentative Selection Balancing Relevance and Diversity
- Task-Driven Augmentation.
- Diversity-Driven Augmentation.
- Final Pruned Visual Token Set.
- 3.4 Caching Intermediate Features in Action Prediction
- 3.4.1 Feature Generation and Temporal Coherence in DiT Blocks
- 3.4.2 Static N-Step Caching Implementation
- 4 Experiment
- 4.1 Experimental Settings
- 4.2 Results on Simulation Environment
- 4.3 Ablation Study
- 5 Conclusion
- References
- Appendix A Experimental Settings
- A.1 SIMPLER Environment
- A.2 Baselines
- Appendix B Impact Statement
- Appendix C Limitations

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：本轮未核得独立项目页
- 代码：未核得独立官方实现
- 代码核验深度：本轮未核得独立官方实现；不等于未开源
- 权重：未核得独立权重，或不适用
- 权重核验深度：本轮未核得独立权重；综述/算子/数据工作可能不适用
- arXiv首次/更新：2025/06/11 / 2025/06/11；当前HTML版本v1。
- 出版页面标注日期：2026-04-23；会议年份与论文集上线日期分开记录。
- 正式版题名：EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models
- 正式版作者：Yang, Yantai; Wang, Yuhao; Wen, Zichen; Zhongwei, Luo; Zou, Chang; Zhang, Zhipeng; Wen, Chuan; Zhang, Linfeng

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
