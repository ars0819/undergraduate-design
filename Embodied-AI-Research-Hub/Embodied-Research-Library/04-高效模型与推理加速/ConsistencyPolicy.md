---
citekey: ConsistencyPolicy
arxiv: '2405.07503'
checked: 2026-10-24
priority: 扩展12
source_level: 摘要与目录导读
tags: [具身智能, 论文学习]
---

# 一致性策略：通过一致性蒸馏加速视觉运动策略

## 1. 英文标题与中文译名

Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation

一致性策略：通过一致性蒸馏加速视觉运动策略（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Prasad, Aaditya; Lin, Kevin; Wu, Jimmy; Zhou, Linqi; Bohg, Jeannette。首次提交：2024/05/13。RSS 2024；正式会议论文。

[arXiv](https://arxiv.org/abs/2405.07503) · [正文HTML](https://arxiv.org/html/2405.07503) · [公开PDF](https://arxiv.org/pdf/2405.07503) · [正式来源](https://roboticsproceedings.org/rss20/p071.html)

正式DOI：10.15607/RSS.2024.XX.071；arXiv DOI：10.48550/arXiv.2405.07503。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要与目录级。所列章节供后续精读定位，不表示已经精读这些正文段落。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

Consistency Policy 将已学好的扩散策略当老师，训练学生用较少生成步骤得到相近动作。它利用同一生成轨迹不同噪声点应对应一致结果的约束，把原本多步推理的部分工作转移到训练阶段。论文在模拟任务和真实操作中比较速度与成功率，也研究教师质量和采样设置的影响。对你来说，它提供了动作头加速的另一种思路：不只是缓存中间特征，也可以学习一个更快的生成过程。但学生训练需要教师数据和额外成本，不能把推理加速描述成完全免费的收益。

## 4. 通俗背景

怎样让扩散动作策略少步生成且保留性能？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

怎样让扩散动作策略少步生成且保留性能？

## 6. 已有方法的不足

多步去噪延迟高，受限移动硬件难部署。

## 7. 核心方法

从扩散教师蒸馏一致性学生，减少采样步数。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。蒸馏成本与教师偏差。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

观测与噪声→少步动作生成；训练另需教师。

## 10. 公式、变量和推导思路

本轮摘要/结构级，不转录一致性损失；应精读III-B再做实现。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

III-B教师学生、III-C推理、IV-D消融、V局限。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

6模拟任务、3真实任务；论文含笔记本GPU推理，未本机实测。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

作者报告数量级推理改善且成功率有竞争力；具体GPU和基线仍须查实验表。

## 14. 局限及待检验研究空白

蒸馏成本与教师偏差；少步模型可丢失动作模式。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

Diffusion Policy→Consistency Policy，与EfficientVLA缓存互补但不能直接相乘收益。

## 16. 本科毕设与博士价值

本科候选D的参考；先用小动作任务验证。

## 17. 阅读前置知识

DDPM、蒸馏、一致性、动作序列。

## 18. 精读与略读建议

优先精读：III-B教师学生、III-C推理、IV-D消融、V局限。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **需要老师吗？** 该方法从预训练扩散策略蒸馏。
2. **免训练吗？** 不是。
3. **少步一定无损吗？** 需评估。
4. **能只看单前向吗？** 还要训练与闭环成本。
5. **先读哪里？** III和IV-D。

## 20. 卡住时补什么

DDPM、蒸馏、一致性、动作序列。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- Consistency Policy Accelerated Visuomotor Policies via Consistency Distillation
- Abstract
- I Introduction
- II Related Work
- III Consistency Policy
- III-A Preliminaries
- III-B Training
- III-B 1 Teacher Model (EDM)
- III-B 2 Student Model (Consistency Policy)
- III-C Inference
- III-D Implementation Details
- IV Experiments
- IV-A Baselines
- IV-B Simulation Experiments
- IV-C Real World Experiments
- IV-D Ablations
- V Limitations
- VI Conclusion
- References
- Overview
- Appendix A Inference Speeds
- Appendix B Low Variance Initial Samples
- Appendix C Real World Experiment Setup

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://consistency-policy.github.io/
- 代码：https://github.com/Aaditya-Prasad/Consistency-Policy
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：未核得独立权重，或不适用
- 权重核验深度：本轮未核得独立权重；综述/算子/数据工作可能不适用
- arXiv首次/更新：2024/05/13 / 2024/06/28；当前HTML版本v2。
- 出版页面标注日期：2024/07/15；会议年份与论文集上线日期分开记录。
- 正式版题名：Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation
- 正式版作者：Aaditya Prasad; Kevin Lin; Jimmy Wu; Linqi Zhou; Jeannette Bohg

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
