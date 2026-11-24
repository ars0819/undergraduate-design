---
citekey: SmolVLA
arxiv: '2506.01844'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# SmolVLA：面向低成本高效率机器人的视觉语言动作模型

## 1. 英文标题与中文译名

SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics

SmolVLA：面向低成本高效率机器人的视觉语言动作模型（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Shukor, Mustafa; Aubakirova, Dana; Capuano, Francesco; Kooijmans, Pepijn; Palma, Steven; Zouitine, Adil; Aractingi, Michel; Pascal, Caroline; Russi, Martino; Marafioti, Andres; Alibert, Simon; Cord, Matthieu; Wolf, Thomas; Cadene, Remi。首次提交：2025/06/02。arXiv；未核实正式发表；已确认预印本，正式发表状态待核。

[arXiv](https://arxiv.org/abs/2506.01844) · [正文HTML](https://arxiv.org/html/2506.01844) · [公开PDF](https://arxiv.org/pdf/2506.01844)

正式DOI：本轮未核得，不猜测；arXiv DOI：10.48550/arXiv.2506.01844。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

SmolVLA 试图降低机器人基础模型的使用门槛，而不是只追求最大模型规模。它把紧凑的视觉语言模型与流匹配动作专家组合起来，输入多路图像、语言任务和机器人状态，输出动作块。为减少计算，模型限制每张图像的视觉 Token 数，并只使用部分语言模型层；动作专家交替使用交叉注意力和自注意力。论文还把执行与推理解耦，研究异步控制。对你而言，它同时提供较小模型、开源工具链和时序问题三个入口，但模型能载入显存，并不表示无需适配就能操纵新机器人。

## 4. 通俗背景

低资源条件下怎样兼顾通用视觉语言理解和连续动作生成？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

低资源条件下怎样兼顾通用视觉语言理解和连续动作生成？

## 6. 已有方法的不足

大VLA成本高；小模型若只缩参数可能损失能力；同步块执行造成等待。

## 7. 核心方法

SmolVLM2特征 + 流匹配动作专家；视觉Token缩减、层截断和异步执行。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。新硬件仍需示范微调。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

多路RGB、状态、文本→VLM上下文→动作专家→50步动作块（论文默认）。

## 10. 公式、变量和推导思路

教学表达：Aτ=τA+(1−τ)ε；训练向量场拟合连接噪声与示范的方向。本文正文的τ方向与目标符号需对照实现，不能把不同论文时间约定直接拼接。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

§3.1架构、§3.3异步、§4.3实现、§4.6异步实验与§4.7消融；先读架构图，再看执行时序。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

论文主模型450M，动作专家约100M；仅取前16层；每帧64视觉Token；LIBERO、Meta-World与SO-100/101。论文预训练使用4 GPU，完整项目耗时不能等同于个人微调成本。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

主要指标SR；真实任务使用子任务分数，模拟使用二元成功。两者不能直接平均比较。论文固定10步流匹配推理。

## 14. 局限及待检验研究空白

新硬件仍需示范微调；原生Windows、5070驱动/算子和峰值显存未本机实测。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

ACT提供块基线；pi0提供流动作专家路线；VLASH进一步处理异步状态错位。

## 16. 本科毕设与博士价值

推荐作为12GB条件下优先评估的VLA候选；先离线推理，后小规模模拟，而非预训练。

## 17. 阅读前置知识

Transformer、行为克隆、流匹配直观、Python队列与时间戳。

## 18. 精读与略读建议

优先精读：§3.1架构、§3.3异步、§4.3实现、§4.6异步实验与§4.7消融；先读架构图，再看执行时序。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **小模型一定弱吗？** 不一定，要在相同任务协议比较。
2. **需要自己的数据吗？** 新平台通常需要。
3. **450M就是500M矛盾吗？** 论文与模型卡命名/计数口径需区分。
4. **异步等于更短前向吗？** 不是。
5. **12GB已保证？** 尚未实测。

## 20. 卡住时补什么

Transformer、行为克隆、流匹配直观、Python队列与时间戳。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- SmolVLA: A vision-language-action model for affordable and efficient robotics
- Abstract
- 1 Introduction
- 2 Related work
- Vision-language models (VLMs).
- Vision-language-action models (VLAs).
- 3 SmolVLA: small, efficient and capable
- Overview.
- 3.1 Model architecture
- Vision-language model (VLM).
- State, action, and feature projectors.
- Visual tokens reduction.
- Faster inference through layer skipping.
- Flow matching action expert.
- Interleaved cross and causal self-attention layers.
- 3.2 Pretraining data collected by the community
- Task annotation with VLM.
- Camera viewpoint normalization.
- 3.3 Asynchronous inference
- Implementation details
- 4 Experiments
- 4.1 Experimental setup
- Evaluation metrics.
- Simulated environments.
- Real-world tasks.
- 4.2 Robots
- 4.3 Implementation details.
- 4.4 Baselines
- π 𝟎 \bf{\pi}_{0} ( Black et al., 2024 ) .
- ACT ( Zhao et al., 2023 ) .
- 4.5 Main results
- Simulation Evaluation.
- Real-World Evaluation.
- Effect of pretraining and multitask learning.
- 4.6 Asynchronous inference
- Results.
- 4.7 Ablation Study
- Cross-attention (CA) vs. self-attention (SA) between VLM and 𝐯 θ \mathbf{v}_{\theta} .
- Causal vs. bidirectional attention on action tokens within 𝐯 θ \mathbf{v}_{\theta} .
- Using early LLM layers in the VLM.
- Action Expert Capacity.
- Regression vs. Flow Matching training objectives.
- States to the VLM or Action Expert?
- Action chunk size, n n .
- Number of executed actions before updating observations.
- 5 Discussion
- 5.1 Limitations
- 6 Aknowledgements
- References
- Appendix A Appendix
- A.1 Community datasets
- Task annotation
- List of datasets.

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://huggingface.co/docs/lerobot/smolvla
- 代码：https://github.com/huggingface/lerobot
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：https://huggingface.co/lerobot/smolvla_base
- 权重核验深度：官方入口已定位；未下载、未加载验证
- arXiv首次/更新：2025/06/02 / 2025/06/02；当前HTML版本v1。
- 出版页面标注日期：未提取；会议年份与论文集上线日期分开记录。
- 正式版题名：SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics
- 正式版作者：未从出版元数据抽取；上方保留arXiv作者

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
