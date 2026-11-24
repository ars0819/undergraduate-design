---
citekey: DDPM
arxiv: '2006.11239'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# 去噪扩散概率模型

## 1. 英文标题与中文译名

Denoising Diffusion Probabilistic Models

去噪扩散概率模型（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Ho, Jonathan; Jain, Ajay; Abbeel, Pieter。首次提交：2020/06/19。NeurIPS 2020；正式会议论文。

[arXiv](https://arxiv.org/abs/2006.11239) · [正文HTML](https://arxiv.org/html/2006.11239) · [公开PDF](https://arxiv.org/pdf/2006.11239) · [正式来源](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)

正式DOI：本轮未核得，不猜测；arXiv DOI：10.48550/arXiv.2006.11239。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

DDPM 把生成数据描述成一系列逐渐去除噪声的步骤。正向过程按预设日程向真实样本加入高斯噪声，训练神经网络根据带噪样本和噪声级别预测噪声，反向再从随机噪声逐步恢复数据。它把复杂分布学习转化为大量局部去噪问题。原论文主要研究图像，而 Diffusion Policy 把类似思想搬到机器人动作序列。你的阅读重点是分清数据时间、噪声时间和网络输入，先理解简化噪声预测损失，再补变分下界推导；不要把训练时加噪与机器人执行中的物理噪声当成同一件事。

## 4. 通俗背景

怎样用可训练的多步反向过程生成高质量样本？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

怎样用可训练的多步反向过程生成高质量样本？

## 6. 已有方法的不足

直接拟合复杂数据分布困难；先前生成方法在质量、稳定性和似然目标之间存在取舍。

## 7. 核心方法

固定高斯正向过程；参数化反向均值；简化噪声预测目标。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。多步采样慢。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

训练x0、噪声ε、噪声步t→x_t→预测ε；采样从高斯逐步生成x0。

## 10. 公式、变量和推导思路

原文式(4)：x_t=√ᾱ_t x0+√(1−ᾱ_t)ε；式(14)简化目标E||ε−εθ(x_t,t)||²。ᾱ_t为各步保留比例乘积。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

§2背景、§3.2反向参数化、§3.4简化目标；算法1训练、算法2采样；表1样本质量。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

CIFAR10和高分辨率图像实验；其网络和采样预算不能直接当作动作策略配置。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

FID衡量图像分布距离；不要用FID判断机器人控制成败。

## 14. 局限及待检验研究空白

多步采样慢；简化损失与完整似然目标并不完全一致；图像质量不能保证控制可行性。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

DDPM→Diffusion Policy；流匹配共享连续生成思想但训练形式不同。

## 16. 本科毕设与博士价值

为动作头加速、采样步数实验建立概念基础，不必先训练图像扩散模型。

## 17. 阅读前置知识

高斯分布、期望、方差、条件概率、MSE。

## 18. 精读与略读建议

优先精读：§2背景、§3.2反向参数化、§3.4简化目标；算法1训练、算法2采样；表1样本质量。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **t是控制时间吗？** 这里是噪声步。
2. **ε是输出图片？** 网络预测噪声。
3. **可直接跳到任意步吗？** 需合适采样规则。
4. **FID是成功率吗？** 不是。
5. **先看哪个公式？** 式(4)和式(14)。

## 20. 卡住时补什么

高斯分布、期望、方差、条件概率、MSE。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- Denoising Diffusion Probabilistic Models
- Abstract
- 1 Introduction
- 2 Background
- 3 Diffusion models and denoising autoencoders
- 3.1 Forward process and L T L_{T}
- 3.2 Reverse process and L 1 : T − 1 L_{1:T-1}
- 3.3 Data scaling, reverse process decoder, and L 0 L_{0}
- 3.4 Simplified training objective
- 4 Experiments
- 4.1 Sample quality
- 4.2 Reverse process parameterization and training objective ablation
- 4.3 Progressive coding
- Progressive lossy compression
- Progressive generation
- Connection to autoregressive decoding
- 4.4 Interpolation
- 5 Related Work
- 6 Conclusion
- Broader Impact
- Acknowledgments and Disclosure of Funding
- References
- Extra information
- LSUN
- Progressive compression
- Appendix A Extended derivations
- Appendix B Experimental details
- Appendix C Discussion on related work
- Appendix D Samples
- Additional samples
- Latent structure and reverse process stochasticity
- Coarse-to-fine interpolation

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://hojonathanho.github.io/diffusion/
- 代码：https://github.com/hojonathanho/diffusion
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：未核得独立权重，或不适用
- 权重核验深度：本轮未核得独立权重；综述/算子/数据工作可能不适用
- arXiv首次/更新：2020/06/19 / 2020/12/16；当前HTML版本v2。
- 出版页面标注日期：2020；会议年份与论文集上线日期分开记录。
- 正式版题名：Denoising Diffusion Probabilistic Models
- 正式版作者：Ho, Jonathan; Jain, Ajay; Abbeel, Pieter

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
