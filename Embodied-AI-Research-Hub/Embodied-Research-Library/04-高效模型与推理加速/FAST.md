---
citekey: FAST
arxiv: '2501.09747'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# FAST：用于视觉语言动作模型的高效动作词元化

## 1. 英文标题与中文译名

FAST: Efficient Action Tokenization for Vision-Language-Action Models

FAST：用于视觉语言动作模型的高效动作词元化（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Pertsch, Karl; Stachowicz, Kyle; Ichter, Brian; Driess, Danny; Nair, Suraj; Vuong, Quan; Mees, Oier; Finn, Chelsea; Levine, Sergey。首次提交：2025/01/16。RSS 2025；正式会议论文。

[arXiv](https://arxiv.org/abs/2501.09747) · [正文HTML](https://arxiv.org/html/2501.09747) · [公开PDF](https://arxiv.org/pdf/2501.09747) · [正式来源](https://roboticsproceedings.org/rss21/p012.html)

正式DOI：10.15607/RSS.2025.XXI.012；arXiv DOI：10.48550/arXiv.2501.09747。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

高频动作在相邻时刻通常比较平滑，若把每个维度、每个时刻都单独分箱成 Token，序列很长而且包含大量重复。FAST 先把一段动作转换到频率空间，用离散余弦变换集中时序结构，再量化并用序列压缩方式形成更紧凑的动作符号。这样自回归模型不必为每一个采样点生成同样数量的 Token。它与权重量化不同：这里压缩的是动作数据的表示。你应该同时观察压缩率、动作重建误差和闭环任务成功，避免只因为 Token 少了就认定机器人会更灵巧。

## 4. 通俗背景

如何为高频连续动作设计紧凑且有效的离散Token表示？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

如何为高频连续动作设计紧凑且有效的离散Token表示？

## 6. 已有方法的不足

逐维逐时刻分箱序列长、时序冗余多，高频灵巧任务学习效果差。

## 7. 核心方法

DCT变到频域、量化系数、展平与BPE压缩；FAST+为跨数据动作tokenizer。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。压缩可能损失接触瞬间高频细节。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

连续动作序列→频率系数→离散Token；逆过程还原动作序列。

## 10. 公式、变量和推导思路

教学表达：c_k∝Σ_t a_t cos[π(t+1/2)k/T]。省略归一化常数的概念式，非原文完整编号；低频描述平滑趋势，高频描述快速变化。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

V-A DCT、V-B tokenization、V-C通用tokenizer；VI-D消融及附录B压缩/重建权衡。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

FAST+用约1M真实机器人动作轨迹；可应用不同动作维度与频率，但需核对配置与归一化。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

作者报告部分训练设置最高5倍训练提速；不等同推理/任务完成提速。

## 14. 局限及待检验研究空白

压缩可能损失接触瞬间高频细节；自回归仍需输出Token；重建好不必然控制好。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

OpenVLA简单分箱→FAST；与OFT连续输出形成可比较两条路线。

## 16. 本科毕设与博士价值

可先用本地小动作数组做无模型压缩实验，门槛低于训练VLA。

## 17. 阅读前置知识

离散余弦变换直观、采样频率、量化、BPE。

## 18. 精读与略读建议

优先精读：V-A DCT、V-B tokenization、V-C通用tokenizer；VI-D消融及附录B压缩/重建权衡。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **FAST量化模型权重吗？** 不是。
2. **高频可全部删除吗？** 接触信息可能重要。
3. **压缩率唯一指标？** 还看误差与成功率。
4. **训练快就是执行快吗？** 不是。
5. **需要真机才能入门吗？** 可先用动作序列。

## 20. 卡住时补什么

离散余弦变换直观、采样频率、量化、BPE。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- FAST: Efficient Action Tokenization for Vision-Language-Action Models Thanks: ∗ : Core contributors Correspondence to: research@physicalintelligence.company
- Abstract
- I Introduction
- II Related Work
- III Preliminaries
- IV Case Study: How Does Tokenization Affect VLA Training?
- V Efficient Action Tokenization via Time-Series Compression
- V-A Time-Series Compression via Discrete Cosine Transform
- V-B The FAST Tokenization Algorithm
- V-C A Universal Robot Action Tokenizer
- VI Experiments
- VI-A Experimental Setup
- VI-B Comparing Action Tokenizers for VLA Training
- VI-C Universal Action Tokenizer
- VI-D Ablation Studies
- VI-E Comparing FAST to Diffusion
- VI-F Scaling Autoregressive VLAs to Large Robot Datasets
- VII Discussion and Future Work
- Acknowledgements
- References
- -A Data Mixture for Training Universal Tokenizer
- -B Trading off Between Compression and Reconstruction
- -C Policy Training
- -D DROID Policy Setup
- -E Evaluation Tasks and Training Datasets

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 官方资源

项目：https://pi.website/research/fast

权重/Tokenizer：https://huggingface.co/physical-intelligence/fast（未下载验证）。

正式发表来源：https://roboticsproceedings.org/rss21/p012.html；DOI 10.15607/RSS.2025.XXI.012。
