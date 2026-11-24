---
citekey: OpenVLAOFT
arxiv: '2502.19645'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# 微调视觉语言动作模型：同时优化速度与成功率

## 1. 英文标题与中文译名

Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success

微调视觉语言动作模型：同时优化速度与成功率（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Kim, Moo Jin; Finn, Chelsea; Liang, Percy。首次提交：2025/02/27。RSS 2025；正式会议论文。

[arXiv](https://arxiv.org/abs/2502.19645) · [正文HTML](https://arxiv.org/html/2502.19645) · [公开PDF](https://arxiv.org/pdf/2502.19645) · [正式来源](https://roboticsproceedings.org/rss21/p017.html)

正式DOI：10.15607/RSS.2025.XXI.017；arXiv DOI：10.48550/arXiv.2502.19645。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

这篇工作把“怎样微调 VLA”当成一个可以系统研究的问题。以 OpenVLA 为基础，作者比较动作解码方式、离散或连续表示、动作分块和训练目标，形成 OFT 配方：并行解码、连续动作块和简单的 L1 目标共同提高性能与动作生成吞吐。后续版本还加入语言条件调制。阅读时不能把整套改动的收益归给某一个开关，也不能把每秒生成多少动作直接理解成机器人有多快完成任务。它帮助你判断大型模型加速有时来自改变输出结构，而不仅是压缩权重。

## 4. 通俗背景

VLA微调中的动作表示、解码和目标如何影响成功与效率？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

VLA微调中的动作表示、解码和目标如何影响成功与效率？

## 6. 已有方法的不足

默认自回归/离散设计未必适合高频动作；微调超参和输入输出设计未系统比较。

## 7. 核心方法

并行动作解码 + chunking + 连续表示 + L1回归；更新版OFT+含FiLM增强语言联系。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。多个设计同时变化。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

多相机、状态、指令→并行连续动作块；与原OpenVLA逐Token输出不同。

## 10. 公式、变量和推导思路

教学重写：L1=E||A_hat−A_demo||_1。简单目标有效不意味着复杂分布建模普遍无用，需绑定任务与数据分布。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

IV设计比较；V-C效率、V-D输入输出灵活性；VI双臂；附录A-D训练细节。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

OpenVLA-7B基础；LIBERO及ALOHA。增加相机与状态时，必须与同输入基线比较。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

摘要给76.5%→97.1%与26倍动作生成吞吐；这是整套配方与协议下比较，不是单次控制延迟缩短26倍。

## 14. 局限及待检验研究空白

多个设计同时变化；真实任务仍需示范；大主干显存和推理内核门槛存在。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

OpenVLA→OFT；FAST压缩离散动作序列，OFT走连续并行输出。

## 16. 本科毕设与博士价值

动作头设计与接口评估的重要文献；12GB先读，硬件实测后再决定基线。

## 17. 阅读前置知识

L1/L2、并行解码、动作分块、输入公平性。

## 18. 精读与略读建议

优先精读：IV设计比较；V-C效率、V-D输入输出灵活性；VI双臂；附录A-D训练细节。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **更快的是哪个指标？** 动作生成吞吐。
2. **L1一定优于扩散吗？** 不是普遍结论。
3. **比较多相机与单相机公平吗？** 应分协议报告。
4. **OFT等于权重量化吗？** 不是。
5. **新版变化要记录吗？** 要，特别是FiLM。

## 20. 卡住时补什么

L1/L2、并行解码、动作分块、输入公平性。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
- Abstract
- I Introduction
- II Related Work
- III Preliminaries
- IV Studying Key VLA Fine-Tuning Design Decisions
- IV-A VLA Fine-Tuning Design Decisions
- IV-B Implementing Alternative Design Components
- IV-C Augmenting OpenVLA-OFT with FiLM for Enhanced Language Grounding.
- V Experiments: Evaluating VLA Fine-Tuning Design Decisions
- V-A LIBERO Experimental Setup
- V-B LIBERO Task Performance Comparisons
- V-C LIBERO Inference Efficiency Comparisons
- V-D Model Input-Output Flexibility
- V-E Optimized Fine-Tuning Recipe
- V-F Additional Experiments
- VI Experiments: Adapting OpenVLA to a Real-World ALOHA Robot
- VI-A ALOHA Experimental Setup
- VI-B Methods in Comparison
- VI-C ALOHA Task Performance Results
- VI-D ALOHA Inference Efficiency Comparisons
- VII Discussion
- VIII Limitations
- Acknowledgments
- References
- Appendix A Appendix
- A-A Model Architecture Details
- A-B Implementation Details
- A-B 1 Parallel Decoding Implementation
- A-B 2 Continuous Action Representations
- A-B 3 Input Processing Details
- A-C Feature-wise Linear Modulation (FiLM) Implementation Details
- A-D OpenVLA-OFT Hyperparameters and Training Details
- A-E Baseline Methods Hyperparameters and Training Details
- A-F ALOHA Evaluation Details
- A-F 1 ALOHA Task Suite Details
- A-F 2 ALOHA Task Scoring Rubric
- A-G Additional Experiments
- A-G 1 Single OpenVLA-OFT Policy for All LIBERO Task Suites Combined
- A-G 2 Ablating FiLM in LIBERO
- A-G 3 Ablating the OpenVLA Pretrained Representation
- A-G 4 Scaling Up OpenVLA-OFT to a Larger Real-World Dataset (BridgeData V2)

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://openvla-oft.github.io/
- 代码：https://github.com/moojink/openvla-oft
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial
- 权重核验深度：官方入口已定位；未下载、未加载验证
- arXiv首次/更新：2025/02/27 / 2025/04/28；当前HTML版本v2。
- 出版页面标注日期：2025/06/21；会议年份与论文集上线日期分开记录。
- 正式版题名：Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
- 正式版作者：Moo Jin Kim; Chelsea Finn; Percy Liang

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
