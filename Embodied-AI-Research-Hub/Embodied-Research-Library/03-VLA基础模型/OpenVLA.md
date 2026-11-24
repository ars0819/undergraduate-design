---
citekey: OpenVLA
arxiv: '2406.09246'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# OpenVLA：开源视觉语言动作模型

## 1. 英文标题与中文译名

OpenVLA: An Open-Source Vision-Language-Action Model

OpenVLA：开源视觉语言动作模型（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Kim, Moo Jin; Pertsch, Karl; Karamcheti, Siddharth; Xiao, Ted; Balakrishna, Ashwin; Nair, Suraj; Rafailov, Rafael; Foster, Ethan; Lam, Grace; Sanketi, Pannag; Vuong, Quan; Kollar, Thomas; Burchfiel, Benjamin; Tedrake, Russ; Sadigh, Dorsa; Levine, Sergey; Liang, Percy; Finn, Chelsea。首次提交：2024/06/13。CoRL 2024；PMLR 270于2025出版；正式会议论文。

[arXiv](https://arxiv.org/abs/2406.09246) · [正文HTML](https://arxiv.org/html/2406.09246) · [公开PDF](https://arxiv.org/pdf/2406.09246) · [正式来源](https://proceedings.mlr.press/v270/kim25c.html)

正式DOI：本轮未核得，不猜测；arXiv DOI：10.48550/arXiv.2406.09246。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

OpenVLA 将预训练视觉语言模型改造成机器人策略。视觉侧融合 DINOv2 与 SigLIP，语言侧采用 Llama 2，再把连续机器人动作分箱为离散 Token，用预测下一个 Token 的方式训练。它使用跨机器人示范数据，并公开代码、权重和适配流程，因而成为研究 VLA 效率的重要参照。它的价值不只在于成功率，还在于让动作表示、微调方法和部署成本可以被系统比较。你应先看清图像到动作的完整链路，再理解为什么逐个输出动作 Token 会成为加速研究的对象。

## 4. 通俗背景

怎样构建可访问、可微调且可部署的通用VLA？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

怎样构建可访问、可微调且可部署的通用VLA？

## 6. 已有方法的不足

闭源模型难复现；泛化能力不能替代对新机器人适配；逐Token解码有计算开销。

## 7. 核心方法

双视觉编码器融合 + Llama2 + 动作分箱；公开训练和LoRA/量化适配流程。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。5070 12GB不宜默认全精度放得下。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

RGB与指令→视觉特征投影→语言模型→离散动作Token→反归一化连续动作。

## 10. 公式、变量和推导思路

教学重写：L=−Σ_i log pθ(a_i|I,l,a_<i)；a_i是离散动作Token。每维256桶、按1%至99%分位设置范围；不是权重量化。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

§3.2训练、§3.3数据、§3.5资源；图2架构；§5微调和效率。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

7B语言主干配视觉编码器，训练混合约970k示范。主干FP16裸权重约14GB十进制，尚未算视觉、激活与缓存。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

作者比较多平台任务与数据高效适配；不要把模型论文的成功率差异解释为量化单因素贡献。

## 14. 局限及待检验研究空白

5070 12GB不宜默认全精度放得下；量化内核兼容与闭环精度需验证。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

RT2→OpenVLA→OpenVLA-OFT；AWQ和VLA-Cache可研究其不同瓶颈。

## 16. 本科毕设与博士价值

理解离散动作VLA基线，量化/缓存方案应先确认目标层与计算路径。

## 17. 阅读前置知识

交叉熵、Token、归一化、视觉编码器、LoRA。

## 18. 精读与略读建议

优先精读：§3.2训练、§3.3数据、§3.5资源；图2架构；§5微调和效率。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **动作Token是单词吗？** 编码形式相同而含义不同。
2. **分箱是权重量化吗？** 不是。
3. **7B FP16适合12GB吗？** 裸主干已超。
4. **LoRA代表推理很省吗？** 不直接代表。
5. **官方版本作者一致吗？** arXiv与正式版有差异，分别保留。

## 20. 卡住时补什么

交叉熵、Token、归一化、视觉编码器、LoRA。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- OpenVLA: An Open-Source Vision-Language-Action Model
- Abstract
- 1 Introduction
- 2 Related Work
- Visually-Conditioned Language Models
- Generalist Robot Policies
- Vision-Language-Action Models
- 3 The OpenVLA Model
- 3.1 Preliminaries: Vision-Language Models
- 3.2 OpenVLA Training Procedure
- 3.3 Training Data
- 3.4 OpenVLA Design Decisions
- 3.5 Infrastructure for Training and Inference
- 4 The OpenVLA Codebase
- 5 Experiments
- 5.1 Direct Evaluations on Multiple Robot Platforms
- 5.2 Data-Efficient Adaptation to New Robot Setups
- 5.3 Parameter-Efficient Fine-Tuning
- 5.4 Memory-Efficient Inference via Quantization
- 6 Discussion and Limitations
- Acknowledgments
- References
- Appendix A Data Mixture Details
- Appendix B Evaluation Tasks and Detailed Results
- B.1 BridgeData V2 WidowX Evaluation Details
- B.1.1 BridgeData V2 Evaluation Tasks
- B.1.2 Comparing Evaluation Tasks to Original BridgeData V2 Training Data
- B.1.3 Detailed BridgeData V2 Evaluation Results
- B.2 Google Robot Evaluation Details
- B.2.1 Google Robot Evaluation Tasks
- B.2.2 Detailed Google Robot Evaluation Results
- B.3 Data-Efficient Adaptation Experiment Details
- B.3.1 Franka-Tabletop and Franka-DROID Tasks
- B.3.2 Detailed Franka-Tabletop and Franka-DROID Evaluation Results
- Appendix C RT-2-X vs. OpenVLA in BridgeData V2 Evaluations
- Appendix D Additional Experiments and Ablations
- D.1 OpenX Training Data Ablation Experiments
- D.2 Dual vs. Single Vision Encoder Experiments
- D.3 Fine-Tuned vs. Frozen Vision Encoder Experiments
- D.4 Additional Quantized Inference Experiments: Disentangling Policy Performance and Model Inference Speed
- Appendix E LIBERO Simulation Experiments
- E.1 LIBERO Simulation Experimental Setup
- E.2 LIBERO Simulation Experimental Results

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://openvla.github.io/
- 代码：https://github.com/openvla/openvla
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：https://huggingface.co/openvla/openvla-7b
- 权重核验深度：官方入口已定位；未下载、未加载验证
- arXiv首次/更新：2024/06/13 / 2024/09/05；当前HTML版本v3。
- 出版页面标注日期：2025/01/12；会议年份与论文集上线日期分开记录。
- 正式版题名：OpenVLA: An Open-Source Vision-Language-Action Model
- 正式版作者：Moo Jin Kim; Karl Pertsch; Siddharth Karamcheti; Ted Xiao; Ashwin Balakrishna; Suraj Nair; Rafael Rafailov; Ethan P Foster; Pannag R Sanketi; Quan Vuong; Thomas Kollar; Benjamin Burchfiel; Russ Tedrake; Dorsa Sadigh; Sergey Levine; Percy Liang; Chelsea Finn

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
