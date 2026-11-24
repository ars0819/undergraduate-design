---
citekey: DiffusionPolicy
arxiv: '2303.04137'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# 扩散策略：通过动作扩散学习视觉运动策略

## 1. 英文标题与中文译名

Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

扩散策略：通过动作扩散学习视觉运动策略（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Chi, Cheng; Xu, Zhenjia; Feng, Siyuan; Cousineau, Eric; Du, Yilun; Burchfiel, Benjamin; Tedrake, Russ; Song, Shuran。首次提交：2023/03/07。RSS 2023；IJRR扩展版2024（作者主页核实）；正式会议论文；扩展期刊版。

[arXiv](https://arxiv.org/abs/2303.04137) · [正文HTML](https://arxiv.org/html/2303.04137) · [公开PDF](https://arxiv.org/pdf/2303.04137) · [正式来源](https://roboticsproceedings.org/rss19/p026.html)

正式DOI：10.15607/RSS.2023.XIX.026；arXiv DOI：10.48550/arXiv.2303.04137。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

机器人在同一个场景下可能有多种正确动作，例如从物体左边或右边绕过去。直接用均方误差预测一个动作，可能得到两条可行路线的平均，反而撞上障碍。Diffusion Policy 把动作序列当作需要生成的数据：训练时给示范动作加噪声，让网络学会去噪；运行时从噪声逐步生成动作块，再执行其中一段并重新观察。这把多模态生成与滚动时域控制连接起来。它不是生成未来图像，而是生成未来动作，也是后来许多流匹配 VLA 动作头的重要前置知识。

## 4. 通俗背景

如何表达多峰动作分布并生成连续、稳定的视觉控制序列？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

如何表达多峰动作分布并生成连续、稳定的视觉控制序列？

## 6. 已有方法的不足

简单回归可能平均掉多个模式；隐式能量策略训练和采样复杂；单步预测缺乏序列一致性。

## 7. 核心方法

条件去噪生成动作序列，视觉特征作条件；滚动时域执行；比较CNN与时序Transformer。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。迭代去噪有延迟。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

历史视觉/状态→条件特征→去噪网络→动作块→仅执行前段后重规划。

## 10. 公式、变量和推导思路

教学重写：E||ε−εθ(A_k,o,k)||²；ε是注入噪声，A_k是第k噪声级动作。先读DDPM，再理解条件o和动作维度替换；不是论文原编号。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

当前获取的是扩展版：§2、§3.4、§4.1、§5、§7；图1比较策略表示、图2给出框架。RSS版与IJRR扩展版章节不可混用。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

多种模拟操纵基准与真实机器人；扩展版增加双臂任务。各环境输入和控制模式不完全相同。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

扩展版摘要报告4类基准15任务平均提升46.9%；这是作者定义的相对改进口径，不是成功率增加46.9个百分点。

## 14. 局限及待检验研究空白

迭代去噪有延迟；离线动作误差不能充分预测闭环成功；多模态生成不等于任务规划。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

DDPM→Diffusion Policy→Consistency Policy/RTC；pi0采用流匹配，不能直接视为同一个训练目标。

## 16. 本科毕设与博士价值

适合作为小任务基线，分析采样步数、执行时域和反馈延迟的耦合。

## 17. 阅读前置知识

高斯分布、MSE、条件概率、PyTorch批次和时间维度。

## 18. 精读与略读建议

优先精读：当前获取的是扩展版：§2、§3.4、§4.1、§5、§7；图1比较策略表示、图2给出框架。RSS版与IJRR扩展版章节不可混用。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **生成的是图片吗？** 主要是动作。
2. **为何不平均两条路线？** 平均动作可能不可行。
3. **预测长度等于执行长度吗？** 不一定。
4. **步数减半必然整机快一倍吗？** 否。
5. **该先读哪部分？** 图1、图2和§2。

## 20. 卡住时补什么

高斯分布、MSE、条件概率、PyTorch批次和时间维度。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- Abstract
- keywords
- 1 Introduction
- 2 Diffusion Policy Formulation
- 2.1 Denoising Diffusion Probabilistic Models
- 2.2 DDPM Training
- 2.3 Diffusion for Visuomotor Policy Learning
- 3 Key Design Decisions
- 3.1 Network Architecture Options
- 3.2 Visual Encoder
- 3.3 Noise Schedule
- 3.4 Accelerating Inference for Real-time Control
- 4 Intriguing Properties of Diffusion Policy
- 4.1 Model Multi-Modal Action Distributions
- 4.2 Synergy with Position Control
- 4.3 Benefits of Action-Sequence Prediction
- 4.4 Training Stability
- 4.5 Connections to Control Theory
- 5 Evaluation
- 5.1 Simulation Environments and datasets
- 5.2 Evaluation Methodology
- 5.3 Key Findings
- 5.4 Ablation Study
- 6 Realworld Evaluation
- 6.1 Realworld Push-T Task
- 6.2 Mug Flipping Task
- 6.3 Sauce Pouring and Spreading
- 7 Realworld Bimanual Tasks
- 7.1 Observation and Action Spaces
- 7.2 Teleoperation
- 7.3 Bimanual Egg Beater
- 7.4 Bimanual Mat Unrolling
- 7.5 Bimanual Shirt Folding.
- 8 Related Work
- 9 Limitations and Future Work
- 10 Conclusion
- 11 Acknowledgement
- funding.
- References
- Appendix A Diffusion Policy Implementation Details
- A.1 Normalization
- A.2 Rotation Representation
- A.3 Image Augmentation
- A.4 Hyperparameters
- A.5 Data Efficiency
- Appendix B Additional Ablation Results
- B.1 Observation Horizon
- B.2 Performance Improvement Calculation
- Appendix C Realworld Task Details
- C.1 Push-T
- C.1.1 Demonstrations
- C.1.2 Evaluation
- C.2 Sauce Pouring and Spreading
- C.2.1 Demonstrations
- C.2.2 Evaluation
- Appendix D Realworld Setup Details
- D.0.1 UR5 robot station
- D.1 Franka Robot Station
- D.2 Initial and Final States of Bimanual Tasks

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://diffusion-policy.cs.columbia.edu/
- 代码：https://github.com/columbia-ai-robotics/diffusion_policy
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：未核得独立权重，或不适用
- 权重核验深度：本轮未核得独立权重；综述/算子/数据工作可能不适用
- arXiv首次/更新：2023/03/07 / 2024/03/14；当前HTML版本v5。
- 出版页面标注日期：2023/07/10；会议年份与论文集上线日期分开记录。
- 正式版题名：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- 正式版作者：Cheng Chi; Siyuan Feng; Yilun Du; Zhenjia Xu; Eric Cousineau; Benjamin CM Burchfiel; Shuran Song

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
