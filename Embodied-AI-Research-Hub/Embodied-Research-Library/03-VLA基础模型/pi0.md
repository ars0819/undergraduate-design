---
citekey: pi0
arxiv: '2410.24164'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# π₀：面向通用机器人控制的视觉语言动作流模型

## 1. 英文标题与中文译名

$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control

π₀：面向通用机器人控制的视觉语言动作流模型（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Black, Kevin; Brown, Noah; Driess, Danny; Esmail, Adnan; Equi, Michael; Finn, Chelsea; Fusai, Niccolo; Groom, Lachy; Hausman, Karol; Ichter, Brian; Jakubczak, Szymon; Jones, Tim; Ke, Liyiming; Levine, Sergey; Li-Bell, Adrian; Mothukuri, Mohith; Nair, Suraj; Pertsch, Karl; Shi, Lucy Xiaoyang; Tanner, James; Vuong, Quan; Walling, Anna; Wang, Haohuan; Zhilinsky, Ury。首次提交：2024/10/31。RSS 2025；正式会议论文。

[arXiv](https://arxiv.org/abs/2410.24164) · [正文HTML](https://arxiv.org/html/2410.24164) · [公开PDF](https://arxiv.org/pdf/2410.24164) · [正式来源](https://roboticsproceedings.org/rss21/p010.html)

正式DOI：10.15607/RSS.2025.XXI.010；arXiv DOI：10.48550/arXiv.2410.24164。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

π₀ 把预训练视觉语言模型的语义知识与连续动作生成结合起来。视觉、语言和机器人状态提供上下文，一个动作专家通过流匹配把随机噪声逐步变成动作块。这样既能利用互联网预训练知识，也能表达高频、灵巧的连续操作，而不必逐个维度输出离散动作词。论文以多种单臂、双臂和移动操作平台数据训练，并通过后训练适应复杂任务。阅读时要特别区分预训练带来的广泛能力与特定任务示范带来的精细技能，不能把演示视频当作无条件零样本能力的证明。

## 4. 通俗背景

语义预训练与高频连续机器人控制怎样结合？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

语义预训练与高频连续机器人控制怎样结合？

## 6. 已有方法的不足

离散自回归动作有时序和吞吐瓶颈；单任务策略泛化受限。

## 7. 核心方法

PaliGemma视觉语言主干与流匹配动作专家联合建模；跨机器人预训练和任务后训练。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。多步动作专家有延迟。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

多路RGB、语言、机器人状态及带噪动作→向量场→动作块。

## 10. 公式、变量和推导思路

教学重写：L_FM=E||vθ(Aτ,o,τ)−uτ||²，Aτ是噪声与示范间插值；方向uτ须与作者τ约定一致。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

IV模型、V训练阶段、VI实验；附录A-B结构与A-D推理。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

约3.3B体系，跨单臂/双臂/移动平台；数据和设备成本不能由参数量单独推出。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

按任务完成能力与后训练效果比较；零样本与适配任务分别解读。

## 14. 局限及待检验研究空白

多步动作专家有延迟；大规模私有机器人数据限制完全复刻；新平台需动作归一化。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

Diffusion Policy连续生成→pi0流匹配；SmolVLA是更小的相关路线；pi0.5改进泛化训练。

## 16. 本科毕设与博士价值

读结构以理解动作头加速；12GB上整模型与多相机实测前不承诺可运行。

## 17. 阅读前置知识

Transformer、条件生成、流匹配、状态/动作表示。

## 18. 精读与略读建议

优先精读：IV模型、V训练阶段、VI实验；附录A-B结构与A-D推理。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **流匹配等于自回归？** 不是。
2. **大VLM自己输出电机信号？** 通过动作专家。
3. **跨机器人等于免适配？** 不保证。
4. **训练阶段要区分吗？** 必须。
5. **先复现全训练吗？** 不适合当前预算。

## 20. 卡住时补什么

Transformer、条件生成、流匹配、状态/动作表示。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- π 0 \pi_{0} : A Vision-Language-Action Flow Model for General Robot Control Thanks: Physical Intelligence, San Francisco, California, USA. Correspondance to: research@physicalintelligence.company
- Abstract
- I Introduction
- II Related Work
- III Overview
- IV The π 0 \pi_{0} Model
- V Data Collection and Training Recipe
- V-A Pre-training and post-training
- V-B Language and high-level policies
- V-C Robot system details
- VI Experimental Evaluation
- VI-A Evaluating the base model
- VI-B Following language commands
- VI-C Learning new dexterous tasks
- VI-D Mastering complex multi-stage tasks
- VII Discussion, Limitations, and Future Work
- Acknowledgements
- References
- Appendix A
- A-A Contributions
- A-B Model Architecture Details
- A-C Non-VLM Baseline Architecture
- A-D Inference
- A-E Evaluation Details

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://www.physicalintelligence.company/blog/pi0
- 代码：https://github.com/Physical-Intelligence/openpi
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：openpi官方README权重表（具体checkpoint按任务选择）
- 权重核验深度：官方入口已定位；未下载、未加载验证
- arXiv首次/更新：2024/10/31 / 2026/01/08；当前HTML版本v4。
- 出版页面标注日期：2025/06/21；会议年份与论文集上线日期分开记录。
- 正式版题名：π₀: A Vision-Language-Action Flow Model for General Robot Control
- 正式版作者：Kevin Black; Noah Brown; Danny Driess; Adnan Esmail; Michael Robert Equi; Chelsea Finn; Niccolo Fusai; Lachy Groom; Karol Hausman; Brian Ichter; Szymon Jakubczak; Tim Jones; Liyiming Ke; Sergey Levine; Adrian Li-Bell; Mohith Mothukuri; Suraj Nair; Karl Pertsch; Lucy Xiaoyang Shi; Laura Smith; James Tanner; Quan Vuong; Anna Walling; Haohuan Wang; Ury Zhilinsky

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
