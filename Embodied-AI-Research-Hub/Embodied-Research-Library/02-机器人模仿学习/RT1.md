---
citekey: RT1
arxiv: '2212.06817'
checked: 2026-10-24
priority: 扩展12
source_level: 摘要与目录导读
tags: [具身智能, 论文学习]
---

# RT-1：面向大规模真实世界控制的机器人Transformer

## 1. 英文标题与中文译名

RT-1: Robotics Transformer for Real-World Control at Scale

RT-1：面向大规模真实世界控制的机器人Transformer（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Brohan, Anthony; Brown, Noah; Carbajal, Justice; Chebotar, Yevgen; Dabis, Joseph; Finn, Chelsea; Gopalakrishnan, Keerthana; Hausman, Karol; Herzog, Alex; Hsu, Jasmine; Ibarz, Julian; Ichter, Brian; Irpan, Alex; Jackson, Tomas; Jesmonth, Sally; Joshi, Nikhil J; Julian, Ryan; Kalashnikov, Dmitry; Kuang, Yuheng; Leal, Isabel; Lee, Kuang-Huei; Levine, Sergey; Lu, Yao; Malla, Utsav; Manjunath, Deeksha; Mordatch, Igor; Nachum, Ofir; Parada, Carolina; Peralta, Jodilyn; Perez, Emily; Pertsch, Karl; Quiambao, Jornell; Rao, Kanishka; Ryoo, Michael; Salazar, Grecia; Sanketi, Pannag; Sayed, Kevin; Singh, Jaspiar; Sontakke, Sumedh; Stone, Austin; Tan, Clayton; Tran, Huong; Vanhoucke, Vincent; Vega, Steve; Vuong, Quan; Xia, Fei; Xiao, Ted; Xu, Peng; Xu, Sichun; Yu, Tianhe; Zitkovich, Brianna。首次提交：2022/12/13。RSS 2023；正式会议论文。

[arXiv](https://arxiv.org/abs/2212.06817) · [正文HTML](https://arxiv.org/html/2212.06817) · [公开PDF](https://arxiv.org/pdf/2212.06817) · [正式来源](https://roboticsproceedings.org/rss19/p025.html)

正式DOI：10.15607/RSS.2023.XIX.025；arXiv DOI：10.48550/arXiv.2212.06817。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要与目录级。所列章节供后续精读定位，不表示已经精读这些正文段落。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

RT-1 探索当机器人收集更多不同任务的数据时，一个共享策略是否能持续获得更好的能力。模型接收图像历史与语言指令，把视觉信息压成较少的 Token，再用 Transformer 输出离散化的控制量。它的重要性在于把数据规模、多样性和模型容量放进同一个实验框架，而不是只在单一任务里调参数。对你来说，先理解“一个策略做许多任务”与“许多独立策略”的区别，再看数据多样性的消融。这也是理解后来的跨机器人数据和 VLA 模型路线的前置背景。

## 4. 通俗背景

多任务机器人策略能否随数据量和多样性扩展？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

多任务机器人策略能否随数据量和多样性扩展？

## 6. 已有方法的不足

任务各自训练缺乏共享；模型和数据多样性的作用未清楚量化。

## 7. 核心方法

语言条件视觉编码、Token压缩与Transformer离散动作预测。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。平台与任务覆盖有限，训练成本高。。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

图像历史与指令→离散手臂/底盘控制及终止等动作。

## 10. 公式、变量和推导思路

原文§5模型；本轮不转录未经逐式核对的公式。教学上将离散控制理解为条件分类。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

§4系统、§5模型数据、§6.5数据规模与多样性。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

真实机器人多任务数据；具体训练设备和评估分母需查§6.1，本卡不补填。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

作者展示数据规模和多样性有助于泛化；尚未在本轮逐格校验结果表。

## 14. 局限及待检验研究空白

平台与任务覆盖有限，训练成本高。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

RT1→OXE/RT-X→RT2/OpenVLA。

## 16. 本科毕设与博士价值

了解数据扩展路线，本科不复刻其大规模采集。

## 17. 阅读前置知识

监督分类、Transformer、行为克隆。

## 18. 精读与略读建议

优先精读：§4系统、§5模型数据、§6.5数据规模与多样性。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **更多数据必然更好？** 还看多样性与质量。
2. **输入只有语言吗？** 还有图像历史。
3. **Token压缩是权重量化吗？** 不是。
4. **能从零复刻吗？** 数据成本高。
5. **应关注哪项消融？** 数据量与多样性。

## 20. 卡住时补什么

监督分类、Transformer、行为克隆。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- RT-1: Robotics Transformer for Real-World Control at Scale
- Abstract
- 1 Introduction
- 2 Related Work
- 3 Preliminaries
- Robot learning.
- 4 System Overview
- 5 RT-1: Robotics Transformer
- 5.1 Model
- 5.2 Data
- 6 Experiments
- 6.1 Experimental Setup
- 6.2 Can RT-1 learn to perform a large number of instructions, and to generalize to new tasks, objects and environments?
- 6.3 Can we push the resulting model further by incorporating heterogeneous data sources such as simulation or data from different robots?
- 6.4 How do various methods generalize long-horizon robotic scenarios?
- 6.5 How do generalization metrics change with varying amounts of data quantity and data diversity?
- 7 Conclusions, Limitations and Future Work
- Acknowledgments
- References
- Appendix
- Appendix A Author Contributions
- Appendix B Model Card
- Appendix C Model and Data
- C.1 Model inference
- C.2 Data collection at scale.
- C.3 Model Selection at Scale
- C.4 Data collection process
- Appendix D Experiments
- D.1 Evaluation Details
- D.2 Heterogeneous Data
- D.3 Long-horizon Evaluation Details
- D.4 Model Ablations
- D.5 Summary and Analysis

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://robotics-transformer1.github.io/
- 代码：https://github.com/google-research/robotics_transformer
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：未核得独立权重，或不适用
- 权重核验深度：本轮未核得独立权重；综述/算子/数据工作可能不适用
- arXiv首次/更新：2022/12/13 / 2023/08/11；当前HTML版本v2。
- 出版页面标注日期：2023/07/10；会议年份与论文集上线日期分开记录。
- 正式版题名：RT-1: Robotics Transformer for Real-World Control at Scale
- 正式版作者：Anthony Brohan; Noah Brown; Justice Carbajal; Yevgen Chebotar; Joseph Dabis; Chelsea Finn; Keerthana Gopalakrishnan; Karol Hausman; Alexander Herzog; Jasmine Hsu; Julian Ibarz; Brian Ichter; Alex Irpan; Tomas Jackson; Sally Jesmonth; Nikhil Joshi; Ryan Julian; Dmitry Kalashnikov; Yuheng Kuang; Isabel Leal; Kuang-Huei Lee; Sergey Levine; Yao Lu; Utsav Malla; Deeksha Manjunath; Igor Mordatch; Ofir Nachum; Carolina Parada; Jodilyn Peralta; Emily Perez; Karl Pertsch; Jornell Quiambao; Kanishka Rao; Michael S Ryoo; Grecia Salazar; Pannag R Sanketi; Kevin Sayed; Jaspiar Singh; Sumedh Sontakke; Austin Stone; Clayton Tan; Huong Tran; Vincent Vanhoucke; Steve Vega; Quan H Vuong; Fei Xia; Ted Xiao; Peng Xu; Sichun Xu; Tianhe Yu; Brianna Zitkovich

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
