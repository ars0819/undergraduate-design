---
citekey: RT2
arxiv: '2307.15818'
checked: 2026-10-24
priority: 扩展12
source_level: 摘要与目录导读
tags: [具身智能, 论文学习]
---

# RT-2：视觉语言动作模型将网络知识迁移到机器人控制

## 1. 英文标题与中文译名

RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

RT-2：视觉语言动作模型将网络知识迁移到机器人控制（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Brohan, Anthony; Brown, Noah; Carbajal, Justice; Chebotar, Yevgen; Chen, Xi; Choromanski, Krzysztof; Ding, Tianli; Driess, Danny; Dubey, Avinava; Finn, Chelsea; Florence, Pete; Fu, Chuyuan; Arenas, Montse Gonzalez; Gopalakrishnan, Keerthana; Han, Kehang; Hausman, Karol; Herzog, Alexander; Hsu, Jasmine; Ichter, Brian; Irpan, Alex; Joshi, Nikhil; Julian, Ryan; Kalashnikov, Dmitry; Kuang, Yuheng; Leal, Isabel; Lee, Lisa; Lee, Tsang-Wei Edward; Levine, Sergey; Lu, Yao; Michalewski, Henryk; Mordatch, Igor; Pertsch, Karl; Rao, Kanishka; Reymann, Krista; Ryoo, Michael; Salazar, Grecia; Sanketi, Pannag; Sermanet, Pierre; Singh, Jaspiar; Singh, Anikait; Soricut, Radu; Tran, Huong; Vanhoucke, Vincent; Vuong, Quan; Wahid, Ayzaan; Welker, Stefan; Wohlhart, Paul; Wu, Jialin; Xia, Fei; Xiao, Ted; Xu, Peng; Xu, Sichun; Yu, Tianhe; Zitkovich, Brianna。首次提交：2023/07/28。CoRL 2023；PMLR229；正式会议论文。

[arXiv](https://arxiv.org/abs/2307.15818) · [正文HTML](https://arxiv.org/html/2307.15818) · [公开PDF](https://arxiv.org/pdf/2307.15818) · [正式来源](https://proceedings.mlr.press/v229/zitkovich23a.html)

正式DOI：本轮未核得，不猜测；arXiv DOI：10.48550/arXiv.2307.15818。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要与目录级。所列章节供后续精读定位，不表示已经精读这些正文段落。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

RT-2 把机器人动作编码成文本风格的 Token，让预训练视觉语言模型既能处理网络图文任务，也能学习机器人控制。这样模型有机会把视觉语义知识迁移到选择物体、理解新指令等任务中。它不是先让聊天模型写一个计划，再由独立控制器完成全部动作，而是探索端到端的多模态动作预测。理解这项工作时，要分开语义泛化与运动技能泛化：能辨认新的图标，并不意味着掌握了训练中没有的接触操作。对个人复现而言，模型和数据开放程度也比单看演示效果更重要。

## 4. 通俗背景

网络视觉语言知识能否进入端到端机器人动作策略？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

网络视觉语言知识能否进入端到端机器人动作策略？

## 6. 已有方法的不足

仅机器人示范不足以覆盖丰富语义；高层规划与低层策略往往分离。

## 7. 核心方法

VLM与机器人轨迹/图文任务共同微调，动作作为Token。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。代码/完整权重未核实公开，昂贵模型不适合作为首个实现目标。。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

图像、指令→VLM→动作Token→连续动作。

## 10. 公式、变量和推导思路

教学重写的条件Token交叉熵可参考OpenVLA卡片；这里不冒充原文公式编号。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

§3.2动作微调、§3.3实时推理、§4.2新能力、§5局限。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

作者摘要列约6000次评估；大VLM版本不适合默认本机部署。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

报告新物体和语义推理能力；不等于训练外所有运动技能泛化。

## 14. 局限及待检验研究空白

代码/完整权重未核实公开，昂贵模型不适合作为首个实现目标。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

RT1→RT2→OpenVLA。

## 16. 本科毕设与博士价值

历史路线阅读，帮助区分语义和控制泛化。

## 17. 阅读前置知识

VLM、Token、监督微调、泛化划分。

## 18. 精读与略读建议

优先精读：§3.2动作微调、§3.3实时推理、§4.2新能力、§5局限。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **文字动作是自然语言解释吗？** 是编码。
2. **懂物体就会抓吗？** 不保证。
3. **与高层LLM规划完全相同吗？** 不是。
4. **作者顺序有差异吗？** arXiv与PMLR顺序不同。
5. **本科先复刻吗？** 建议先读后用开放小模型。

## 20. 卡住时补什么

VLM、Token、监督微调、泛化划分。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- Abstract
- 1 Introduction
- 2 Related Work
- 3 Vision-Language-Action Models
- 3.1 Pre-Trained Vision-Language Models
- 3.2 Robot-Action Fine-tuning
- 3.3 Real-Time Inference
- 4 Experiments
- 4.1 How does RT-2 perform on seen tasks and more importantly, generalize over new objects, backgrounds, and environments?
- 4.2 Can we observe and measure any emergent capabilities of RT-2?
- 4.3 How does the generalization vary with parameter count and other design decisions?
- 4.4 Can RT-2 exhibit signs of chain-of-thought reasoning similarly to vision-language models?
- 5 Limitations
- 6 Conclusions
- Acknowledgments
- References
- Appendix A Contributions
- Appendix B Datasets
- Appendix C Baselines
- Appendix D VLMs for RT-2
- Appendix E Training Details
- Appendix F Evaluation Details
- F.1 Evaluation Scenarios
- F.2 Evaluation Instructions
- Appendix G Example Failure Cases
- Appendix H Quantitative Experimental Results
- H.1 Overall Performance, for Section 4.1
- H.2 Emergent Evaluation, for Section 4.2
- H.3 Size and Training Ablations, for Section 4.3
- Appendix I Additional Chain-Of-Thought Reasoning Results

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://robotics-transformer2.github.io/
- 代码：未核得独立官方实现
- 代码核验深度：本轮未核得独立官方实现；不等于未开源
- 权重：未核得独立权重，或不适用
- 权重核验深度：本轮未核得独立权重；综述/算子/数据工作可能不适用
- arXiv首次/更新：2023/07/28 / 2023/07/28；当前HTML版本v1。
- 出版页面标注日期：2023/12/02；会议年份与论文集上线日期分开记录。
- 正式版题名：RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- 正式版作者：Brianna Zitkovich; Tianhe Yu; Sichun Xu; Peng Xu; Ted Xiao; Fei Xia; Jialin Wu; Paul Wohlhart; Stefan Welker; Ayzaan Wahid; Quan Vuong; Vincent Vanhoucke; Huong Tran; Radu Soricut; Anikait Singh; Jaspiar Singh; Pierre Sermanet; Pannag R. Sanketi; Grecia Salazar; Michael S. Ryoo; Krista Reymann; Kanishka Rao; Karl Pertsch; Igor Mordatch; Henryk Michalewski; Yao Lu; Sergey Levine; Lisa Lee; Tsang-Wei Edward Lee; Isabel Leal; Yuheng Kuang; Dmitry Kalashnikov; Ryan Julian; Nikhil J. Joshi; Alex Irpan; Brian Ichter; Jasmine Hsu; Alexander Herzog; Karol Hausman; Keerthana Gopalakrishnan; Chuyuan Fu; Pete Florence; Chelsea Finn; Kumar Avinava Dubey; Danny Driess; Tianli Ding; Krzysztof Marcin Choromanski; Xi Chen; Yevgen Chebotar; Justice Carbajal; Noah Brown; Anthony Brohan; Montserrat Gonzalez Arenas; Kehang Han

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
