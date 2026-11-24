---
citekey: VLASH
arxiv: '2512.01031'
checked: 2026-10-24
priority: 核心20
source_level: 正文定向阅读
tags: [具身智能, 论文学习]
---

# VLASH：通过未来状态感知的异步推理实现实时视觉语言动作模型

## 1. 英文标题与中文译名

VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference

VLASH：通过未来状态感知的异步推理实现实时视觉语言动作模型（学习用译名，不冒充作者正式中文题名）。

## 2. 类型、作者与发表信息

方法/系统论文。作者：Tang, Jiaming; Sun, Yufei; Zhao, Yilong; Yang, Shang; Lin, Yujun; Zhang, Zhuoyang; Hou, James; Lu, Yao; Liu, Zhijian; Han, Song。首次提交：2025/11/30。IROS 2026（Han Lab作者页确认；IEEE最终卷页待核）；作者确认会议接收/发表；最终出版记录待核。

[arXiv](https://arxiv.org/abs/2512.01031) · [正文HTML](https://arxiv.org/html/2512.01031) · [公开PDF](https://arxiv.org/pdf/2512.01031)

正式DOI：本轮未核得，不猜测；arXiv DOI：10.48550/arXiv.2512.01031。两者不同。会议不填SCI分区/影响因子；期刊指标本轮未取得官方年度数据，留空。

阅读范围：摘要、结构及卡片所列方法/实验位置的定向阅读；不声称逐字精读完整全文。正文获取成功不等于全部内容均已读完。PDF解析见核验报告。

## 3. 中文摘要释义（非逐句译文）

机器人一边执行旧动作、一边计算新动作时，新动作开始执行的状态已经不同于拍照时的状态。VLASH 用已知旧动作把机器人自身状态向前推到预计接管时刻，再让策略依据这个状态生成动作。仅在测试时换状态通常不够，作者还在微调中固定旧图像、同步偏移状态与动作目标，促使模型真正使用状态信息。它另外研究共享视觉编码和动作时间重采样。你应抓住其核心是预测与执行的时间对齐，不能把它理解成普通多线程，也不能把未来自身状态当作未来环境的完整预测。

## 4. 通俗背景

异步执行时如何避免旧观测对应的新动作在错误状态下接管？ 先画清“观察—决策—执行—新观察”的关系，再定位本论文改变了哪一步。这里的教学解释不替代作者原文。

## 5. 具体研究问题

异步执行时如何避免旧观测对应的新动作在错误状态下接管？

## 6. 已有方法的不足

朴素异步有状态错位；直接输入未来状态可能因策略不使用状态而无效；RTC推理引导有开销。

## 7. 核心方法

未来自身状态滚动预测 + 状态/动作一起偏移的微调数据增强；可共享观测编码；动作重采样调节执行速度。

## 8. 关键创新与成立条件

创新见上节；成立条件应与任务、数据和接口绑定。未来外界仍未知。这也是复现时需要检验的条件，而非保证。

## 9. 模型结构及输入输出

旧视觉o_t、预计接管状态s_(t+Δ)→对应未来动作块。

## 10. 公式、变量和推导思路

教学重写：ŝ_(t+Δ)=s_t+Σ a_i，仅适用于可直接累加的增量状态；绝对动作使用对应目标。训练配对(o_t,s_(t+δ),A_(t+δ))，不把视觉也偏移。

凡标注“教学重写”的式子均非原文编号，不声称逐式复刻推导。

## 11. 关键图表、算法与可复核实验

III时间错位；IV-A至IV-D；图3对齐、图4共享观测、图5动作重采样；V-A Kinetix动态实验。

复核时固定基础策略、任务、数据划分及随机种子，仅改变对应模块；同时记录失败例。实验建议是我的设计，不是作者额外实验。

## 12. 数据、规模和设备

Kinetix与LIBERO，真实R1 Lite和SO101；当前正文V-B用RTX5090笔记本，30Hz执行；不是你的5070。

未明确列出的设备型号、训练时长和版本，不从其他论文补填。

## 13. 指标与作者结果

最新页面摘要为最高11.8倍反应延迟改善；官网旧摘要17.4倍不能混用。Kinetix延迟4步示例81.7% vs朴素异步51.2%。

## 14. 局限及待检验研究空白

未来外界仍未知；接触/打滑使状态外推偏差扩大；低显存微调TODO不能视为已验证。

我的研究假设：在固定成功率约束下，是否还能降低完整闭环延迟？该问题尚未由本资料库实验验证。

## 15. 论文关系

ACT动作块→SmolVLA朴素异步→RTC→VLASH；与ForeAct预测内容不同。

## 16. 本科毕设与博士价值

优先研究概念和轻量延迟注入实验；真机交付后再做车臂状态与执行队列适配。

## 17. 阅读前置知识

动作分块、时间戳、队列、状态转移、监督微调。

## 18. 精读与略读建议

优先精读：III时间错位；IV-A至IV-D；图3对齐、图4共享观测、图5动作重采样；V-A Kinetix动态实验。

第一遍可略读长相关工作、形式证明和完整超参表；准备复现时必须回看设备、数据、动作定义及附录，不能永久略过。完整结构导览见文末。

## 19. 五个思考问题与参考答案

1. **必须微调吗？** 论文指出仅替换状态不足。
2. **预测未来图像吗？** 核心不是。
3. **action quantization是INT4吗？** 不是，是动作时间重采样。
4. **外推总正确吗？** 接触可破坏。
5. **可以照抄官网倍率吗？** 须绑定论文版本。

## 20. 卡住时补什么

动作分块、时间戳、队列、状态转移、监督微调。。用NumPy或PyTorch打印一次输入、隐藏量、输出形状；公式先用小维度数值验证。无需先学完整最优化课程再阅读。具体练习见00与07目录。

## 原文结构导览

以下标题由实际取得HTML抽取；原版/扩展版/后续版本章节可能不同，需与当前PDF对照。

- VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference
- Abstract
- I INTRODUCTION
- II RELATED WORK
- Vision-Language-Action Models (VLAs)
- Asynchronous VLA Inference
- III BACKGROUND AND MOTIVATION
- Action chunking policy
- Prediction and execution intervals
- Asynchronous inference and interval misalignment
- IV VLASH
- IV-A Future State Awareness
- IV-B Fine-tuning with Offsets to States and Actions
- IV-C Efficient Fine-tuning with Shared Observation
- IV-D Action Quantization
- V EXPERIMENTS
- V-A Simulated Evaluation
- V-A 1 Kinetix
- Experimental Setup
- Results
- V-A 2 LIBERO
- Experimental Setup
- Results
- V-B Real-World Evaluation
- V-B 1 Success Rate and Robustness
- Experimental Setup
- Results
- V-B 2 Reaction Speed
- Experimental Setup
- Results
- V-C Fine-tuning Efficiency
- Experimental Setup
- Results
- VI CONCLUSION
- References
- Appendix A Experimental Details
- Appendix B Supplementary Demo Video
- Appendix C Architectural Modifications

## 术语与来源边界

VLA：视觉语言动作模型；policy：策略；observation：观测；action chunk：动作块；latency：延迟。更多词条见术语词典。

作者结论：第13节；教学解释：第3、4、10、19节；研究建议/推测：第14、16节。未生成未经确认授权的全文译稿。


## 项目、代码与权重核验

- 项目：https://github.com/mit-han-lab/vlash
- 代码：https://github.com/mit-han-lab/vlash
- 代码核验深度：作者正文/官方项目关联；链接登记，未安装运行
- 权重：https://huggingface.co/mit-han-lab/vlash-pi05-libero-async5
- 权重核验深度：官方入口已定位；未下载、未加载验证
- arXiv首次/更新：2025/11/30 / 2026/07/26；当前HTML版本v2。
- 出版页面标注日期：未提取；会议年份与论文集上线日期分开记录。
- 正式版题名：VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference
- 正式版作者：未从出版元数据抽取；上方保留arXiv作者

代码与权重入口登记不代表安装、文件完整性或推理性能验证；尚未下载任何模型权重。
