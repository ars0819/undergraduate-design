---
title: 论文候选索引
type: papers
updated: 2026-11-06
---

# Papers Index

> **这不是“已读论文清单”，也不是发表质量排名。**  
> 条目来自既有本科毕设资料的候选范围（2026-11-06），本库未逐篇重新核验 arXiv 版本/录用状态；引用前以原文为准。  
> 每篇必须与当前研究问题或实验有关；不按团队名气自动收录。

## 当前研究问题

见 [`Research_Question.md`](../04_Projects/Undergraduate_Thesis/Research_Question.md)：固定开源 VLA + LIBERO，测量新动作块延迟/显存/动作质量/闭环成功率，再做一种优化。

## 候选清单

| 优先级 | 论文 | 入口 | 与当前问题的关系 | 状态 |
|---|---|---|---|---|
| P0 | SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics | [arXiv 2506.01844](https://arxiv.org/abs/2506.01844) | 首选模型；读结构、动作生成、LIBERO 实验、推理成本 | 待读 |
| P0 | OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success | [arXiv 2502.19645](https://arxiv.org/abs/2502.19645) | 备用模型；动作块并行解码/连续动作头；资源与评测对照 | 待读 |
| P0 | LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning | [arXiv 2306.03310](https://arxiv.org/abs/2306.03310) | 任务套件、初态、成功判据；评测协议基础 | 待读 |
| P1 | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | [arXiv 2306.00978](https://arxiv.org/abs/2306.00978) | 量化方法；判断能否迁移到 VLA/动作头，而不是直接套 | 待读 |
| P1 | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | [arXiv 2211.10438](https://arxiv.org/abs/2211.10438) | 激活/权重离群值处理；真实内核/精度代价 | 待读 |
| P1 | SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models | [arXiv 2411.05007](https://arxiv.org/abs/2411.05007) | 低秩+4bit；与 diffusion/动作头的潜在关联 | 待读 |
| P1 | VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models | [arXiv 2609.25376](https://arxiv.org/abs/2609.25376) | 最直接：VLA 量化的闭环评测；读指标、任务、失败 | 待读 |
| P1 | QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models | [arXiv 2602.20309](https://arxiv.org/abs/2602.20309) | 量化校准；与动作头精度/部署相关 | 待读 |
| P2 | π0: A Vision-Language-Action Flow Model for General Robot Control | [arXiv 2410.24164](https://arxiv.org/abs/2410.24164) | 流匹配动作生成；理解动作头与推理迭代 | 待读 |
| P2 | π0.5: a Vision-Language-Action Model with Open-World Generalization | [arXiv 2504.16054](https://arxiv.org/abs/2504.16054) | 泛化/开放世界；与本科部署主线关联较弱 | 扩展 |
| P2 | Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning | [arXiv 2606.11087](https://arxiv.org/abs/2606.11087) | 流策略的测试时引导；与“推理时计算”相关，但不是当前基线 | 扩展 |

> 说明：`2609.25376`、`2602.20309`、`2606.11087` 等编号来自既有资料，当前年份/编号看起来较新；本库未独立核验其元数据，引用前必须打开 arXiv 页面确认。

## 阅读顺序（建议）

1. SmolVLA：先读问题、方法图、LIBERO 实验、推理成本。
2. LIBERO：任务/初态/成功判据，避免“只看模型不看评测”。
3. OpenVLA-OFT：当作备用/对照，不并行搭环境。
4. VLAQuantBench：学习闭环量化评测的指标与失败分类。
5. AWQ/SmoothQuant/SVDQuant：只读与“真实内核/动作头/闭环”相关的部分。
6. π0/π0.5：理解 flow/action expert；不先追大规模训练。

## 纳入规则

- 每篇必须回答：它改变了我哪个 RQ、基线或测量？
- 不把“来自 Han Lab/Levine/许华哲/Zhiyuan Zhou”本身当纳入理由。
- 预印本不等于正式发表；不编造 venue/影响因子。
- 没读全文就只写“待读”，不写声称的架构细节。

## 论文笔记

读完后在 `03_Research/Literature/` 建独立文件夹，用 [`Paper_Note.md`](../07_Templates/Paper_Note.md)。  
Zotero 继续管理条目/PDF/引用；本库不修改 Zotero 数据库。
